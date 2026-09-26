"""
analysis/dls/distributions.py
=============================

Distribution methods: recover a full decay-rate / size distribution rather than a
few parameters.

We invert the FIELD ACF g1 (not |g1|^2) on a grid of decay rates Gamma_n
(equivalently hydrodynamic radii Rh_n):

    g1(tau_m)  ~  sum_n  x_n exp(-Gamma_n tau_m)             (A x)_m

with a single-Gamma kernel A[m, n] = exp(-Gamma_n tau_m). This is the accepted-
standard DLS distribution inversion (Berne & Pecora 2000 Sec. 8.11 + App. 4.C):
g1 is recovered from the measured correlogram through the Siegert relation
g2 - 1 = beta |g1|^2 (Chu 1991), so the fit lives in g1-space and keeps the full
cross-term structure of a multimodal g1. (The older diagonal form
|g1|^2 ~ sum_n x_n exp(-2 Gamma_n tau_m) is the "cross-terms-negligible"
approximation, Liénard et al. 2022 Eq. 11 -- exact only for a single narrow mode;
applied to a multimodal sample it fabricates a phantom peak at the mean decay
rate.) We solve for the non-negative weights
x_n, which are the intensity-weighted contributions (g1 amplitudes normalize with
the scattered intensity).

  NNLS      : min ||M^^.5 (A x - y)||^2          s.t. x >= 0     (no smoothing)
  CONTIN    : min ||M^^.5 (A x - y)||^2 + alpha^2 ||L x||^2   s.t. x >= 0
              with L the second-difference operator (Provencher's regularizor).
  Lognormal : a single-mode parametric distribution fit through the same A.

Statistically-weighted residuals (the M above). Recovering g1 = sqrt((g2-1)/beta)
rectifies zero-mean long-lag noise into a positive pedestal that an UNWEIGHTED
non-negative fit would absorb as spurious large-Rh weight. We therefore weight the
residual by the delta-method inverse-variance of the recovered g1: propagating a
g2-1 noise of per-channel SD sigma through the sqrt gives Var(g1) ~ sigma^2 /
(4 beta^2 g1^2), so the per-lag weight is w(tau) proportional to
g1(tau)^2 / sigma(tau)^2 -- long-lag / clipped channels get ~0 weight, which
suppresses the pedestal AND keeps the Provencher F-test residual (Provencher 1982a
Eq. 3.9 is the weighted V) honest.

TWO WEIGHTING FORMS, requested per fit through `weighting` and recorded on the result
(`DISTRIBUTION_WEIGHTINGS` / `RECORDED_DISTRIBUTION_WEIGHTINGS`;
`DISTRIBUTION_WEIGHTING_DEFAULT` is what `weighting=None` means, and it is the sourced
'biganzoli' form). Beside the form, a result records WHAT THAT WEIGHT READ -- which
per-channel sample time and which run-duration form the variance was evaluated on
(`DistributionResult.sample_time_provenance` / `run_duration_form`, the cumulant result's
vocabulary). An UNWEIGHTED fit is deliberately NOT among the forms: that is the pedestal
regression above.

  'g1-squared' takes Var(g2-1) as constant across channels, which leaves w = y^2 --
  the recovered g1, squared. It is the OLDER form, kept selectable: it needs no input
  beyond the correlogram -- no sample time, no duration, no count rate -- and it is the
  form to select to reproduce a result reported under it.

  'biganzoli' is the sourced diagonal weight, w proportional to g1^2 / sigma^2, with
  BOTH factors evaluated on a fitted model and NEITHER on the noisy observation:

    * sigma is the relative per-channel SD of g2-1 of Biganzoli and Ferri (2018)
      Eq. (9), exact in Gamma, through the one implementation of that equation in the
      tree (`utilities._biganzoli_sd`, reached by `cumulants._sourced_sigma`, so both
      fitters read one variance at one evaluation point). That point is the Gamma and
      beta of an order-1 uniform cumulant fit -- Koppel's prescription, the prose after
      his Eq. 26, printed p. 4818: to keep the variance "as general as possible,
      applicable for any distribution of decay rates", |g1(tau_j)| in it is approximated
      "with the best fit single exponential". sigma stays at ONE exponential because
      Eq. (9) is a closed form for one; a variance taking a general correlation function
      is not implemented here. Recorded as 'biganzoli-full' when the measurement carries
      a committed mean count rate (Eq. (9)'s two shot-noise brackets enter) and
      'biganzoli-classical' when it does not (the classical infinite-count-rate limit):
      the weight never REQUIRES a count rate, on any instrument. A fit that ran in that
      limit carries `NO_COUNT_RATE_NOTE`, the one passive sentence saying which input was
      absent and what committing it would buy.

    * g1 is the fitted field correlation A x0 of a FIRST PASS by the same arm under the
      'g1-squared' weight, at the same selector or the same fixed alpha -- where Maier,
      Marth, Honerkamp and Weese (1999) evaluate their error model (p. 4674): an
      intermediate spectrum computed "assuming that the statistical errors are
      independent and have the same variance", one pass being enough because the
      estimates "hardly changed after the first iteration" (p. 4675). Only the
      EVALUATION POINT is Maier's. Maier fits g2 directly (his Eq. 14 carries no square
      root), so the delta-method factor g1^2 is this program's, from its g1-space fit.
      The first pass's own result is not reported, and the fit costs about twice as long.

THE WEIGHT'S SCALE, and the anchor. The overall scale cancels in the NNLS argmin, in
the lognormal arm's OBJECTIVE (`absolute_sigma=False`) and at any FIXED alpha. It does
NOT cancel in the alpha selection: rescaling w by c reparametrizes the Tikhonov objective
as alpha -> alpha/sqrt(c), so the selected alpha slides by sqrt(c) across the fixed
[alpha_min, alpha_max] sweep -- and w = |g2-1-baseline|/beta makes the per-run beta
exactly such a rescale (a beta-halving is c = 2, a slide of sqrt(2) = 1.41x, about one
step of the default 40-point six-decade sweep, 1.43x). The automatic selectors track the
shift, so the recovered distribution is near-invariant until the tracked optimum
reaches a sweep bound (flagged at either end: alpha_at_ceiling / alpha_at_floor);
a USER-FIXED alpha does not compensate at all. A rescale is therefore a SLIDE of the
sweep window along the solution path, and `_biganzoli_sd` returns a relative shape in
arbitrary units -- so the sourced weight is SCALE-ANCHORED on every fit. `_anchor_weight`
multiplies it by the one scalar that makes tr(A^T W A), the squared Frobenius norm of
the weighted design, equal the 'g1-squared' weight's over the same channels; that pins
the effective-dof-against-alpha curve to the old one at the high-alpha end and leaves
the Settings-seeded sweep bounds where they were derived. What no anchor fixes is that
a hand-set alpha means a different amount of smoothing under a weight of a different
SHAPE. A WEIGHT THAT CANNOT BE FORMED falls back rather than refusing: if Eq. (9)
refuses an input, the order-1 fit does not fit, or a first pass leaves the anchor
nothing to match, the fit runs under the 'g1-squared' weight, records that form and
carries a passive note naming the refusal; a fit whose SOLVER fails after the weight
was formed is an ordinary fit failure, unchanged. The weight is a precision map that
improves the FIT only; no statistical +/- is ever
reported from a single correlogram (invariant 8) -- Schätzel 1990 (the correlated,
non-uniform correlator noise this diagonal approximates) is exactly why a single-shot ±
is still deferred to replicate averaging.

Solver note (a deliberate departure from the Salazar et al. 2023 GUI): we do NOT
use SLSQP with a sum(x)=1 equality constraint. Instead we use the standard
augmented-NNLS formulation -- stack [M^.5 A; alpha L] over [M^.5 y; 0] and call
scipy.optimize.nnls -- which is faster, more robust, and avoids forcing the
distribution to integrate to exactly 1 when the data do not support it. The
reported distribution is normalized to sum 1 afterwards, for display only.

The data y fed to the solver is the recovered field ACF, y = sign(u) sqrt(|u|) with
u = (g2 - 1 - baseline) / beta, so that y(0) ~ 1 and the recovered weights are an
intensity-weighted distribution. The signed (not clipped) square root keeps the
long-lag noise zero-mean so no positive g1 pedestal is rectified into the fit. beta
and baseline are estimated by default (see _estimate_beta / _estimate_baseline in
_common) but may be overridden.

This module also owns the distribution post-processing shared by the GUI: the
Rh<->Gamma axis toggle (distribution_axis) and peak detection
(find_distribution_peaks).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, NamedTuple, Optional

import numpy as np
from scipy import optimize, special

from core.data_models import DLSMeasurement
from analysis.dls._common import (
    _apply_tau_window,
    _apply_tau_window_live,
    _build_rh_gamma_grid,
    _estimate_baseline,
    _estimate_beta,
    _measurement_q_m,
    _require_viscosity,
    _result_flags,
    _rms_error,
    channel_sample_times,
)
from analysis.dls.cumulants import _dead_channel_note, _sourced_sigma, fit_cumulants


# The residual weightings a caller may REQUEST on any of the three distribution fits, each
# a statement about what the g1-space residual is measured in: 'biganzoli', the sourced
# diagonal weight g1^2/sigma^2 with both factors on a fitted model (Biganzoli and Ferri
# 2018 Eq. (9) at Koppel's evaluation point; Maier et al. 1999 p. 4674 for the spectrum the
# g1 factor is read from); 'g1-squared', the older form that takes Var(g2-1) as constant
# across channels and leaves w = y^2, kept selectable so a result reported under it can be
# reproduced. An UNWEIGHTED fit is NOT on the roster: on a recovered g1 it is the pedestal
# regression the module docstring records, so it is refused rather than offered.
DISTRIBUTION_WEIGHTINGS = ('biganzoli', 'g1-squared')

# The distribution arms' default weighting, chosen HERE and nowhere else: `weighting=None`
# on any of the three fits means this constant, so the program-wide default is one line to
# move and every result records which value produced it. The sourced diagonal weight is
# the default (owner-ruled 2026-09-20): the older form's premise -- one variance on every
# channel -- is measured false on real correlators, and the Theory-and-Equations-Guide's
# "The residual weight" carries the evidence. The one request resolves to the classical
# limit on an instrument that reports no mean count rate, so the default runs everywhere;
# 'g1-squared' stays selectable to reproduce a result reported under it.
DISTRIBUTION_WEIGHTING_DEFAULT = 'biganzoli'

# What `DistributionResult.weighting` may hold. The request token 'biganzoli' splits in two
# on the way to the result, because the same sample is weighted differently on an instrument
# that reports a mean count rate (Eq. (9)'s two shot-noise brackets enter) and one that
# cannot (the classical infinite-count-rate limit), and a result must say which it got.
# 'g1-squared', the older form, is recorded when it was asked for, and also where the
# sourced weight could not be formed and the fit fell back to it (`_weight_fallback_note`).
RECORDED_DISTRIBUTION_WEIGHTINGS = frozenset({
    'biganzoli-full', 'biganzoli-classical', 'g1-squared',
})

# The recorded value(s) the DEFAULT request resolves to ('biganzoli' splits by whether a
# count rate entered) and, through them, the set an EXPORT IS SILENT ON -- the cumulant
# arm's `DEFAULT_RECORDED_WEIGHTINGS`, in this family's material and gated the same way:
# the program's default weighting is not labelled where its numbers travel (the guide
# describes it, as it describes HC3 and 'calibrated'), while a weighting the user opted
# into is named. There is no linear arm here, so the set is just the default's two forms.
DEFAULT_RECORDED_DISTRIBUTION_WEIGHTINGS = frozenset({
    'biganzoli-full', 'biganzoli-classical',
})

# The one passive sentence a fit carries when it ran under the sourced weight with no
# committed mean count rate. It names the ABSENT INPUT and what committing it buys
# (invariant 4 clause 1: the precondition, never a proxy), and it is UI-neutral -- the
# panel, the per-fit export and the summary export all show this same string. Not a
# refusal and not a reliability verdict: the classical limit is a documented form of the
# variance, so the weight is sourced either way and the fit is reported either way.
NO_COUNT_RATE_NOTE = (
    'No mean count rate was committed for this run, so the sourced residual weight was '
    'built from the infinite-count-rate limit of its variance model; committing one adds '
    'the photon-counting terms to the weight.'
)


@dataclass
class DistributionResult:
    """A decay-rate / size distribution from NNLS or CONTIN."""
    method: str                       # 'nnls' or 'contin'
    alpha: Optional[float]            # regularization parameter (None for NNLS)
    # the grid (paired: rh_grid_nm[i] corresponds to gamma_grid_s_inv[i])
    rh_grid_nm: np.ndarray
    gamma_grid_s_inv: np.ndarray
    weights: np.ndarray               # normalized intensity weights (sum = 1)
    # summary statistics of the distribution
    mean_rh_nm: float                 # intensity-weighted mean Rh
    mean_gamma_s_inv: float           # intensity-weighted mean decay rate
    peak_rh_nm: float                 # Rh at the largest weight
    # quality / context
    beta: float
    beta_estimated: bool
    baseline: float
    baseline_estimated: bool
    q_m_inv: float
    fit_tau_s: np.ndarray             # delay times used (after windowing)
    fitted_g2m1: np.ndarray           # beta * (A x)^2 + baseline reconstructed on fit_tau_s
    residuals: np.ndarray             # (g2-1) data - fitted, on fit_tau_s
    rms_error: float
    residual_norm: float              # weighted ||M^.5 (A x - y)||^2 (g1-space residual)
    solution_norm: float              # ||x||^2
    n_skipped: int = 0                # leading channels dropped (skip_initial_channels)
    # False only when a nonlinear fit (lognormal) failed to converge and the curve is
    # the seed estimate, not a fit. The linear solvers (NNLS/CONTIN) always deliver
    # their solution, so they leave the default. Consumers must never present the
    # summary values, peaks, or a rho source from a success=False result as fitted.
    success: bool = True
    # Passive notes, the shape CumulantResult carries and the GUI/export read generically:
    # the dead-channel note when the live-channel mask dropped a channel from the inversion
    # (`_prepare_distribution_inputs`), and the note a weight that could not be formed
    # carries (`_weight_fallback_note`).
    notes: tuple = ()
    # WHICH WEIGHT BUILT THIS FIT'S RESIDUALS, one of `RECORDED_DISTRIBUTION_WEIGHTINGS`
    # (the `CumulantResult.weighting` pattern): a weight moves the recovered distribution,
    # the selected alpha and the sweep's diagnostic arrays, so a result says which form
    # produced it. 'biganzoli-...' names whether a committed count rate entered Eq. (9);
    # 'g1-squared' is the older constant-variance form, asked for or fallen back to.
    # Defaulted so a record written before this field existed reads truthfully -- that
    # record predates the sourced form, so the older weight is what produced it.
    weighting: str = 'g1-squared'
    # WHICH per-channel sample time the sourced variance was evaluated on: 'instrument'
    # (the parser recorded the correlator's own, `DLSMeasurement.sample_times_s`), 'axis'
    # (a stand-in derived from the delay axis, `channel_sample_times`), or '' when this fit
    # read none -- the 'g1-squared' form, asked for or fallen back to. On a correlator whose
    # sampling time is not its lag spacing the stand-in is off by decades on most channels,
    # so a number weighted on it never travels without saying so, and the exporter names
    # the axis-derived case.
    sample_time_provenance: str = ''
    # WHICH form of Eq. (9)'s 1/(M - k) that variance used: 'finite' (the measurement
    # carries `DLSMeasurement.duration_s`, so 1/(M - k) = Delta_t_k / (T - tau_k)),
    # 'infinite' (none carried: the T -> infinity form, a named practicality divergence), or
    # '' when this fit read no duration. The weighting says which variance ran; this says
    # only which form of its 1/(M - k). All three travel out of the program as values
    # (`tests/test_export.py::test_the_distribution_export_writes_what_its_weight_read`
    # pins it). Both fields are `CumulantResult`'s, in its vocabulary, and both
    # default to '' -- a record written before they existed read neither input either.
    run_duration_form: str = ''


@dataclass
class LCurveResult:
    """The alpha sweep used to choose CONTIN's regularization parameter.

    Despite the name (kept for backward compatibility), this holds the sweep for
    ANY of the three alpha-selection methods — GCV, the L-curve corner, or the
    Provencher F-test. `dof_eff` is populated for GCV and the F-test (both need the
    Tikhonov hat-trace); `ftest_fc` only for the F-test; `gcv` only for GCV.

    `dof_eff` comes from ONE decomposition of the weighted design, shared by every alpha
    of the sweep (`_tikhonov_filter_basis`), so it is non-increasing in alpha by
    construction. `gcv` holds the score the GCV-family selector actually MINIMIZED —
    the classical functional at γ = 1, Lukas's robust one (RGCV) below it, which is why
    `ContinResult` records the γ beside the selector's name.

    `x_norms` is the L-curve's SOLUTION AXIS, ‖x‖² per alpha — the size of the solution
    vector (Hansen & O'Leary 1993 p. 1495; Scotti et al. 2015 §II.A), what
    `_lcurve_corner` plots against the residual. `solution_norms` holds ‖L x‖², the
    seminorm the fit actually penalises: a diagnostic of how much smoothing each alpha
    bought, and NO LONGER the L-curve's axis (`_regularization_seminorm_sq` says why).
    Its name is kept for the same reason this class's is — a session saved before the
    split reads back truthfully, and its L-curve was selected on the seminorm. `x_norms`
    is None only on such a record; every sweep and every user-fixed alpha populates it.
    """
    alphas: np.ndarray
    residual_norms: np.ndarray        # weighted ||M^.5 (A x - y)||^2 for each alpha
    solution_norms: np.ndarray        # ||L x||^2 (regularization seminorm; a diagnostic)
    optimal_alpha: float
    optimal_index: int
    dof_eff: Optional[np.ndarray] = None    # Tikhonov effective DOF per alpha (GCV + F-test)
    ftest_fc: Optional[np.ndarray] = None   # cumulative F ("probability to reject") per alpha
    gcv: Optional[np.ndarray] = None        # the GCV-family score per alpha (minimized)
    x_norms: Optional[np.ndarray] = None    # ||x||^2 per alpha (the L-curve's solution axis)


@dataclass
class ContinResult:
    """CONTIN result: the chosen distribution plus the sweep it came from.

    `alpha_selection_method` records how the automatic alpha was chosen
    ('gcv' | 'lcurve' | 'ftest'); `ftest_prob_reject` the F-test level and `gcv_gamma`
    the robust-GCV robustness parameter, each recorded only under the selector it
    belongs to and None otherwise, so a distribution is never ambiguous about how its
    regularization was picked. 'gcv' names the whole GCV family, classical and robust
    alike: γ = 1 is the classical functional and the pair (method, γ) is the full
    answer. When alpha was user-supplied, the method is reported as 'user' and both
    are None -- there was no sweep for either to act on.

    `alpha_at_ceiling` flags that the selection sits at the HIGH-alpha end of the
    sweep: the window holds only the under-smoothed part of the problem (under the
    L-curve, only its vertical branch), so the answer the selector wanted lies above
    the ceiling and the alpha reported is the range's most-regularized fit. For GCV this
    is the documented flat-minimum failure mode (Hansen 1998 p.185); the note names the
    position alone (raise the ceiling, or inspect the L-curve). `alpha_at_floor` is its
    twin for the LOW end: the window holds only the
    over-smoothed part (the L-curve's horizontal leg) and the selector wanted a smaller
    alpha than the range allows -- at a low floor (the default 1e-4) that is essentially
    unregularized NNLS, whose peak structure may be inversion noise rather than
    populations (the "extra peaks" of an unregularized fit, CONTIN Users Manual §3.2.3);
    at a user-raised floor it is only the least-smoothed answer the range permitted.
    Under either flag the alpha reported IS the one at that end of the range -- for GCV
    and the F-test because the state is read off the selected index, for the L-curve
    because a window that missed the corner reports the end its state names
    (`_lcurve_corner`; owner-ruled 2026-09-21) -- so the flag, the note and the plot's
    edge halo all describe the alpha the distribution was computed at. Both derive from
    ONE end state per fit, so they are exclusive whenever the sweep has two or more
    points; a one-point sweep is both edges under the index-derived selectors (GCV, the
    F-test) and 'no corner' under the L-curve; a user-supplied alpha sets neither (there
    is no sweep). Both flag, neither withholds: the distribution is always returned
    (CLAUDE.md invariant 4, flag-not-refuse).

    `alpha_reliable` / `reliability_note` are the family pair every parametric result
    carries (CumulantResult's shape): the end state gives its note through
    `_ALPHA_STATE_NOTES`, the selector's own preconditions give theirs, and the two roll
    up through `_result_flags` into the ONE string every surface shows -- so the flag and
    the sentence come from one call and cannot disagree. The states: ALPHA_FLOOR_NOTE,
    ALPHA_CEILING_NOTE, ALPHA_NO_CORNER_NOTE (the L-curve found no corner in the window,
    so neither edge flag fires and the note says so rather than naming a proxy),
    ALPHA_ONE_POINT_NOTE on a one-point sweep under GCV or the F-test. The preconditions,
    under the F-test only: FTEST_REFERENCE_NOTE and FTEST_BAND_NOTE_TEMPLATE, the two
    checks that criterion's own manual prescribes (`fit_contin`'s
    `ftest_reference_agreement` and `ftest_prob_band`). Every live note is kept, in that
    order, and the string is '' when the selection is interior, every precondition held,
    or the alpha was user-fixed. Every surface reads this string (pinned by
    tests/test_workspace.py::test_contin_edge_note_is_one_string_on_every_surface).
    """
    distribution: DistributionResult
    lcurve: LCurveResult
    alpha_was_user_supplied: bool
    alpha_selection_method: str = 'lcurve'    # 'gcv' | 'lcurve' | 'ftest' | 'user'
    ftest_prob_reject: Optional[float] = None
    gcv_gamma: Optional[float] = None         # robust-GCV gamma, under 'gcv' only
    alpha_at_ceiling: bool = False
    alpha_at_floor: bool = False
    alpha_reliable: bool = True
    reliability_note: str = ''
    notes: tuple = ()                 # the chosen distribution's notes, lifted for readers


# The sweep-edge wording, in its one home. Both phrases are sized to the predicate: the
# index says only that the selector wanted past the bound, so each names the direction
# and nothing more, and what the solution then IS (unregularized at the default floor,
# not at alpha = 30; the range's smoothest fit at the ceiling) lives in the how-to help,
# conditioned on the bound's value. The ceiling phrase no longer hedges "possible
# over-regularization": that is GCV's flat-minimum reading (Hansen 1998 p. 185) and the
# opposite of the L-curve's ceiling state, whose corner lies ABOVE the window -- a
# proxy for the failed precondition (CLAUDE.md invariant 4, clause 1).
ALPHA_FLOOR_NOTE = ('CONTIN α at the sweep floor — the selector wanted a smaller α than '
                    'the range allows')
ALPHA_CEILING_NOTE = ('CONTIN α at the sweep ceiling — the selector wanted a larger α than '
                      'the range allows')
# The L-curve's two further states. "No corner" is not an edge and must not borrow the
# floor's phrase: the window's points carry no L shape at all (too few distinct points,
# or no horizontal leg AND no vertical branch around the pick), so the fit reports the
# least-regularized alpha of the range and says exactly that. The one-point wording
# replaces joining the floor and ceiling phrases: a single-point sweep has no interior
# for any selector to choose in, which is one fact about the range, not two edge hits.
ALPHA_NO_CORNER_NOTE = ('CONTIN α by L-curve: the sweep shows no corner — the '
                        'least-regularized α of the range is reported')
ALPHA_ONE_POINT_NOTE = ('CONTIN α on a one-point sweep — the range has no interior, so '
                        'the selector had no choice')

# The F-test's OWN two preconditions, which its manual prescribes checking on every run
# (CONTIN Users Manual §3.6.5 points 1 and 3, printed p. 17). They are statements about
# the sweep the test ran on, not about where the pick landed, so they are stated BESIDE
# the end state's note rather than instead of it: independently failed preconditions are
# each named (CLAUDE.md invariant 4, clause 1). Point (1) reads on the reference solution
# -- the least-squares end every F-statistic is measured against -- and a floor too high
# for it is a different fact from the pick sitting on that floor. Point (3)'s numbers are
# formatted from the band that produced them, so a non-default band prints no stale pair.
FTEST_REFERENCE_NOTE = ('CONTIN α by F-test: the sweep floor is not small enough for the '
                        'reference solution — its penalty is not negligible beside its '
                        'residual')
FTEST_BAND_NOTE_TEMPLATE = ('CONTIN α by F-test: the chosen solution\'s probability to '
                            'reject lies outside {lo:g}–{hi:g} — the sweep does not '
                            'bracket the test\'s transition')

# Selected-alpha END STATE -> the one reliability note it carries. 'interior' is absent
# on purpose: an interior pick is the healthy case and carries no note, so a lookup that
# misses is the empty string. Every selector reduces its outcome to one of these states
# (`_lcurve_corner`, `_end_state_of_index`), and `fit_contin` derives both edge flags and
# the note/flag pair from it -- one source, so the two can never disagree.
_ALPHA_STATE_NOTES = {
    'floor': ALPHA_FLOOR_NOTE,
    'ceiling': ALPHA_CEILING_NOTE,
    'no-corner': ALPHA_NO_CORNER_NOTE,
    'both': ALPHA_ONE_POINT_NOTE,
}


# ---------------------------------------------------------------------------
# Solver internals
# ---------------------------------------------------------------------------

# Iteration budget for scipy.optimize.nnls, as a multiple of the number of unknowns.
# scipy's own default is 3n, which one committed real correlogram exceeds at one alpha of
# its 40-point sweep (a 298x100 augmented system: 300 raises, 1000 converges, 10000 gives
# the identical vector) -- and an uncaught raise there cost the user the whole
# distribution. A budget, not a tolerance (invariant 4: no physics threshold hides here):
# nnls stops on optimality, so raising the cap never changes a solve that already
# converged. Serves every _solve_distribution caller (fit_contin's sweep and fit_nnls).
NNLS_MAXITER_FACTOR = 10


def _second_difference_operator(n: int) -> np.ndarray:
    """The (n-2) x n second-difference matrix L (rows [.. 1 -2 1 ..])."""
    L = np.zeros((max(n - 2, 0), n))
    for i in range(n - 2):
        L[i, i] = 1.0
        L[i, i + 1] = -2.0
        L[i, i + 2] = 1.0
    return L


def _apply_weight(A: np.ndarray, y: np.ndarray,
                  w: Optional[np.ndarray]) -> tuple:
    """Row-scale (A, y) by sqrt(w) so an ordinary LS solve minimizes the WEIGHTED
    residual sum(w (A x - y)^2). w = None leaves the system unweighted (M = I)."""
    if w is None:
        return A, y
    sw = np.sqrt(w)
    return A * sw[:, None], y * sw


def _solve_distribution(
    A: np.ndarray,
    y: np.ndarray,
    w: Optional[np.ndarray],
    alpha: float,
    L: Optional[np.ndarray],
) -> np.ndarray:
    """Solve a non-negative, statistically-weighted, optionally-regularized problem.

    Minimizes ||M^.5 (A x - y)||^2 + alpha^2 ||L x||^2 subject to x >= 0, where M is
    the diagonal per-lag weight w (delta-method inverse-variance of the recovered
    g1). The data rows are pre-scaled by sqrt(w) and the (unweighted) regularizer is
    stacked below: [sqrt(w) A; alpha L] over [sqrt(w) y; 0], solved by
    scipy.optimize.nnls. For NNLS pass alpha = 0 (or L = None); the augmentation then
    vanishes. Pass w = None for an unweighted solve.
    """
    Aw, yw = _apply_weight(A, y, w)
    if alpha > 0 and L is not None and L.shape[0] > 0:
        A_aug = np.vstack([Aw, alpha * L])
        y_aug = np.concatenate([yw, np.zeros(L.shape[0])])
    else:
        A_aug, y_aug = Aw, yw
    # NNLS_MAXITER_FACTOR is an iteration BUDGET, not a tolerance: nnls terminates on
    # optimality, so any converged solve is unchanged by a larger cap, and the cap only
    # decides whether a slow solve finishes or raises. See the constant's comment.
    x, _ = optimize.nnls(A_aug, y_aug, maxiter=NNLS_MAXITER_FACTOR * A_aug.shape[1])
    return x


def _distribution_summary(
    x: np.ndarray,
    A: np.ndarray,
    y: np.ndarray,
    w: Optional[np.ndarray],
    rh_grid_nm: np.ndarray,
    gamma_grid: np.ndarray,
    beta: float,
    baseline: float,
    g2m1_data: np.ndarray,
    tau: np.ndarray,
    q_m_inv: float,
    method: str,
    alpha: Optional[float],
    beta_estimated: bool,
    baseline_estimated: bool,
    n_skipped: int = 0,
    success: bool = True,
    notes: tuple = (),
    *,
    weighting: str,
    sample_time_provenance: str,
    run_duration_form: str,
) -> DistributionResult:
    """Assemble a DistributionResult from a raw solver solution x.

    `weighting` is the RECORDED form `w` was built by, one of
    `RECORDED_DISTRIBUTION_WEIGHTINGS` -- passed in rather than inferred from `w`, since
    the sourced weight and the form it falls back to are the same array shape.
    `sample_time_provenance` and `run_duration_form` say what that weight READ, resolved
    from the recorded form by `_distribution_weight_disclosure`. All three are keyword-only
    with NO default, so a return path that forgets one fails loudly here rather than
    recording a fit as having read something it did not.
    """
    total = x.sum()
    # When the solution cannot be normalized, WITHHOLD the whole summary rather than
    # returning the unnormalized vector. Three reasons it has to be NaN and not `x`:
    # `weights` is documented as "normalized intensity weights (sum = 1)", so handing back
    # an unnormalized vector breaks that contract silently; `mean_rh` computed from it is
    # meaningless; and -- the one that actually bit -- `peak_rh = rh_grid[argmax(weights)]`
    # is UNCHANGED by any guard on the mean, so an all-equal degenerate vector still
    # reports the smallest grid point as "the peak", a perfectly plausible small size.
    # Guarding the divisor alone fixed the mean and left the peak lying (spec-03 review).
    if math.isfinite(total) and total > 0:
        weights = x / total
    else:
        weights = np.full_like(np.asarray(x, dtype=float), np.nan)
    # intensity-weighted means over the distribution
    mean_gamma = float(np.sum(weights * gamma_grid))
    mean_rh = float(np.sum(weights * rh_grid_nm))
    peak_rh = (float(rh_grid_nm[int(np.nanargmax(weights))])
               if weights.size and np.any(np.isfinite(weights)) else float('nan'))
    # A x is the recovered field ACF g1; map back to measured space via the Siegert
    # relation g2 - 1 = beta |g1|^2 (+ residual baseline).
    g1_fit = A @ x
    fitted_g2m1 = beta * g1_fit ** 2 + baseline
    residuals = g2m1_data - fitted_g2m1
    # the WEIGHTED g1-space residual the F-test / L-curve read (M = w, or I if None)
    resid_g1 = g1_fit - y
    wv = resid_g1 ** 2 if w is None else w * resid_g1 ** 2
    residual_norm = float(np.sum(wv))
    solution_norm = float(np.sum(x ** 2))
    return DistributionResult(
        method=method, alpha=alpha,
        rh_grid_nm=rh_grid_nm, gamma_grid_s_inv=gamma_grid, weights=weights,
        mean_rh_nm=mean_rh, mean_gamma_s_inv=mean_gamma, peak_rh_nm=peak_rh,
        beta=beta, beta_estimated=beta_estimated,
        baseline=baseline, baseline_estimated=baseline_estimated,
        q_m_inv=q_m_inv, fit_tau_s=tau, fitted_g2m1=fitted_g2m1,
        residuals=residuals, rms_error=_rms_error(residuals),
        residual_norm=residual_norm, solution_norm=solution_norm,
        n_skipped=int(n_skipped or 0),
        success=success,
        notes=tuple(notes),
        weighting=weighting,
        sample_time_provenance=sample_time_provenance,
        run_duration_form=run_duration_form,
    )


def _prepare_distribution_inputs(
    measurement, tau_min_s, tau_max_s, beta, baseline,
    rh_min_nm, rh_max_nm, n_grid, skip_initial_channels=0,
):
    """Shared setup for NNLS/CONTIN/lognormal: window, live channels, baseline, beta,
    grid, A, y -- and the dead-channel note (or '') as the last element.

    The inversion works on the LIVE channels only (`_common._live_channels`, one rule for
    every fitter): a dead channel at g2-1 = -1 sits exactly where the tail statistics read
    the baseline and beta, and it pulled NNLS on the committed dead-tail correlogram to a
    baseline of -0.67, a beta of 1.4 and a tenfold-wrong size, green. The returned `tau`
    and `g2m1` are therefore the live channels; the note counts what was dropped.
    """
    tau, g2m1, alive = _apply_tau_window_live(
        measurement.delay_times_s, measurement.correlogram,
        tau_min_s, tau_max_s, min_points=4,
        skip_initial_channels=skip_initial_channels,
    )
    dead_note = _dead_channel_note(int(np.count_nonzero(~alive)), int(g2m1.size))
    tau, g2m1 = tau[alive], g2m1[alive]
    baseline_estimated = baseline is None
    if baseline is None:
        baseline = _estimate_baseline(tau, g2m1)
    beta_estimated = beta is None
    if beta is None:
        beta = _estimate_beta(tau, g2m1, baseline)
    # isfinite as well as > 0: `beta <= 0` admits nan AND +inf, and every line below
    # divides by beta -- an infinite one silently yields a signal of zeros, a NaN one a
    # signal of NaNs, and neither reaches the user as an error. `_estimate_beta` now
    # guarantees a positive finite value, but a USER-supplied beta arrives unchecked
    # (2026-07-31 spec-01).
    if not (math.isfinite(beta) and beta > 0):
        raise ValueError(f"beta must be a positive finite number, got {beta!r}.")

    q = _measurement_q_m(measurement)
    eta = _require_viscosity(measurement)   # distributions are reported in Rh
    rh_grid, gamma_grid = _build_rh_gamma_grid(
        rh_min_nm, rh_max_nm, n_grid, q, measurement.temperature_K, eta,
    )
    # g1-space inversion (Berne & Pecora 2000): single-Gamma kernel
    # A[m,n] = exp(-Gamma_n tau_m); recover the field ACF from the Siegert relation
    # g2 - 1 = beta |g1|^2, i.e. g1 = sqrt((g2-1-baseline)/beta).
    A = np.exp(-np.outer(tau, gamma_grid))
    u = (g2m1 - baseline) / beta                       # the measured |g1|^2 estimate
    # SIGN-PRESERVING recovery: in the long-lag noise floor u is zero-mean and dips
    # negative; a hard clip-at-0 would keep only the positive excursions and rectify
    # that noise into a POSITIVE g1 pedestal, which a non-negative fit absorbs as
    # spurious large-Rh weight (the noisy-unimodal mean-Rh regression). Taking
    # the signed square root keeps the noise zero-mean, so no pedestal forms; the
    # fitted model A x (a true non-negative g1) still only chases the real signal.
    y = np.sign(u) * np.sqrt(np.abs(u))
    # Delta-method inverse-variance weight of the recovered g1: a (locally uniform)
    # g2-1 noise through the sqrt gives Var(g1) prop 1/|u|, so w prop |u| = y^2. The
    # overall scale cancels at any FIXED alpha (and for NNLS) but NOT in the alpha
    # selection: w carries 1/beta, and rescaling w by c slides the selected alpha by
    # sqrt(c) across the fixed sweep (module docstring); noise-floor channels
    # (|u| -> 0) get ~0 weight. (Note sqrt(w) y = u exactly, so the weighted fit
    # targets the same unbiased (g2-1)/beta as the legacy diagonal kernel, but
    # through a cross-term-preserving g1 design -- noise-robust AND phantom-free.)
    w = y ** 2
    return (tau, g2m1, baseline, baseline_estimated, beta, beta_estimated,
            q, rh_grid, gamma_grid, A, y, w, dead_note)


# ---------------------------------------------------------------------------
# The residual weight: which form, its two factors, its anchor, and its fallback
# ---------------------------------------------------------------------------

def _resolve_distribution_weighting(weighting: Optional[str]) -> str:
    """The residual weighting a distribution fit will run under, as a REQUEST token.

    `weighting=None` means `DISTRIBUTION_WEIGHTING_DEFAULT`, the one home of the arms'
    default; any other value is returned as given once it is on the roster. One resolution
    shared by all three fits, so a token refused on one is refused on every one.
    """
    resolved = DISTRIBUTION_WEIGHTING_DEFAULT if weighting is None else weighting
    if resolved not in DISTRIBUTION_WEIGHTINGS:
        raise ValueError(
            f"weighting={weighting!r} is not on the distribution fits' roster, which "
            f"weights by {list(DISTRIBUTION_WEIGHTINGS)}. An unweighted fit is not among "
            f"them: on a recovered g1 it rectifies the long-lag noise into a positive "
            f"pedestal a non-negative fit absorbs as spurious large-Rh weight (module "
            f"docstring), so it is refused rather than offered."
        )
    return resolved


def _recorded_distribution_weighting(resolved: str, measurement: DLSMeasurement) -> str:
    """The RECORDED form of a resolved request: 'biganzoli' splits by whether the
    measurement carries a committed mean count rate, because Eq. (9)'s two shot-noise
    brackets weight the same sample differently from its classical limit."""
    if resolved != 'biganzoli':
        return resolved
    return ('biganzoli-full' if measurement.mean_count_rate_cps is not None
            else 'biganzoli-classical')


def _distribution_weight_disclosure(recorded: str,
                                    measurement: DLSMeasurement) -> tuple:
    """What the fit's residual weight READ, for a RECORDED form:
    `(sample_time_provenance, run_duration_form, count-rate note or '')`.

    Resolved from the FINAL recorded form -- the one the fit actually ran under -- so the
    fallback is covered by construction: 'g1-squared' reads no sample time, no duration and
    no count rate whether it was asked for or fallen back to, and records ('', '') with no
    note. Under either sourced form the variance is evaluated on the instrument's own
    per-channel sample time when the parser recorded one and on the delay-axis stand-in
    otherwise, and on the finite-duration form of Eq. (9)'s 1/(M - k) when the measurement
    carries a duration and on its T -> infinity form otherwise.

    The count-rate note rides here rather than at each arm because it is the same question
    asked of the same string: 'biganzoli-classical' IS "the sourced weight ran without a
    committed count rate", so one predicate serves all three arms instead of three that
    have to agree (rule 12 -- guard and operation from one computation).
    """
    if recorded == 'g1-squared':
        return '', '', ''
    return (('instrument' if measurement.sample_times_s is not None else 'axis'),
            ('finite' if measurement.duration_s is not None else 'infinite'),
            (NO_COUNT_RATE_NOTE if recorded == 'biganzoli-classical' else ''))


def _weight_fallback_note(arm: str, refusal: str) -> str:
    """The passive note a fit carries when its sourced weight could not be FORMED.

    The cumulant arm's house pattern (`cumulants._fit_cumulants_nonlinear`, "A weight that
    cannot be formed"): the fit runs, under the form it fell back to, and the note names
    the refused input rather than a proxy (invariant 4 clause 1). It is a NOTE and not a
    reliability verdict. The distribution family has no reliability flag: a fallback is
    disclosed by the recorded form (`DistributionResult.weighting`) and by this note. The
    cumulant arm also turns `rh_reliable` off on its own fallback; that flag predates this
    form and is not extended here (owner-ruled 2026-09-20, this docstring its home; it
    describes what the two arms do and chooses no reading of invariant 4 between them).
    """
    return (
        f'The {arm} fit could not form its residual weight, so it ran under the '
        f'g1-squared weight (w = g1^2, which takes the g2-1 noise as constant across '
        f'channels) instead of the sourced variance: {refusal} The distribution is '
        f'reported, fitted under the older weight rather than withheld.'
    )


def _fitted_sample_times(measurement: DLSMeasurement, tau_min_s, tau_max_s,
                         skip_initial_channels) -> np.ndarray:
    """Per-channel sample time Delta_t_k on the channels the inversion actually fits.

    The instrument's own when the parser recorded one (`DLSMeasurement.sample_times_s`),
    else the stand-in `channel_sample_times` reads off the delay axis. Windowed and
    live-masked through the SAME `_apply_tau_window_live` call, on the same arguments,
    that `_prepare_distribution_inputs` puts the correlogram through, so the sample times
    and the fitted lags name one channel set rather than two that happen to match.
    """
    dt_full = (measurement.sample_times_s if measurement.sample_times_s is not None
               else channel_sample_times(measurement.delay_times_s))
    _tau, _g2m1, alive = _apply_tau_window_live(
        measurement.delay_times_s, measurement.correlogram,
        tau_min_s, tau_max_s, min_points=4,
        skip_initial_channels=skip_initial_channels,
    )
    _tau_dt, dt = _apply_tau_window(
        measurement.delay_times_s, np.asarray(dt_full, dtype=float),
        tau_min_s, tau_max_s, min_points=4,
        skip_initial_channels=skip_initial_channels,
    )
    return dt[alive]


def _sourced_distribution_sigma(measurement: DLSMeasurement, tau: np.ndarray,
                                tau_min_s, tau_max_s,
                                skip_initial_channels) -> np.ndarray:
    """The sourced weight's variance factor: Biganzoli and Ferri (2018) Eq. (9) on the
    fitted channels, at the best-fit single exponential.

    Koppel's evaluation point (1972, the prose after Eq. 26, printed p. 4818) is an order-1
    fit under uniform weights, and `cumulants._sourced_sigma` -- called, never copied -- is
    the wrapper both fitters read it through, so the cumulant and distribution arms evaluate
    ONE variance implementation at ONE point. Raises ValueError naming what failed: an
    order-1 first pass that did not fit or returned a non-finite Gamma or beta, or an input
    Eq. (9) refuses.
    """
    first = fit_cumulants(measurement, order=1, method='nonlinear', weighting='uniform',
                          tau_min_s=tau_min_s, tau_max_s=tau_max_s,
                          skip_initial_channels=skip_initial_channels)
    if not (first.success and math.isfinite(first.gamma_s_inv)
            and math.isfinite(first.beta)):
        raise ValueError(
            f"the order-1 uniform cumulant first pass the variance is evaluated on did "
            f"not fit this correlogram (Gamma = {first.gamma_s_inv!r} 1/s, "
            f"beta = {first.beta!r}), so Eq. (9) has no evaluation point."
        )
    dt = _fitted_sample_times(measurement, tau_min_s, tau_max_s, skip_initial_channels)
    return _sourced_sigma(tau, dt, first.gamma_s_inv, first.beta,
                          measurement.mean_count_rate_cps, measurement.duration_s)


def _anchor_weight(w_raw: np.ndarray, w_ref: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Rescale `w_raw` by ONE scalar so its tr(A^T W A) equals `w_ref`'s.

    tr(A^T W A) = sum_k w_k sum_j A[k,j]^2 is the squared Frobenius norm of the weighted
    design, the matrix the effective degrees of freedom are built from against alpha^2 L^T L
    (`_tikhonov_effective_dof`). Rescaling w by c is alpha -> alpha/sqrt(c), a slide of the
    Settings-seeded sweep window along the solution path, and `_biganzoli_sd` returns a
    relative shape in arbitrary units, so some anchor is mandatory on every fit (module
    docstring). Matching the trace pins the dof-against-alpha curve to the g1-squared
    weight's at the HIGH-alpha end of the sweep (measured within 0.3 dof at alpha = 10 to
    100). It does not hold the whole curve, and no scalar can: the weight's shape changes
    the curve's slope, so the two part toward alpha_min. Every arm reads the anchored
    weight, the lognormal arm included: its objective is scale-free but `curve_fit` stops
    on an absolute gradient norm, so an unanchored rescale moves where it stops.

    Raises ValueError naming why when the raw weight has a non-finite or negative entry,
    when either trace is not finite, when the scale is not finite and strictly positive
    (it overflowed, or underflowed to zero), when the raw trace is not strictly positive
    (nothing to match against), or when the anchored weight is not finite. The
    guards are on the COMPUTED quantities as well as the inputs (rule 12): the trace is a
    sum of products, and saturation is a property of the product, not of any one entry.
    """
    raw = np.asarray(w_raw, dtype=float)
    ref = np.asarray(w_ref, dtype=float)
    if raw.size == 0 or not np.all(np.isfinite(raw)) or bool(np.any(raw < 0.0)):
        raise ValueError(
            f"the sourced residual weight cannot be scale-anchored: of its {raw.size} "
            f"channels {int(np.count_nonzero(~np.isfinite(raw)))} are non-finite and "
            f"{int(np.count_nonzero(raw < 0.0))} negative, and a weight must be finite and "
            f"non-negative on every channel it weights."
        )
    with np.errstate(over='ignore', invalid='ignore'):
        # `A * A`, not `A ** 2`: the guards below are sized to this product, and a Python
        # float `**` raises OverflowError where `*` saturates (CLAUDE.md rule 12).
        row_sq = np.sum(A * A, axis=1)
        trace_raw = float(np.sum(raw * row_sq))
        trace_ref = float(np.sum(ref * row_sq))
    if not (math.isfinite(trace_raw) and trace_raw > 0.0):
        raise ValueError(
            f"the sourced residual weight cannot be scale-anchored: its "
            f"tr(A^T W A) = sum_k w_k sum_j A[k,j]^2 is {trace_raw!r}, and only a finite "
            f"strictly positive trace can be scaled onto another. A first-pass spectrum "
            f"that is zero on every channel leaves the anchor nothing to match."
        )
    if not math.isfinite(trace_ref):
        raise ValueError(
            f"the sourced residual weight cannot be scale-anchored: the reference "
            f"tr(A^T W A) of the g1-squared weight is {trace_ref!r}."
        )
    scale = trace_ref / trace_raw
    # Both tails (rule 12): a scale that OVERFLOWS is non-finite, one that UNDERFLOWS is
    # 0.0 -- finite, and it would hand back a weight that is zero on every channel.
    if not (math.isfinite(scale) and scale > 0.0):
        raise ValueError(
            f"the sourced residual weight cannot be scale-anchored: matching "
            f"tr(A^T W A) = {trace_raw!r} onto {trace_ref!r} needs a scale of {scale!r}, "
            f"which is zero or outside the floating-point range."
        )
    with np.errstate(over='ignore', invalid='ignore'):
        anchored = raw * scale
    if not np.all(np.isfinite(anchored)):
        raise ValueError(
            f"the sourced residual weight cannot be scale-anchored: scaling it by "
            f"{scale!r} leaves "
            f"{int(np.count_nonzero(~np.isfinite(anchored)))} of {anchored.size} channels "
            f"non-finite."
        )
    return anchored


def _sourced_weight(A: np.ndarray, x0: np.ndarray, sd: np.ndarray,
                    w_ref: np.ndarray) -> np.ndarray:
    """The sourced diagonal weight, anchored: w proportional to (A x0)^2 / sd^2.

    `x0` is the first pass's solution, so `A x0` is the fitted field correlation Maier
    et al. (1999, p. 4674) evaluate an error model on, and `sd` the per-channel SD of g2-1
    at Koppel's single exponential. Neither factor is read off the noisy observation. The
    anchor is looked up through the module, so every fit under this form passes through the
    one `_anchor_weight` (module docstring).
    """
    g1 = A @ np.asarray(x0, dtype=float)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        raw = (g1 * g1) / (sd * sd)
    return _anchor_weight(raw, w_ref, A)


def _sourced_sigma_or_fallback(resolved: str, measurement: DLSMeasurement,
                               tau: np.ndarray, tau_min_s, tau_max_s,
                               skip_initial_channels, arm: str) -> tuple:
    """`(recorded form, sd or None, fallback note)` for a resolved weighting request.

    `sd is None` means the fit runs under the g1-squared weight -- because that is what was
    asked for (no note), or because the variance could not be formed (the note names the
    refusal). Shared by all three arms so one request resolves one way everywhere.
    """
    if resolved != 'biganzoli':
        return resolved, None, ''
    try:
        sd = _sourced_distribution_sigma(measurement, tau, tau_min_s, tau_max_s,
                                         skip_initial_channels)
    except (RuntimeError, ValueError, TypeError) as exc:
        return 'g1-squared', None, _weight_fallback_note(arm, str(exc))
    return _recorded_distribution_weighting(resolved, measurement), sd, ''


# ---------------------------------------------------------------------------
# NNLS
# ---------------------------------------------------------------------------

def fit_nnls(
    measurement: DLSMeasurement,
    rh_min_nm: float = 1.0,
    rh_max_nm: float = 1000.0,
    n_grid: int = 100,
    beta: Optional[float] = None,
    baseline: Optional[float] = None,
    tau_min_s: Optional[float] = None,
    tau_max_s: Optional[float] = None,
    skip_initial_channels: int = 0,
    weighting: Optional[str] = None,
) -> DistributionResult:
    """Recover a decay-rate distribution by non-negative least squares.

    Solves min ||A x - y||^2 subject to x >= 0, with no smoothing. NNLS is the
    unregularized special case (alpha = 0) of CONTIN. It can resolve clearly
    separated modes but, lacking a smoothness constraint, tends to produce spiky
    distributions that are sensitive to noise; CONTIN is usually preferable for
    real data.

    Parameters
    ----------
    measurement : DLSMeasurement
    rh_min_nm, rh_max_nm : float
        Hydrodynamic-radius grid limits (nm). Default 1 to 1000 nm, log-spaced.
    n_grid : int
        Number of grid points (default 100).
    beta : float, optional
        Coherence factor for normalization. Estimated from the data if omitted.
    baseline : float, optional
        Residual baseline offset to subtract. Estimated from the long-delay tail
        if omitted.
    tau_min_s, tau_max_s : float, optional
        Inclusive delay-time window. Default uses all points.
    weighting : str, optional
        Which residual weight to fit under, one of `DISTRIBUTION_WEIGHTINGS`; None
        means `DISTRIBUTION_WEIGHTING_DEFAULT`. The form that ran is on the result as
        `DistributionResult.weighting`, and under 'biganzoli' the fit runs twice -- a
        first pass supplies the weight's g1 factor (module docstring).

    Returns
    -------
    DistributionResult
    """
    resolved = _resolve_distribution_weighting(weighting)
    (tau, g2m1, baseline, baseline_est, beta, beta_est,
     q, rh_grid, gamma_grid, A, y, w, dead_note) = _prepare_distribution_inputs(
        measurement, tau_min_s, tau_max_s, beta, baseline,
        rh_min_nm, rh_max_nm, n_grid,
        skip_initial_channels=skip_initial_channels)
    recorded, sd, weight_note = _sourced_sigma_or_fallback(
        resolved, measurement, tau, tau_min_s, tau_max_s, skip_initial_channels, 'NNLS')

    x = _solve_distribution(A, y, w, alpha=0.0, L=None)
    if sd is not None:
        # The first pass above ran under the g1-squared weight, which is exactly Maier's
        # equal-variance assumption; its spectrum supplies the weight's g1 factor. A
        # refusal here leaves `x` and `w` as that pass produced them, so the fallback IS
        # the g1-squared fit rather than a second, differently-seeded one.
        try:
            w = _sourced_weight(A, x, sd, w)
        except ValueError as exc:
            recorded, weight_note = 'g1-squared', _weight_fallback_note('NNLS', str(exc))
        else:
            x = _solve_distribution(A, y, w, alpha=0.0, L=None)
    st_prov, dur_form, rate_note = _distribution_weight_disclosure(recorded, measurement)
    return _distribution_summary(
        x, A, y, w, rh_grid, gamma_grid, beta, baseline, g2m1, tau, q,
        method='nnls', alpha=None,
        beta_estimated=beta_est, baseline_estimated=baseline_est,
        n_skipped=skip_initial_channels,
        notes=tuple(n for n in (dead_note, weight_note, rate_note) if n),
        weighting=recorded,
        sample_time_provenance=st_prov, run_duration_form=dur_form)


# ---------------------------------------------------------------------------
# Lognormal (single-mode parametric distribution)
# ---------------------------------------------------------------------------

def fit_lognormal(
    measurement: DLSMeasurement,
    rh_min_nm: float = 1.0,
    rh_max_nm: float = 1000.0,
    n_grid: int = 100,
    beta: Optional[float] = None,
    baseline: Optional[float] = None,
    tau_min_s: Optional[float] = None,
    tau_max_s: Optional[float] = None,
    skip_initial_channels: int = 0,
    weighting: Optional[str] = None,
) -> DistributionResult:
    """Fit a single-mode LOGNORMAL Rh distribution to g2-1.

    A parametric distribution method: it assumes the intensity-weighted size
    distribution is lognormal in Rh and fits its median Rh and log-width, using
    the SAME forward model as NNLS/CONTIN (the g1-space kernel A[m,n] =
    exp(-Gamma_n tau_m)), so the result drops straight into the distribution
    plot/summary. On the log-spaced Rh
    grid the discrete lognormal weight is w_i proportional to
    exp(-(ln Rh_i - mu)^2 / 2 sigma^2) (the 1/Rh of the pdf cancels the log-grid
    spacing); mu = ln(median Rh), sigma is the log-width (polydispersity).

    Unlike NNLS/CONTIN it cannot resolve multiple modes, but it is robust to noise
    and always yields a smooth, single-peaked distribution -- a good default when
    the sample is known to be unimodal. Returns a DistributionResult
    (method='lognormal').

    `weighting` selects the residual weight, one of `DISTRIBUTION_WEIGHTINGS`; None means
    `DISTRIBUTION_WEIGHTING_DEFAULT`, and the form that ran is on the result as
    `DistributionResult.weighting`. Under 'biganzoli' a first lognormal fit under the
    g1-squared weight supplies the weight's g1 factor and the fit then runs again; the
    arm is scale-free (`absolute_sigma=False`), so only the weight's SHAPE reaches it.
    """
    resolved = _resolve_distribution_weighting(weighting)
    (tau, g2m1, baseline, baseline_est, beta, beta_est,
     q, rh_grid, gamma_grid, A, y, w, dead_note) = _prepare_distribution_inputs(
        measurement, tau_min_s, tau_max_s, beta, baseline,
        rh_min_nm, rh_max_nm, n_grid,
        skip_initial_channels=skip_initial_channels)
    recorded, sd, weight_note = _sourced_sigma_or_fallback(
        resolved, measurement, tau, tau_min_s, tau_max_s, skip_initial_channels,
        'lognormal')

    ln_rh = np.log(rh_grid)

    def _weights(mu: float, sigma: float) -> np.ndarray:
        w = np.exp(-0.5 * ((ln_rh - mu) / sigma) ** 2)
        s = w.sum()
        return w / s if (math.isfinite(s) and s > 0) else w

    def model(_x, amp, mu, sigma):
        return amp * (A @ _weights(mu, sigma))

    # Seed the median from a quick cumulant Rh when it falls on the grid.
    mu0 = 0.5 * (ln_rh[0] + ln_rh[-1])
    # This seed keeps the ENGINE DEFAULT cutoff and does not read the user's Settings
    # value. A seed is an implementation detail of the method that owns it, and letting a
    # display-side setting perturb it would make this fit depend on a control the user was
    # not thinking about. The one site in `analysis/` that DOES read the user's value is
    # the Gamma-vs-q2 cumulant in `angular.py`, because that is the same cumulant on the
    # same measurement the user just configured. Sibling gates share a name, not a line.
    try:
        cum = fit_cumulants(measurement, order=2,
                            tau_min_s=tau_min_s, tau_max_s=tau_max_s,
                            skip_initial_channels=skip_initial_channels)
        if np.isfinite(cum.rh_nm) and rh_grid[0] <= cum.rh_nm <= rh_grid[-1]:
            mu0 = math.log(cum.rh_nm)
    except Exception:  # noqa: BLE001 — optional seed
        pass

    p0 = [1.0, mu0, 0.3]
    bounds = ([1e-6, ln_rh[0], 1e-2], [np.inf, ln_rh[-1], 3.0])

    def _pass(w_pass: np.ndarray) -> tuple:
        """One weighted fit at the shared seed: (amp, mu, sigma, converged)."""
        # weight the fit by the same delta-method w: curve_fit minimizes
        # sum(((model - y)/sigma)^2), so sigma = 1/sqrt(w) reproduces sum(w (model-y)^2).
        # Clip so clipped-to-zero channels get a large (not infinite) sigma, i.e. ~0 weight.
        sigma_w = 1.0 / np.sqrt(np.clip(w_pass, 1e-12, None))
        try:
            popt, _ = optimize.curve_fit(
                model, np.arange(tau.size), y, p0=p0, bounds=bounds, maxfev=10000,
                sigma=sigma_w, absolute_sigma=False)
            return float(popt[0]), float(popt[1]), float(popt[2]), True
        except (RuntimeError, ValueError, TypeError):
            # Same exception family the exponential fitters catch around curve_fit. The
            # seed shape stands in as a "roughly here, unconverged" hint; the flag
            # keeps it from ever reading as a fit downstream.
            return 1.0, mu0, 0.3, False

    amp, mu, sigma, converged = _pass(w)
    if sd is not None:
        # The first pass ran under the g1-squared weight -- Maier's equal-variance
        # assumption -- and its spectrum supplies the weight's g1 factor. Either refusal
        # below leaves the parameters and `w` as that pass produced them, so the fallback
        # IS the g1-squared fit. Both passes are seeded from the same p0: what the second
        # pass changes is the precision map, not where the search starts.
        if not converged:
            recorded = 'g1-squared'
            weight_note = _weight_fallback_note(
                'lognormal',
                'the lognormal first pass the weight is evaluated on did not converge, '
                'so there is no fitted spectrum to read its g1 factor from.')
        else:
            try:
                w = _sourced_weight(A, amp * _weights(mu, sigma), sd, w)
            except ValueError as exc:
                recorded = 'g1-squared'
                weight_note = _weight_fallback_note('lognormal', str(exc))
            else:
                amp, mu, sigma, converged = _pass(w)

    x = amp * _weights(mu, sigma)
    st_prov, dur_form, rate_note = _distribution_weight_disclosure(recorded, measurement)
    return _distribution_summary(
        x, A, y, w, rh_grid, gamma_grid, beta, baseline, g2m1, tau, q,
        method='lognormal', alpha=None,
        beta_estimated=beta_est, baseline_estimated=baseline_est,
        n_skipped=skip_initial_channels, success=converged,
        notes=tuple(n for n in (dead_note, weight_note, rate_note) if n),
        weighting=recorded,
        sample_time_provenance=st_prov, run_duration_form=dur_form)


# ---------------------------------------------------------------------------
# CONTIN with L-curve alpha selection
# ---------------------------------------------------------------------------

def _regularization_seminorm_sq(L: Optional[np.ndarray], x: np.ndarray) -> float:
    """||L x||^2 -- the Tikhonov regularization seminorm, recorded as a diagnostic.

    This is the penalty the fit minimises (alpha^2 ||L x||^2, with L the
    second-difference operator), so it says how much smoothing each swept alpha bought.
    It is NOT the L-curve's solution axis. Under a second-difference L the seminorm has
    no floor to flatten onto: every constant-plus-linear component of the solution lies
    in L's null space, N(L2) = span{(1,...,1), (1,2,...,n)}, and "is unaffected by the
    smoothing norm" (Hansen 2010 p. 175, Eq. 8.7), so on this kernel the over-smoothed
    branch keeps falling and the curve never turns horizontal -- no corner rule can find
    a corner that is not there. The L-curve is plotted on the SIZE OF THE SOLUTION
    VECTOR, ||x||^2, instead; `_lcurve_corner` carries that axis and its sources.
    Falls back to ||x||^2 only if L is degenerate (no rows).
    """
    if L is not None and L.shape[0] > 0:
        return float(np.sum((L @ x) ** 2))
    return float(np.sum(x ** 2))


def _unit_step(p_from: np.ndarray, p_to: np.ndarray) -> tuple:
    """The unit vector from one L-curve point to the next; (nan, nan) if they coincide."""
    dx = float(p_to[0]) - float(p_from[0])
    dy = float(p_to[1]) - float(p_from[1])
    length = math.hypot(dx, dy)
    if length > 0.0:
        return (dx / length, dy / length)
    return (math.nan, math.nan)


def _turn_sign(v_in: tuple, v_out: tuple) -> float:
    """The 2-D cross product of two consecutive unit segments -- w_i of Hansen, Jensen &
    Rodriguez (2007) Sec. 3.2, which carries the sign of the turn angle theta. Negative
    is a convex turn, the curvature a corner candidate must have."""
    return v_in[0] * v_out[1] - v_in[1] * v_out[0]


def _slope_from_horizontal(v: tuple) -> float:
    """The UNSIGNED angle between the horizontal vector v_H = (-1, 0) and a unit segment
    (Hansen, Jensen & Rodriguez 2007 Sec. 3.3): |phi| = arccos(-v_x), about 0 along the
    horizontal leg and pi/2 up the vertical branch.

    Unsigned on purpose. The curve is traversed from its most-regularized end, so the
    vertical branch RISES and the paper's oriented phi would put it at -pi/2, which no
    `phi > pi/4` test could ever accept.
    """
    if not (math.isfinite(v[0]) and math.isfinite(v[1])):
        return math.nan
    return math.acos(max(-1.0, min(1.0, -v[0])))


def _pruned_corner_by_angles(pruned: np.ndarray):
    """Corner location by the angles between consecutive segments (Hansen, Jensen &
    Rodriguez 2007 Sec. 3.2), as an index into `pruned`, or None.

    Normalize every segment, form w_i for each interior point, and take the turn closest
    to -pi/2 as k = argmin_i |w_i + 1|. The point is ACCEPTED only if w_k is negative;
    otherwise the pruned curve is flat or concave and the routine finds no corner.
    """
    m = int(pruned.shape[0])
    if m < 3:
        return None
    steps = [_unit_step(pruned[j - 1], pruned[j]) for j in range(1, m)]
    w = np.full(m, np.inf)
    for i in range(1, m - 1):
        turn = _turn_sign(steps[i - 1], steps[i])
        # A coincident pair gives a NaN turn, and NaN WINS an argmin; +inf can only lose.
        w[i] = turn if math.isfinite(turn) else np.inf
    k = int(np.argmin(np.abs(w + 1.0)))
    return k if w[k] < 0.0 else None


def _pruned_corner_by_global_behavior(pruned: np.ndarray, full: np.ndarray):
    """Corner location from the curve's global behaviour (Hansen, Jensen & Rodriguez
    2007 Sec. 3.3), as an index into `full`, or None.

    Identify the most horizontal segment l_h = argmin_j |(v_j)_2| and the most vertical
    one l_v = argmax_j |(v_j)_2| -- the paper ranks by the SECOND COMPONENT of the unit
    segment vector, |sin phi_j|, not by the angle itself (`rise` below is that
    component; a segment running backwards, toward smaller zeta, ranks by the same
    rule) -- constrained so that the horizontal segment lies to the RIGHT of the
    vertical one. The "origin" O is where the horizontal line at eta(l_h) meets the
    line through segment l_v, and the corner is the point of the ENTIRE curve with the
    smallest Euclidean distance to O.

    OURS, not the source's: the paper specifies the right-of constraint only through the
    cost of its double loop, so "to the right" is read here on the segment MIDPOINTS,
    scanned most-horizontal outer and most-vertical inner, accepting the first admissible
    pair. The split between the two lists is at 45 degrees.
    """
    m = int(pruned.shape[0])
    if m < 2:
        return None
    steps = [_unit_step(pruned[j - 1], pruned[j]) for j in range(1, m)]
    rise = np.array([abs(s[1]) for s in steps])
    run = np.array([abs(s[0]) for s in steps])
    mid_x = np.array([(float(pruned[j - 1][0]) + float(pruned[j][0])) / 2.0
                      for j in range(1, m)])
    usable = np.isfinite(rise) & np.isfinite(run)
    horizontal = [j for j in np.argsort(rise, kind='stable')
                  if usable[j] and rise[j] <= run[j]]
    vertical = [j for j in np.argsort(-rise, kind='stable')
                if usable[j] and rise[j] >= run[j]]
    pair = None
    for h in horizontal:
        for v in vertical:
            if h != v and mid_x[h] > mid_x[v]:
                pair = (h, v)
                break
        if pair is not None:
            break
    if pair is None:
        return None
    h, v = pair
    y_o = float(pruned[h + 1][1])          # eta at the horizontal segment's end point
    p0, p1 = pruned[v], pruned[v + 1]
    dy = float(p1[1]) - float(p0[1])
    if dy == 0.0:
        return None
    t = (y_o - float(p0[1])) / dy
    x_o = float(p0[0]) + t * (float(p1[0]) - float(p0[0]))
    return int(np.argmin(np.hypot(full[:, 0] - x_o, full[:, 1] - y_o)))


def _pruning_corner(points: np.ndarray) -> tuple:
    """The adaptive pruning algorithm of Hansen, Jensen & Rodriguez (2007), Fig. 2.

    `points` are the distinct, finite log-log L-curve points IN THE PAPER'S ORDER (index
    0 the most regularized), and the return is `(k, candidates)` as indices into them --
    `k` None when no pruned curve yielded a candidate, which is the paper's "all the
    pruned L-curves are found to be concave" error exit (Sec. 3.1).

    Stage one (lines 1-8) walks pruned curves of the p_hat largest line segments for
    p_hat = 5, 10, 20, ..., p - 1, running both corner-location routines on each and
    collecting their answers; duplicates collapse. Stage two (lines 9-13) chooses one:
    a single candidate wins outright; otherwise phi_i is the unsigned slope of the
    segment LEAVING candidate i (line 10 runs to #L - 1, so the last candidate has no
    phi, which fixes the reading), and either every candidate's outgoing segment is flat
    -- line 12, no vertical branch in the window, take the least-regularized candidate --
    or line 13 takes the first candidate whose outgoing segment is steep AND whose own
    turn is convex. That is the paper's prose: "the wanted corner is the last candidate
    point before reaching the vertical part" (p. 486).

    OURS, not the source's: line 13's set can be empty -- a steep segment exists but no
    candidate before one turns convexly -- where the paper promises an index. The fall
    back is the first candidate with a steep outgoing segment, curvature unchecked: the
    prose's primary rule with its second criterion dropped. And line 12 is read as "no
    candidate's outgoing segment is steep" rather than as the paper's `max(phi) < pi/4`:
    the two agree for every finite slope off the boundary, and only the first is the
    complement of the fallback it guards (a slope that is NaN, or exactly pi/4, counts
    as not steep on both sides of the test). And when the curve's own first point is
    itself a candidate, the P_{k_0} augmentation coincides with it: its incoming step is
    zero-length, its turn sign NaN, so it can enter the steep set but never the feasible
    one. The paper defines no incoming segment for that point, so this reading leaves it
    out rather than inventing one.
    """
    p = int(points.shape[0])
    segments = points[1:] - points[:-1]
    seg_len = np.hypot(segments[:, 0], segments[:, 1])
    found = set()
    p_hat = min(5, p - 1)                                              # line 1
    while p_hat < 2 * (p - 1):                                         # line 2
        p_hat = min(p_hat, p - 1)                                      # line 3
        longest = np.argsort(-seg_len, kind='stable')[:p_hat]          # line 4
        keep = np.unique(np.concatenate([longest, longest + 1]))
        pruned = points[keep]
        by_angle = _pruned_corner_by_angles(pruned)                    # lines 5-7
        if by_angle is not None:
            found.add(int(keep[by_angle]))
        by_global = _pruned_corner_by_global_behavior(pruned, points)
        if by_global is not None:
            found.add(int(by_global))
        p_hat *= 2                                                     # line 8
    candidates = sorted(found)
    if not candidates:
        return None, candidates
    if len(candidates) == 1:                                           # line 9
        return candidates[0], candidates
    # The candidate list is augmented with the whole curve's first point as P_{k_0}
    # (Sec. 3.1), so that the first candidate has an incoming segment to turn through.
    chain = [0] + candidates
    phi = [_slope_from_horizontal(_unit_step(points[chain[i]], points[chain[i + 1]]))
           for i in range(1, len(chain) - 1)]                          # lines 10-11
    # The steep set is built FIRST and line 12's exit is "it is empty", so the exit and
    # the fallback at the bottom are one predicate. Written as `max(phi) < pi/4` they
    # were complements only for finite slopes off the boundary: a NaN slope (or one on
    # pi/4 exactly) failed both tests and `min` of an empty list raised out of the fit.
    feasible, steep = [], []
    for i in range(1, len(chain) - 1):
        if phi[i - 1] > math.pi / 4:
            steep.append(chain[i])
            v_in = _unit_step(points[chain[i - 1]], points[chain[i]])
            v_out = _unit_step(points[chain[i]], points[chain[i + 1]])
            if _turn_sign(v_in, v_out) < 0.0:      # theta(k_{i-1}, k_i, k_{i+1}) < 0
                feasible.append(chain[i])
    if not steep:                                                      # line 12
        return candidates[-1], candidates
    if feasible:
        return min(feasible), candidates                               # line 13
    return min(steep), candidates


def _corner_end_state(points: np.ndarray, k: int, candidates) -> str:
    """Which parts of the L the sweep window actually holds, around the corner at `k`.

    OURS, not the source's: the algorithm's own exits do not answer this. Its line-12
    exit is a NORMAL outcome on a full window (the vertical branch simply yielded no
    candidate beyond the corner), and a window that holds only one branch of the L still
    exits through line 13 at whatever small bend that branch has, which can sit anywhere
    inside the window (measured on the planted cases: a third of the way in, and deeper)
    -- so neither the exit taken nor the index alone distinguishes a healthy interior
    pick from a window that missed the corner. The predicate is two-sided and built on
    the candidate chain the algorithm already produced, running from the window's
    most-regularized end P_1 through the candidates to its least-regularized end P_p:

      * a HORIZONTAL part exists on the more-regularized side of the pick iff some
        secant of P_1 -> ... -> k is flatter than pi/4;
      * a VERTICAL part exists on the less-regularized side iff some secant of
        k -> ... -> P_p is steeper than pi/4.

    Both -> 'interior'. No horizontal part before the pick -> 'ceiling': the window holds
    only the vertical branch, so the corner is above its ceiling. No vertical part after
    it -> 'floor': the window holds only the horizontal leg. Neither -> 'no-corner'.
    Angles are the same unsigned slope from v_H that stage two uses. The state names
    the window's shape, not the pick's position: which alpha a non-interior state then
    reports is `_lcurve_corner`'s rule, not this predicate's.
    """
    chain = sorted({0, int(k), int(points.shape[0]) - 1} | {int(c) for c in candidates})
    pos = chain.index(int(k))
    slopes = [_slope_from_horizontal(_unit_step(points[chain[i]], points[chain[i + 1]]))
              for i in range(len(chain) - 1)]
    horizontal_before = any(s < math.pi / 4 for s in slopes[:pos] if math.isfinite(s))
    vertical_after = any(s > math.pi / 4 for s in slopes[pos:] if math.isfinite(s))
    if horizontal_before and vertical_after:
        return 'interior'
    if horizontal_before:
        return 'floor'
    if vertical_after:
        return 'ceiling'
    return 'no-corner'


def _lcurve_corner(alphas, residual_norms, x_norms) -> tuple:
    """Locate the L-curve's corner, and say which parts of the L the window holds.

    Returns `(index into the ascending sweep, end state)`, the state one of 'interior',
    'floor', 'ceiling' or 'no-corner'.

    THE CORNER is Hansen's canonical criterion -- the point of maximum curvature of the
    log-log curve (Hansen 1998 Eq. 7.24, p. 189). On a SAMPLED, non-negativity-
    constrained sweep that definition needs an operational form, and the one used here is
    the adaptive pruning algorithm of Hansen, Jensen & Rodriguez (2007), Fig. 2 and
    Secs. 3.1-3.3: corner candidates from pruned curves at several resolutions, then one
    choice among them. Its two open points are filled in by us and named where they are
    made (`_pruning_corner`, `_pruned_corner_by_global_behavior`). The source is candid
    that no implementation makes the criterion infallible -- "the heuristic L-curve
    algorithm can fail no matter how it is implemented" (Sec. 5, p. 492).

    THE AXES are

        zeta = log||M^.5 (A x - y)||,   eta = log||x||_2,

    the residual against the SIZE OF THE SOLUTION VECTOR. That size may be taken in "the
    1, 2, or p norms" of the solution vector (Hansen & O'Leary 1993 p. 1495), the
    locator's own paper states the discrete L-curve as (log||A x - b||_2, log||x||_2)
    (Hansen, Jensen & Rodriguez 2007 Eq. 1), and it is the axis the DLS literature plots
    under this same second-difference penalty (Scotti et al. 2015 Sec. II.A). It is NOT
    the general-form seminorm ||L x||: that one has no floor on this kernel, so the
    over-smoothed branch never flattens and there is no corner to find
    (`_regularization_seminorm_sq`). ||x||_2 does flatten, because a non-negative
    distribution's 1-norm is pinned by the fitted amplitude and ||x||_2 >= ||x||_1/sqrt(n)
    bounds it from below.

    `residual_norms` and `x_norms` are stored SQUARED, so zeta/eta take 0.5*log10.
    Non-finite points are dropped before anything else -- a NaN norm carries no geometry
    and would win the nearest-point search of Sec. 3.3 -- and every repeated point then
    collapses to its first occurrence: an equal-bounds sweep degenerates to one point
    that way, and a repeat that is NOT adjacent would otherwise put a zero-length step,
    hence a NaN slope, into stage two's candidate chain. Fewer than three distinct
    points left, or no candidate at all, is 'no-corner': the source returns an error
    there, and since a flagged result still owes the user a distribution (CLAUDE.md
    invariant 4, flag-not-refuse), the least-regularized end of the range is reported
    under a note that says the L-curve found no corner.

    WHICH ALPHA A NON-INTERIOR STATE REPORTS (owner-ruled 2026-09-21; `ContinResult`'s
    docstring is the contract's home): the end the state names. 'floor' reports the
    window's smallest alpha and 'ceiling' its largest, so the flag, the note ("at the
    sweep floor / ceiling") and the plot's edge halo describe the alpha actually
    reported rather than a pick somewhere along the one branch the window holds; and
    'no-corner' reports the smallest alpha on EVERY route, the located-but-legless one
    included, which is what its note promises. Only 'interior' reports the corner.
    """
    n = int(np.asarray(alphas, dtype=float).size)
    res = np.clip(np.asarray(residual_norms, dtype=float), 1e-300, None)
    sol = np.clip(np.asarray(x_norms, dtype=float), 1e-300, None)
    # log of the NORM (not the squared norm the sweep stores) -> factor 0.5. The paper
    # indexes from the MOST regularized end and our sweep ascends in alpha, so the curve
    # is reversed for the algorithm and the pick is mapped straight back through `source`.
    zeta = 0.5 * np.log10(res)[::-1]
    eta = 0.5 * np.log10(sol)[::-1]
    source = np.arange(n)[::-1]
    finite = np.isfinite(zeta) & np.isfinite(eta)
    zeta, eta, source = zeta[finite], eta[finite], source[finite]
    if zeta.size < 3:
        return 0, 'no-corner'
    seen, keep = set(), []
    for i in range(zeta.size):
        if (float(zeta[i]), float(eta[i])) not in seen:
            seen.add((float(zeta[i]), float(eta[i])))
            keep.append(i)
    zeta, eta, source = zeta[keep], eta[keep], source[keep]
    if zeta.size < 3:
        return 0, 'no-corner'
    points = np.column_stack([zeta, eta])
    k, candidates = _pruning_corner(points)
    if k is None:
        return 0, 'no-corner'
    state = _corner_end_state(points, k, candidates)
    if state == 'interior':
        return int(source[k]), state
    return (n - 1 if state == 'ceiling' else 0), state


def _end_state_of_index(idx: int, n: int) -> str:
    """The end state of a selector that answers with a sweep index alone (GCV, F-test).

    A one-point sweep is 'both' -- it has no interior, so the pick is at the floor and at
    the ceiling at once -- and otherwise the two ends of the ascending sweep are 'floor'
    (the smallest alpha) and 'ceiling' (the largest). One computation, so the two edge
    flags a caller derives from it are exclusive on any sweep of two or more points.
    """
    if n == 1:
        return 'both'
    if idx == 0:
        return 'floor'
    if idx == n - 1:
        return 'ceiling'
    return 'interior'


def _masked(a: np.ndarray) -> np.ndarray:
    """A copy of `a` with every non-finite entry replaced by +inf, so it can never win an
    `argmin`.

    `np.argmin` returns the INDEX OF A NaN when one is present, so an alpha whose criterion
    could not be computed is exactly the alpha a bare `argmin` selects -- a silent wrong
    selection, strictly worse than the withhold it came from. `np.clip` does not help:
    it passes NaN straight through.

    One helper shared by both alpha selectors AND by both of their fallback branches --
    masking a selector's main objective while leaving its last-resort `argmin` bare is the
    same defect one branch down. The L-curve corner-finder answers the same hazard at its
    own end, by dropping non-finite points from the curve before it looks for a corner.
    """
    a = np.asarray(a, dtype=float)
    return np.where(np.isfinite(a), a, np.inf)


def _gcv_corner(residual_norms, dof_eff, n_data: int, *,
                gamma: float = 1.0, mu2=None):
    """Generalized Cross-Validation alpha selection (Golub, Heath & Wahba 1979).

    Minimizes the GCV functional (their Eq. 1.4), which for our weighted Tikhonov
    fit and with Tr(I - A(alpha)) = n - dof(alpha) reduces to

        GCV(alpha) = n * V(alpha) / (n - dof(alpha))^2                          (1.4)

    where n = n_data is the number of lag channels, V(alpha) the WEIGHTED residual
    ||M^.5 (A x - y)||^2 (`residual_norms`), and dof(alpha) the effective degrees of
    freedom = trace of the Tikhonov hat matrix (`_tikhonov_effective_dof`, from the one
    decomposition of the weighted design). The n-scaling cancels for the minimizer, so
    the absolute noise level is irrelevant -- GCV is sigma^2-free and asymptotically
    optimal (Wahba). The chosen alpha is the argmin. Returns (index, score_array). The
    score array is the one MINIMIZED here, and is what the sweep records.

    `gamma` < 1 selects Lukas's ROBUST GCV (RGCV) instead, whose score is

        Vbar(alpha) = [ gamma + (1 - gamma) * mu_2(alpha) ] * GCV(alpha)

    (Lukas 2010 Eq. (1.4), printed p. 521; its origin is Lukas 2006, Inverse Problems 22,
    1883). mu_2 = n^-1 tr H^2 is the influence measure `_tikhonov_influence_mu2` computes,
    and gamma is a robustness parameter on (0, 1): as it falls, the selector weighs the
    solution's variance more and is less likely than GCV to under-smooth, which is GCV's
    failure under correlated errors. mu_2 is a trace divided by n, so no scaling of lambda
    enters it and gamma needs no normalization of its own (what that does and does not
    make invariant is in `_tikhonov_influence_mu2`). The printed value is 0.1 in all three
    papers (Lukas 2006, printed pp. 1885-1886;
    Lukas 2010, printed p. 543; Bauer & Lukas 2011 Eq. (33)) and is `fit_contin`'s
    default. At gamma = 1 the factor is exactly 1 and Eq. (1.4) IS GCV, and this function
    then computes the classical score untouched -- no factor of 1.0 is applied, so a non-finite mu_2 cannot reach a result
    that does not depend on it. `mu2` is required when gamma < 1.

    The STRONG form, R1GCV, multiplies by mu_12 = (mu_1 - mu_2) / lambda instead (Lukas
    2010 Eq. (1.5)). It is not offered: mu_12 carries lambda's scale, so its gamma belongs
    to a normalization, and neither printed value (0.9999, Lukas 2010 p. 543; 0.95, Bauer
    & Lukas 2011 Eq. (34)) is in one this kernel's design can reproduce. The owner's
    ruling of 2026-09-22 and its measurement are in the Theory-and-Equations Guide's
    CONTIN section.

    Guarded against a non-positive (n - dof): dof is the Tikhonov hat-trace, a mild
    upper bound on the free-parameter count that for the ill-posed Laplace kernel
    stays well below n, but it is clipped to (0, n) defensively.
    """
    v = np.asarray(residual_norms, dtype=float)
    dof = np.clip(np.asarray(dof_eff, dtype=float), 1e-6, n_data - 1e-6)
    gcv = n_data * v / (n_data - dof) ** 2
    if gamma < 1.0:
        if mu2 is None:
            raise ValueError(
                "a robust GCV score (gamma < 1) needs its influence measure: pass "
                f"mu2 alongside gamma={gamma!r}.")
        gcv = gcv * (gamma + (1.0 - gamma) * np.asarray(mu2, dtype=float))
    # An unanswerable alpha must not WIN the selection. `np.argmin` returns the index of a
    # NaN, and `np.clip` passes NaN straight through, so an alpha whose hat-trace could
    # not be computed was exactly the alpha this returned -- a silent wrong selection,
    # strictly worse than the withhold it came from. Masked to a never-wins +inf.
    if not np.any(np.isfinite(gcv)):
        # No alpha in the sweep has a usable score, so the criterion did not answer.
        # Fall back to the minimum finite RESIDUAL -- the same reference the F-test uses --
        # rather than to index 0 by accident. (Index 0 IS the smallest alpha for the
        # ascending geomspace the caller builds, but that is a property of the caller, not
        # of this function, and the two must not agree only by coincidence.) If even the
        # residuals are all non-finite the masked argmin ties at index 0, which is the one
        # case where there is genuinely no basis to prefer any alpha.
        return int(np.argmin(_masked(v))), gcv
    return int(np.argmin(_masked(gcv))), gcv


class _TikhonovFilterBasis(NamedTuple):
    """One weighted design's regularization geometry, alpha-free.

    `cos` and `sin` are the CS pair (c_i, s_i) of the stacked pair's orthonormal range,
    one per direction: c_i^2 + s_i^2 = 1 by construction, and c_i / s_i is that
    direction's GENERALIZED SINGULAR VALUE of the pair (weighted design, L) -- Hansen
    1998 Eqs. (2.6)-(2.7), printed p. 22, where the same pair is written (sigma_i, mu_i)
    with sigma_i^2 + mu_i^2 = 1 and gamma_i = sigma_i / mu_i. Every alpha's filter
    factors, hat trace and influence measure are functions of these two alone;
    `_tikhonov_filter_basis` derives them once.
    """
    cos: np.ndarray
    sin: np.ndarray


def _tikhonov_filter_basis(A_w: np.ndarray,
                           L: np.ndarray) -> Optional[_TikhonovFilterBasis]:
    """The ONE decomposition every alpha's Tikhonov filter factors come from.

    Returns None -- the withhold every consumer already handles -- when either input
    carries a non-finite entry, because `np.linalg.svd` RAISES `LinAlgError` on one. The
    check is the two lines below and precedes both factorizations, so guard and operation
    name the same matrices (CLAUDE.md rule 12).

    DERIVATION (ours; the identity is checked in
    `tests/test_dls.py::test_the_filter_basis_reproduces_the_hat_trace_on_a_well_conditioned_problem`).
    Let K = [A_w; L] and let K = Q R be any decomposition with Q's r columns orthonormal
    and r = rank(K) -- here Q is the thin SVD's left factor and R = S_r V_r^T. Splitting
    Q into its first m rows Q1 and its last rows Q2 gives A_w = Q1 R and L = Q2 R
    exactly, so

        A_w^T A_w + alpha^2 L^T L = R^T (Q1^T Q1 + alpha^2 Q2^T Q2) R .

    Q's columns are orthonormal, so Q1^T Q1 + Q2^T Q2 = I_r. Writing the SVD
    Q2 = U2 S Z^T gives Q2^T Q2 = Z S^2 Z^T, hence Q1^T Q1 = Z (I - S^2) Z^T = Z C^2 Z^T
    with c_i = ||Q1 z_i||. The hat (influence) matrix is therefore

        H(alpha) = A_w (A_w^T A_w + alpha^2 L^T L)^+ A_w^T
                 = Q1 Z (C^2 + alpha^2 S^2)^-1 Z^T Q1^T ,

        tr H  = sum_i c_i^2 / (c_i^2 + alpha^2 s_i^2) ,
        tr H^2 = sum_i [ c_i^2 / (c_i^2 + alpha^2 s_i^2) ]^2 ,

    because Z^T Q1^T Q1 Z = C^2. These are Hansen's general-form Tikhonov filter factors
    f_i = gamma_i^2 / (gamma_i^2 + alpha^2) at the generalized singular values
    gamma_i = c_i / s_i (Hansen 1998 sec. 4.2, the unnumbered display printed p. 73, for
    L != I; the GSVD itself Eqs. (2.6)-(2.7), printed p. 22), written so that no division
    by a small s_i is needed. The n - p directions in the null space of L take f_i = 1
    (Hansen 1998 Eq. (4.5), printed p. 72): here they are exactly the s_i = 0 directions,
    which the construction produces exactly rather than by a tolerance.

    This is CONTIN's own route. Provencher's s_j (1982a Eq. (A.15), printed p. 225, and
    Eq. (A.23), printed p. 226) are the same generalized singular values, reached there
    through the regularizor's SVD (Eq. (A.7), printed p. 225) with its null space
    PERTURBED "to at least a small fraction of the largest singular value" to make the
    transformation invertible -- where the pair's own range keeps those directions exact.
    NDF = sum_j s_j^2 / (s_j^2 + alpha^2) (Eqs. (3.15)-(3.16), printed p. 219) is sum f_i,
    and Eq. (A.25), printed p. 226, gives filter factor 0 to the directions an
    under-determined design cannot see -- which is what c_i = 0 gives here.

    Parameters
    ----------
    A_w : the WEIGHTED design M^.5 A, so the trace agrees with the weighted V(alpha).
    L : the regularizer, which may have zero rows (then every f_i is 1).
    """
    if not (np.all(np.isfinite(A_w)) and np.all(np.isfinite(L))):
        return None
    m = int(A_w.shape[0])
    K = np.vstack([A_w, L])
    U, sv, _Vt = np.linalg.svd(K, full_matrices=False)
    # numpy's own `matrix_rank` tolerance, spelled out because the rank IS the basis:
    # a direction below it contributes a filter factor that is noise either way, and
    # including it would put a denormal s_i in the gamma_i = c_i/s_i ratio. The leading
    # singular value is tested for FINITENESS as well as sign: a finite design can still
    # saturate it (measured, numpy 2.4.6: a 2x2 of 1e308 decomposes to s = [inf, 0] with
    # U finite), and an infinite scale leaves no trustworthy tolerance, so no direction
    # is admitted and the filter sum reports the floor dof every consumer handles.
    if sv.size == 0 or not (math.isfinite(sv[0]) and sv[0] > 0.0):
        r = 0
    else:
        r = int(np.count_nonzero(sv > sv[0] * max(K.shape) * np.finfo(float).eps))
    Q = U[:, :r]
    Q1, Q2 = Q[:m], Q[m:]
    _U2, s2v, Zt = np.linalg.svd(Q2, full_matrices=True)
    Z = Zt.T
    # full_matrices=True is what makes the null space exact: Z spans all r directions,
    # and the r - min(p, r) trailing ones are the null space of L inside the range, where
    # s_i = 0 identically rather than to a tolerance.
    sin = np.zeros(r)
    sin[:s2v.size] = s2v
    cos = np.linalg.norm(Q1 @ Z, axis=0)
    return _TikhonovFilterBasis(cos=cos, sin=sin)


def _tikhonov_filter_factors(basis: _TikhonovFilterBasis,
                             alpha: float) -> np.ndarray:
    """Per-direction Tikhonov filter factors f_i at this alpha, from one basis.

    f_i = c_i^2 / (c_i^2 + (alpha s_i)^2), the general-form factors of
    `_tikhonov_filter_basis`'s derivation. Two-sided by construction (CLAUDE.md rule 12),
    and the spelling is the guard:

      * the penalty is `np.square(alpha * sin)`, never `alpha ** 2 * sin ** 2`: a PYTHON
        float `**` RAISES OverflowError where a numpy `square` saturates, and squaring
        alpha on its own overflows decades before the product a tiny s_i actually forms
        does;
      * alpha = 1e+300 gives (alpha s_i) = inf wherever s_i > 0, hence f_i = 0 there and
        f_i = 1 on the null space of L (alpha * 0 = 0);
      * alpha = 1e-300 underflows the penalty to 0, giving f_i = 1 where c_i > 0;
      * a zero denominator (c_i = 0 and s_i = 0, a direction outside both ranges) is
        written as f_i = 0 rather than 0/0, so the array is never NaN for a finite alpha.
    """
    c2 = basis.cos * basis.cos
    pen = np.square(alpha * basis.sin)
    denom = c2 + pen
    return np.divide(c2, denom, out=np.zeros_like(c2), where=denom > 0)


def _tikhonov_effective_dof(basis: Optional[_TikhonovFilterBasis],
                            alpha: float) -> float:
    """Effective degrees of freedom of the Tikhonov solution at this alpha.

    The regularized least-squares solution has the linear "hat" (influence) matrix
    H = A (A^T A + alpha^2 L^T L)^+ A^T; its trace is the effective number of free
    parameters -- the honest degrees of freedom, NOT the raw grid size (Hansen 1998; the
    number of grid points is "to a large extent arbitrary", Provencher 1982). It is the
    sum of the filter factors, so it decreases smoothly from rank(A_w) toward 0 as alpha
    grows (more smoothing = fewer effective parameters) and is NON-INCREASING in alpha by
    construction: each f_i is. For the ill-conditioned Laplace-inversion kernel it stays
    well below the data count, so the F-test degrees of freedom are always well defined.
    The non-negativity constraint can only lower it further, so this is a mild upper bound.

    Takes the basis, not the matrices: every alpha of a sweep reads ONE decomposition
    (Provencher 1982a's own arrangement, printed p. 226: the SVDs "are done only once for
    the whole series"). The retired route formed the normal matrix A^T A + alpha^2 L^T L
    per alpha and solved it, which Hansen 1998 sec. 5.1, printed p. 101, warns against on
    both counts -- "forming A^T A explicitly can lead to loss of information in
    finite-precision arithmetic, and a new Cholesky factorization is required for each
    regularization parameter" -- and whose error is bounded by cond(M) * eps, the SQUARE
    of the design's own condition number.

    Returns NaN when the basis could not be built (a non-finite design or regularizer).
    The 1e-6 floor keeps a positive, defined dof for a design of rank 0, where the sum is
    exactly 0 and every consumer divides by something derived from it.
    """
    if basis is None:
        return float('nan')
    return float(np.clip(_tikhonov_filter_factors(basis, alpha).sum(), 1e-6, None))


def _tikhonov_influence_mu2(basis: Optional[_TikhonovFilterBasis],
                            alpha: float, n_data: int) -> float:
    """Lukas's influence measure mu_2 = n^-1 tr H^2 at this alpha.

    The variance term of the robust GCV functional `_gcv_corner` minimises (Lukas 2010
    Eq. (1.4), printed p. 521; Lukas 2006, whose spectral form is Eq. (2.10), printed
    p. 1887: mu_2 = n^-1 [m + sum_i (lambdabar_i / (lambdabar_i + lambda))^2], m the
    dimension of the penalty's null space). H is the Tikhonov hat matrix, so tr H^2 is the
    sum of the SQUARED filter factors over every direction of the basis -- the null space
    of L at f_i = 1, which is Lukas's m, and a direction the design cannot see at
    f_i = 0 (Provencher 1982a Eq. (A.25), printed p. 226).

    DIMENSIONLESS, which is the property the selector rests on, and its reach stated
    exactly: mu_2 depends on alpha only through the filter factors, so no scaling of lambda
    enters it (exact), and the robust selection's planted verdict did not change across
    40-, 100- and 150-point Rh grids (measured, the Theory-and-Equations Guide's CONTIN
    section). It is a trace divided by n, the number of fitted channels, so at a fixed
    amount of smoothing it scales as 1/n: the variance term's weight does depend on the
    delay window, skipped channels and the instrument's channel layout, modestly (the
    robust factor at the pick moved from 0.143 to 0.155 as a planted draw's window was cut
    from 183 to 144 channels, measured 2026-09-22).

    Bounded two-sidedly by construction (CLAUDE.md rule 12): each f_i lies in [0, 1]
    (`_tikhonov_filter_factors`, finite at alpha = 1e-300 and 1e+300 alike), so mu_2 lies
    in [0, r / n] with r the basis's direction count. Returns NaN when the basis could not
    be built -- the withhold every consumer already handles. `n_data` is the number of
    fitted lag channels, Lukas's n.
    """
    if basis is None:
        return float('nan')
    f = _tikhonov_filter_factors(basis, alpha)
    return float(np.sum(f * f) / n_data)


def _ftest_reference_index(residual_norms, dof_eff) -> int:
    """The sweep index of the F-test's REFERENCE solution.

    Reference alpha_0 = the solution with the MINIMUM residual (the least-squares end;
    normally the smallest alpha, but argmin also covers Provencher's note that a numerical
    instability can push it to a slightly larger alpha), chosen among alphas whose
    effective dof is USABLE. NDF0 is taken from the reference and appears in both the
    scaling factor and the F-distribution, so a reference whose hat-trace could not be
    computed makes every fc NaN -- and since `np.argmin` returns the index of a NaN, the
    criterion would then select that same failed alpha. Masking the objective alone cannot
    fix it: the poison enters through the reference. The last-resort branch is masked for
    the same reason: a bare `argmin(v)` would select an alpha whose RESIDUAL is NaN.

    One helper, so the selector and the manual's precondition on this same solution
    (`_ftest_precondition_notes`) can only ever read ONE reference -- a note that named a
    different solution than the statistic was measured against would describe a fit that
    never happened (CLAUDE.md rule 12, guard and operation from one computation).
    """
    v = np.asarray(residual_norms, dtype=float)
    dof = np.asarray(dof_eff, dtype=float)
    ref_pool = np.where(np.isfinite(v) & np.isfinite(dof), v, np.inf)
    return int(np.argmin(ref_pool if np.any(np.isfinite(ref_pool)) else _masked(v)))


def _ftest_precondition_notes(alphas, residual_norms, solution_norms, dof_eff, fc,
                              opt_idx: int, *, agreement: float, band,
                              level: float = 0.5) -> tuple:
    """The F-test's two manual preconditions as `(reference note, band note)`, '' each
    where the precondition held.

    Both come from the CONTIN Users Manual's own checklist for a CONTIN run (§3.6.5,
    printed p. 17), on quantities the sweep already holds -- neither costs a solve.

    (1) "The VARIANCE of the Reference Solution ... should agree with OBJ. FCTN. in the
    first few figures. If not, alpha is not small enough". Provencher 1982 (Part II)
    printed p. 237 says what those two printed columns are: VARIANCE = V(0) of Part I
    Eq. (3.9), the weighted residual sum of squares of that solution, and OBJ. FCTN. =
    V(alpha) of Eq. (3.10), that same residual plus alpha^2 ||r - R x||^2. Our regularizor
    has r = 0, so the two differ by exactly alpha_0^2 ||L x_0||^2 and "agree in the first
    few figures" is that penalty being negligible beside the residual:

        alphas[ref]^2 * ||L x_ref||^2 / V_res(ref)  <=  agreement

    (2) is not checked here: it asks the user to scale the t_k and g_m so the solution
    does not overflow, which is a choice of units on the input rather than a test on the
    finished fit.

    (3) "If the CHOSEN SOLUTION ... does not have a PROB1 between about 0.05 and 0.95,
    see Sec 4.1.3" -- the chosen solution's PROB1 outside `band` says the sweep does not
    bracket the transition the test selects in, so the pick is at the end of a curve
    rather than on its slope. Read on the CHOSEN solution, `fc[opt_idx]`, as the manual
    states it -- and only when the requested `level` lies strictly inside the band.
    `_ftest_corner` picks nearest the REQUESTED level, so a level at or beyond a band
    edge (0.04, 0.96, the edges themselves) puts the pick outside the band on every fit
    by the user's own construction; firing there would name an unbracketed transition
    when the sweep bracketed it (a proxy, invariant 4 clause 1, and policing the user's
    disclosed choice, clause 2), so the note stays silent for such a level, which is
    recorded on the result. At CONTIN's own PLEVEL = 0.5 the check is exactly "some
    sweep point's PROB1 lies inside the band"; at any other interior level it is not (a
    point inside the band need not be nearer the level than one outside), which is why
    the chosen point is consulted rather than the sweep.

    A precondition that cannot be EVALUATED did not hold: a non-finite ratio (an
    unusable reference residual) or a NaN PROB1 fires the note it belongs to, never a
    silent pass. Each note names its own precondition, and a caller rolls both up with
    the end state's note through `_result_flags`.
    """
    ref = _ftest_reference_index(residual_norms, dof_eff)
    a_ref = float(np.asarray(alphas, dtype=float)[ref])
    residual = float(np.asarray(residual_norms, dtype=float)[ref])
    penalty = float(np.asarray(solution_norms, dtype=float)[ref])
    # `a_ref * a_ref`, never `a_ref ** 2`: a_ref is a PYTHON float here, and `**` RAISES
    # OverflowError past ~1e154 where `*` saturates to inf -- a raise happens before the
    # comparison below can decide anything (CLAUDE.md rule 12). Both tails are real: a
    # saturated ratio fires the note (the penalty is anything but negligible) and an
    # underflowed one clears it, and a residual that is zero or non-finite leaves the
    # ratio NaN, which fires.
    ratio = ((a_ref * a_ref) * penalty / residual
             if (math.isfinite(residual) and residual > 0.0) else float('nan'))
    lo, hi = float(band[0]), float(band[1])
    fc_curve = np.asarray(fc, dtype=float) if fc is not None else np.array([np.nan])
    # The chosen solution's PROB1, read when the level lies strictly inside the band. A
    # NaN compares False on both sides, so a curve that could not be read fires the note
    # -- a precondition that cannot be evaluated did not hold -- and an index the curve
    # does not hold reads as NaN for the same reason.
    idx = int(opt_idx)
    chosen = float(fc_curve[idx]) if 0 <= idx < fc_curve.size else float('nan')
    level_inside = lo < float(level) < hi
    chosen_inside = bool(lo <= chosen <= hi)
    band_failed = level_inside and not chosen_inside
    return (FTEST_REFERENCE_NOTE if not (ratio <= agreement) else '',
            FTEST_BAND_NOTE_TEMPLATE.format(lo=lo, hi=hi) if band_failed else '')


def _ftest_corner(residual_norms, dof_eff, n_data: int,
                  prob_reject: float = 0.5):
    """Provencher's F-test ("probability to reject") alpha selection.

    Implements the original criterion of Provencher (1982a, "A constrained
    regularization method...", Comput. Phys. Commun. 27:213), Eqs. (3.23)-(3.24).
    Among the sweep of increasingly-smoothed solutions, judge how significant each
    solution's residual increase over the least-squares reference is, and pick the one
    at the chosen probability level. For each alpha the F-statistic is

        F(alpha) = [ (V(alpha) - V0) / V0 ] * (Ny - NDF0) / NDF0             (Eq. 3.24)

    with V(alpha) = ||A x - y||^2, and V0 the MINIMUM residual over the sweep, at the
    reference alpha_0 (the least-regularized / least-squares end). Ny is the number of
    data points and NDF0 = NDF(alpha_0) the effective degrees of freedom AT that
    reference -- Provencher's NDF = sum_j s_j^2/(s_j^2 + alpha^2) (Eqs. 3.15-3.16),
    which is exactly the trace of the Tikhonov hat matrix (`_tikhonov_effective_dof`,
    computed here from the one decomposition of the weighted design -- Provencher's own
    arrangement, and his s_j are that decomposition's generalized singular values),
    NOT the raw grid size. NDF0 is a single fixed value (taken at alpha_0), used in
    BOTH the scaling factor and the F-distribution -- not the per-alpha DOF. F is
    F-distributed with (NDF0, Ny - NDF0) degrees of freedom, so its cumulative value

        fc(alpha) = P[F(alpha); NDF0, Ny - NDF0]                             (Eq. 3.23)

    is Provencher's PROB1, the "probability to reject": ~0 where the regularizer barely
    changes the fit (rough, under-smoothed) and ~1 where it changes it a lot (Provencher:
    "only when PROB1 > ~0.9 are there significant grounds to suspect alpha may be too
    large"). The chosen solution is the one whose fc is closest to `prob_reject`
    (Provencher's default 0.5). A HIGHER prob_reject tolerates more fit degradation ->
    selects a LARGER alpha (smoother, more parsimonious); a LOWER one a smaller alpha
    (rougher). For an ill-posed Laplace kernel NDF0 is small (Provencher: "surprisingly
    small even with accurate data"), so the F-test dof are always well defined.

    Returns (index, fc_array). fc_array (PROB1 per alpha) is returned for the result/plot.
    The two preconditions the criterion's own manual attaches to a run of it -- on this
    same reference solution, and on the chosen solution's PROB1 -- are checked separately
    in `_ftest_precondition_notes`.

    Caveat (documented in the guide): the F-test assumes independent residuals, but a
    single correlogram's lag channels are correlated (Schaetzel 1990), so Ny overstates
    the independent information and the level is a guide, not an exact test -- why the
    default is a selector built for correlated errors (robust GCV, `fit_contin`).
    """
    v = np.asarray(residual_norms, dtype=float)
    dof = np.asarray(dof_eff, dtype=float)
    ny = float(n_data)
    # The reference alpha_0 (and why it is masked) is `_ftest_reference_index`, shared
    # with the manual's precondition on that same solution so the two cannot name
    # different ones.
    ref = _ftest_reference_index(v, dof)
    v0 = float(v[ref])
    ndf0 = float(np.clip(dof[ref], 1e-6, ny - 1e-6))   # NDF(alpha_0), fixed
    # An unusable reference residual yields NaN, NOT zeros. Substituting zeros turns the
    # poison into a perfectly finite F-statistic of 0 for every alpha, so `fc` becomes a
    # fabricated flat curve, `objective` is 0.5 everywhere, the "criterion did not answer"
    # branch below never fires, and the tie resolves to index 0 -- the "index 0 by
    # accident" the sibling selector explicitly disowns -- and the fabricated curve would
    # also be stored on `LCurveResult.ftest_fc` and plotted as a real PROB1.
    frac_increase = np.clip((v - v0) / v0 if (math.isfinite(v0) and v0 > 0)
                            else np.full_like(v, np.nan), 0.0, None)
    f_stat = frac_increase * (ny - ndf0) / ndf0
    fc = special.fdtr(ndf0, ny - ndf0, f_stat)         # F CDF, dof (NDF0, Ny-NDF0)
    # As in `_gcv_corner`: a NaN would WIN this argmin, so an alpha whose effective dof
    # could not be computed would be the one selected. Masked to a never-wins +inf. If
    # NDF0 itself is unanswerable every fc is NaN, and the fallback is the reference alpha
    # (the least-squares end this criterion is defined against), not index 0 by accident.
    objective = np.abs(fc - prob_reject)
    if not np.any(np.isfinite(objective)):
        return ref, fc
    return int(np.argmin(_masked(objective))), fc


def _contin_sweep_and_select(A: np.ndarray, y: np.ndarray, w: np.ndarray,
                             L: np.ndarray, alphas: np.ndarray,
                             alpha_method: str, ftest_prob_reject: float,
                             gcv_gamma: float, ftest_reference_agreement: float,
                             ftest_prob_band) -> tuple:
    """Solve the whole alpha sweep under `w` and apply `alpha_method`'s selector.

    Returns `(solutions, residual_norms, solution_norms, x_norms, dof_eff, ftest_fc,
    gcv, opt_idx, state, precondition_notes)`, where `state` is the selected alpha's END
    STATE -- the one value `fit_contin` turns into both edge flags and the reliability
    note, so no selector can set a flag and a note that disagree. `gcv` is the score the
    GCV-family selector MINIMIZED: the classical functional at `gcv_gamma` = 1, Lukas's
    robust one (RGCV) below it. `precondition_notes` are the selector's OWN preconditions,
    checked on the finished sweep: the F-test has the two its manual prescribes, and the
    other two selectors have none, so the tuple is empty for them.

    Factored out because the sourced weight runs the sweep TWICE -- a first pass under the
    g1-squared weight supplies the weight's g1 factor at the same selector, then the fit
    itself runs at the anchored weight (module docstring) -- and the two passes must be one
    body, not two that happen to agree.
    """
    n_alpha = int(alphas.size)
    residual_norms = np.empty(n_alpha)
    solution_norms = np.empty(n_alpha)
    x_norms = np.empty(n_alpha)
    solutions = []
    for i, a in enumerate(alphas):
        x = _solve_distribution(A, y, w, alpha=a, L=L)
        solutions.append(x)
        residual_norms[i] = np.sum(w * (A @ x - y) ** 2)   # weighted V(alpha)
        solution_norms[i] = _regularization_seminorm_sq(L, x)   # ||L x||^2 (diagnostic)
        x_norms[i] = float(np.sum(x ** 2))                 # ||x||^2 (the L-curve's axis)

    dof_eff = ftest_fc = gcv_curve = None
    precondition_notes: tuple = ()
    if alpha_method in ('gcv', 'ftest'):
        # Both GCV and the F-test need the Tikhonov hat-trace, which must reflect the
        # WEIGHTED design (the same M^.5 A the solve minimizes over) so it agrees with
        # the weighted V(alpha) (Provencher 1982a Eq. 3.9; Golub 1979 Eq. 1.4). ONE
        # decomposition serves the whole sweep, so the trace is monotone in alpha by
        # construction and no normal matrix is ever formed.
        Aw, _yw = _apply_weight(A, y, w)
        basis = _tikhonov_filter_basis(Aw, L)
        dof_eff = np.array([_tikhonov_effective_dof(basis, a) for a in alphas])
    if alpha_method == 'gcv':
        mu2 = np.array([_tikhonov_influence_mu2(basis, a, int(y.size))
                        for a in alphas])
        opt_idx, gcv_curve = _gcv_corner(residual_norms, dof_eff, n_data=int(y.size),
                                         gamma=gcv_gamma, mu2=mu2)
        state = _end_state_of_index(opt_idx, n_alpha)
    elif alpha_method == 'ftest':
        opt_idx, ftest_fc = _ftest_corner(
            residual_norms, dof_eff, n_data=int(y.size),
            prob_reject=ftest_prob_reject)
        state = _end_state_of_index(opt_idx, n_alpha)
        # Checked AFTER the selection, because one of the two preconditions is about the
        # solution the selection chose. Both read the arrays this sweep already produced.
        precondition_notes = _ftest_precondition_notes(
            alphas, residual_norms, solution_norms, dof_eff, ftest_fc, opt_idx,
            agreement=ftest_reference_agreement, band=ftest_prob_band,
            level=ftest_prob_reject)
    else:
        opt_idx, state = _lcurve_corner(alphas, residual_norms, x_norms)
    return (solutions, residual_norms, solution_norms, x_norms, dof_eff, ftest_fc,
            gcv_curve, opt_idx, state, precondition_notes)


def fit_contin(
    measurement: DLSMeasurement,
    rh_min_nm: float = 1.0,
    rh_max_nm: float = 1000.0,
    n_grid: int = 100,
    alpha: Optional[float] = None,
    alpha_min: float = 1e-4,
    alpha_max: float = 1e2,
    n_alpha: int = 40,
    beta: Optional[float] = None,
    baseline: Optional[float] = None,
    tau_min_s: Optional[float] = None,
    tau_max_s: Optional[float] = None,
    skip_initial_channels: int = 0,
    alpha_method: str = 'gcv',
    ftest_prob_reject: float = 0.5,
    ftest_reference_agreement: float = 1e-3,
    ftest_prob_band: tuple = (0.05, 0.95),
    gcv_gamma: float = 0.1,
    weighting: Optional[str] = None,
) -> ContinResult:
    """Recover a smoothed decay-rate distribution by regularized inversion.

    Solves min ||A x - y||^2 + alpha^2 ||L x||^2 subject to x >= 0, with L the
    second-difference operator (Provencher 1982a Eq. 3.10; implemented as the CONTIN
    program of Provencher 1982). The regularization parameter
    alpha trades fit quality against distribution smoothness.

    If alpha is None (default), a sweep over [alpha_min, alpha_max] is run and alpha
    is chosen automatically by `alpha_method`; the full sweep is returned so the
    choice can be inspected and overridden. If alpha is given, that value is used
    directly and a single-point sweep is returned for consistency.

    Three automatic selectors share the same sweep (only the selection differs):
      * 'gcv' (default): the Generalized Cross-Validation family. At the default
        `gcv_gamma` = 0.1 it is Lukas's ROBUST GCV (RGCV; Lukas 2006, Lukas 2010
        Eq. (1.4)): GCV's functional times a factor that weighs the solution's variance,
        the literature's repair for GCV's under-smoothing when the errors are correlated,
        as a correlogram's lag channels are. At `gcv_gamma` = 1 it is classical GCV
        (Golub, Heath & Wahba 1979), minimizing n V(alpha)/(n - dof(alpha))^2. The
        selector is recorded as 'gcv' either way, with its gamma beside it.
      * 'lcurve': the L-curve corner = point of maximum curvature of the log-log
        residual-vs-||x|| curve (Hansen 1998 Eq. 7.24), located on the discrete sweep
        by the adaptive pruning algorithm of Hansen, Jensen & Rodriguez 2007.
      * 'ftest': Provencher's original F-test / "probability to reject" criterion
        (Provencher 1982a; Scotti et al. 2015), picking the solution whose residual
        increase over the least-regularized fit sits at the `ftest_prob_reject`
        significance level (default 0.5). Higher -> smoother, lower -> rougher.

    Parameters
    ----------
    measurement : DLSMeasurement
    rh_min_nm, rh_max_nm, n_grid : grid specification (default 1-1000 nm, 100 pts)
    alpha : float, optional
        Fixed regularization parameter. If omitted, chosen by `alpha_method`.
    alpha_min, alpha_max, n_alpha :
        Log-spaced alpha sweep (default 1e-4 to 1e2, 40 points), refused if inverted.
        The 40-point sweep gives the L-curve's pruning locator its full ladder of
        resolutions and lets GCV resolve its interior minimum. The floor is a
        documented default (CLAUDE.md invariant 4), derived
        rather than picked: CONTIN's own sweep is scale-relative, from machine
        precision times the largest singular value up to that value (CONTIN Users
        Manual §4.1.3), and its "alpha small enough" test is that the smallest-alpha
        solution's residual agrees with the unregularized one in the first few figures
        (§3.6.5 point 1). 1e-4 is the highest floor at which that still holds on the
        committed reference correlograms (measured 2026-08-24 over n_grid 15-1000: the
        residual at 1e-4 exceeds the unregularized one by <= 1e-5 relative on every
        cell, while at 1e-3 regularization is already acting), so a floor pick is the
        unregularized reference by construction and is flagged as such. A finer Rh grid
        (n_grid) is what drives the F-test and GCV onto the floor; the flag says when
        it happened. Pinned by tests/test_dls.py::test_alpha_floor_is_the_unregularized_reference
        and the floor-flag tests beside it.
    beta, baseline : float, optional
        Coherence factor and baseline; estimated from the data if omitted.
    tau_min_s, tau_max_s : float, optional
        Inclusive delay-time window. Default uses all points.
    alpha_method : str
        'gcv' (default), 'lcurve', or 'ftest'. Ignored if `alpha` is given.
    ftest_prob_reject : float
        F-test significance level (Provencher default 0.5). Only used for 'ftest'.
    ftest_reference_agreement : float
        The relative size at which the reference solution's penalty counts as negligible
        beside its residual -- the F-test's first precondition, which its own manual asks
        a user to check on every run: the reference solution's VARIANCE "should agree with
        OBJ. FCTN. in the first few figures. If not, alpha is not small enough" (CONTIN
        Users Manual §3.6.5 point 1, printed p. 17). Those two printed columns are V(0)
        and V(alpha) of Provencher 1982a Eqs. (3.9) and (3.10) (Provencher 1982 Part II,
        printed p. 237), which differ by alpha^2 ||L x||^2 alone, so the check is that
        ratio against this level. The default 1e-3 reads "the first few figures" as three;
        a documented parameter rather than a constant (CLAUDE.md invariant 4). Only used
        for 'ftest'; a failure flags the result and names the precondition, it does not
        withhold the distribution.
    ftest_prob_band : tuple
        (low, high) probability-to-reject band the chosen solution must land in -- the
        F-test's second precondition: "If the CHOSEN SOLUTION ... does not have a PROB1
        between about 0.05 and 0.95, see Sec 4.1.3" (§3.6.5 point 3, same page), whose two
        printed numbers are this default. Outside it the sweep does not bracket the
        transition the criterion selects in. Read on the chosen solution when
        `ftest_prob_reject` lies strictly inside the band; a level at or beyond an edge
        places the pick there by construction and is the user's disclosed choice, not
        flagged. Only used for 'ftest'.
    gcv_gamma : float
        The robustness parameter of Lukas's ROBUST GCV (RGCV), on (0, 1]. Only used for
        'gcv'. The default 0.1 is the value all three sources print (Lukas 2006, printed
        pp. 1885-1886; Lukas 2010, printed p. 543; Bauer & Lukas 2011 Eq. (33)); 1 is
        classical GCV. Below 1 the selector adds the solution's variance, mu_2 =
        n^-1 tr H^2, to what it minimizes and under-smooths less readily on correlated
        data (Lukas 2010 Eq. (1.4), printed p. 521). mu_2 is dimensionless, so the printed
        value needs no normalization of its own, though the term's weight scales with the
        number of fitted channels
        (`_tikhonov_influence_mu2`). RGCV at 0.1 is the default selector, with the
        L-curve and the F-test offered beside it (owner-ruled 2026-09-22 on its sources;
        the planted check is in the Theory-and-Equations Guide's CONTIN section).
    weighting : str, optional
        Which residual weight to fit under, one of `DISTRIBUTION_WEIGHTINGS`; None means
        `DISTRIBUTION_WEIGHTING_DEFAULT`, and the form that ran is on
        `.distribution.weighting`. Under 'biganzoli' the whole sweep AND its selection run
        twice: a first pass under the g1-squared weight, at this same selector or this
        same fixed alpha, supplies the weight's g1 factor, and the anchored weight then
        reaches the solve, the weighted V(alpha), the effective-dof row scaling and the
        recorded residual norm alike (module docstring).

    Returns
    -------
    ContinResult
        .distribution is the chosen DistributionResult; .lcurve holds the sweep, whose
        .gcv is the score the GCV family minimized and whose .dof_eff comes from one
        decomposition per weighted design; .alpha_selection_method records which
        selector chose alpha and .gcv_gamma the robustness parameter it ran at;
        .alpha_at_ceiling flags a high-alpha-end pick (the selector wanted a larger
        alpha than the range allows);
        .alpha_at_floor flags a low-alpha-end pick (the selector wanted a smaller
        alpha than the range allows); .alpha_reliable is False whenever the selector's
        end state or one of its own preconditions carries a note, and .reliability_note
        is those sentences -- the end state's first, then each failed precondition -- the
        one string every surface shows. The two edge flags follow the end state alone: a
        failed F-test precondition is a statement about the sweep, not about which end
        of it the pick sits on.
    """
    if alpha_method not in ('gcv', 'lcurve', 'ftest'):
        raise ValueError(
            f"alpha_method must be 'gcv', 'lcurve', or 'ftest', got {alpha_method!r}.")
    # Both ends (rule 12), as the alpha guard below: the sources define the robustness
    # parameter on the OPEN interval (0, 1) (Lukas 2010 below Eq. (1.4); Bauer & Lukas
    # 2011 Eq. (34)), and 1 is admitted as the classical-GCV endpoint they both name.
    # gamma = 0 would drop the misfit entirely and is refused with the rest.
    if not (math.isfinite(gcv_gamma) and 0.0 < gcv_gamma <= 1.0):
        raise ValueError(
            f"gcv_gamma must be a finite number in (0, 1], got {gcv_gamma!r}: it is the "
            f"robustness parameter of the robust GCV functional, and 1 is classical GCV.")
    # The F-test's two preconditions are parameters, not constants, so they take the same
    # shape of guard as the bounds below. The agreement level is a RELATIVE size, hence
    # finite and > 0 (zero would demand exact agreement, which no floating-point sweep
    # reaches); the band is two probabilities in order and inside [0, 1], since a band no
    # PROB1 can satisfy would fire its note on every fit and so say nothing at all.
    if not (math.isfinite(ftest_reference_agreement) and ftest_reference_agreement > 0.0):
        raise ValueError(
            f"ftest_reference_agreement must be a finite number > 0, got "
            f"{ftest_reference_agreement!r}: it is the relative size at which the "
            f"reference solution's penalty counts as negligible beside its residual.")
    prob_band = tuple(ftest_prob_band)
    if not (len(prob_band) == 2 and all(math.isfinite(b) for b in prob_band)
            and 0.0 <= prob_band[0] < prob_band[1] <= 1.0):
        raise ValueError(
            f"ftest_prob_band must be two finite probabilities (low, high) with "
            f"0 <= low < high <= 1, got {ftest_prob_band!r}: it is the band the chosen "
            f"solution's probability to reject must lie in.")
    # Both tails (rule 12): a bare `alpha > 0` admits +inf (an infinite regularizer)
    # and lets NaN through, so the guard names finiteness as the siblings do.
    if alpha is not None and not (math.isfinite(alpha) and alpha > 0):
        raise ValueError(
            f"alpha must be a finite number > 0, got {alpha!r}. A non-positive alpha "
            f"disables the Tikhonov regularization entirely (see fit_nnls for that "
            f"unregularized case).")
    resolved = _resolve_distribution_weighting(weighting)
    (tau, g2m1, baseline, baseline_est, beta, beta_est,
     q, rh_grid, gamma_grid, A, y, w, dead_note) = _prepare_distribution_inputs(
        measurement, tau_min_s, tau_max_s, beta, baseline,
        rh_min_nm, rh_max_nm, n_grid,
        skip_initial_channels=skip_initial_channels)
    recorded, sd, weight_note = _sourced_sigma_or_fallback(
        resolved, measurement, tau, tau_min_s, tau_max_s, skip_initial_channels, 'CONTIN')

    L = _second_difference_operator(n_grid)

    if alpha is not None:
        # User-fixed alpha: solve once -- twice under the sourced weight, the first pass
        # at THIS SAME alpha, so the weight's g1 factor comes off a spectrum of the same
        # smoothness as the fit it weights.
        x = _solve_distribution(A, y, w, alpha=alpha, L=L)
        if sd is not None:
            try:
                w = _sourced_weight(A, x, sd, w)
            except ValueError as exc:
                recorded = 'g1-squared'
                weight_note = _weight_fallback_note('CONTIN', str(exc))
            else:
                x = _solve_distribution(A, y, w, alpha=alpha, L=L)
        st_prov, dur_form, rate_note = _distribution_weight_disclosure(
            recorded, measurement)
        dist = _distribution_summary(
            x, A, y, w, rh_grid, gamma_grid, beta, baseline, g2m1, tau, q,
            method='contin', alpha=alpha,
            beta_estimated=beta_est, baseline_estimated=baseline_est,
            n_skipped=skip_initial_channels,
            notes=tuple(n for n in (dead_note, weight_note, rate_note) if n),
            weighting=recorded,
            sample_time_provenance=st_prov, run_duration_form=dur_form)
        lcurve = LCurveResult(
            alphas=np.array([alpha]),
            residual_norms=np.array([dist.residual_norm]),
            solution_norms=np.array([_regularization_seminorm_sq(L, x)]),
            optimal_alpha=alpha, optimal_index=0,
            # `solution_norm` IS ||x||^2 (`_distribution_summary`), so the L-curve's axis
            # is read off the result rather than recomputed: one arithmetic, one value.
            x_norms=np.array([dist.solution_norm]))
        return ContinResult(distribution=dist, lcurve=lcurve,
                            alpha_was_user_supplied=True,
                            alpha_selection_method='user',
                            notes=dist.notes)

    # Sweep (shared by all three selectors -- only the selection function differs).
    # The bounds are the fixed alpha's family -- and the two the GUI exposes as free
    # text -- so they take the same finite-and-positive guard: a NaN or negative bound
    # would otherwise ride np.geomspace onto the result as the recorded alpha, under
    # the sweep-edge note (a proxy, not the failed precondition).
    if not (math.isfinite(alpha_min) and math.isfinite(alpha_max)
            and alpha_min > 0 and alpha_max > 0):
        raise ValueError(
            f"alpha_min / alpha_max must be finite and > 0, got ({alpha_min!r}, "
            f"{alpha_max!r}): the alpha sweep cannot be built from them.")
    # An INVERTED pair is refused rather than swept: np.geomspace runs it descending, on
    # which "floor" and "ceiling" name the opposite ends of the range the user typed.
    if alpha_min > alpha_max:
        raise ValueError(
            f"alpha_min must not exceed alpha_max, got alpha_min={alpha_min!r} and "
            f"alpha_max={alpha_max!r}: the alpha sweep runs from the floor upward.")
    alphas = np.geomspace(alpha_min, alpha_max, n_alpha)
    (solutions, residual_norms, solution_norms, x_norms, dof_eff, ftest_fc, gcv_curve,
     opt_idx, state, precondition_notes) = _contin_sweep_and_select(
        A, y, w, L, alphas, alpha_method, ftest_prob_reject, gcv_gamma,
        ftest_reference_agreement, prob_band)
    if sd is not None:
        # The whole sweep above ran under the g1-squared weight and its selector picked
        # from it: that is Maier's intermediate spectrum, taken at the same selector as
        # the fit it weights. A refusal leaves the first pass standing as the result, so
        # the fallback IS the g1-squared fit, sweep arrays and edge flags included.
        try:
            w = _sourced_weight(A, solutions[opt_idx], sd, w)
        except ValueError as exc:
            recorded = 'g1-squared'
            weight_note = _weight_fallback_note('CONTIN', str(exc))
        else:
            (solutions, residual_norms, solution_norms, x_norms, dof_eff, ftest_fc,
             gcv_curve, opt_idx, state,
             precondition_notes) = _contin_sweep_and_select(
                A, y, w, L, alphas, alpha_method, ftest_prob_reject, gcv_gamma,
                ftest_reference_agreement, prob_band)
    opt_alpha = float(alphas[opt_idx])
    x_opt = solutions[opt_idx]
    # Sweep-edge picks, both read off the selector's ONE end state (so the two can never
    # both fire on a sweep with an interior, and no second threshold exists). Ceiling:
    # the selector wanted a larger alpha than the range allows (for GCV its flat-minimum
    # failure mode, Hansen 1998 p.185; for the L-curve a corner above the window).
    # Floor: essentially unregularized NNLS at the default floor (see the fit_contin
    # docstring on alpha_min for what reaches it), whose extra peaks are inversion
    # noise. The note comes from the same state through the family's own assembler, so
    # flag and sentence are one call.
    alpha_at_floor = state in ('floor', 'both')
    alpha_at_ceiling = state in ('ceiling', 'both')
    # The end state's note first, then each of the selector's own failed preconditions:
    # independently failed preconditions are each stated, and neither implies the other
    # (CLAUDE.md invariant 4, clause 1). The two edge FLAGS above stay the end state's
    # alone -- a precondition is about the sweep, not about which end the pick sits on.
    alpha_reliable, reliability_note = _result_flags(
        _ALPHA_STATE_NOTES.get(state, ''), *precondition_notes)

    st_prov, dur_form, rate_note = _distribution_weight_disclosure(recorded, measurement)
    dist = _distribution_summary(
        x_opt, A, y, w, rh_grid, gamma_grid, beta, baseline, g2m1, tau, q,
        method='contin', alpha=opt_alpha,
        beta_estimated=beta_est, baseline_estimated=baseline_est,
        n_skipped=skip_initial_channels,
        notes=tuple(n for n in (dead_note, weight_note, rate_note) if n),
        weighting=recorded,
        sample_time_provenance=st_prov, run_duration_form=dur_form)
    lcurve = LCurveResult(
        alphas=alphas, residual_norms=residual_norms,
        solution_norms=solution_norms, optimal_alpha=opt_alpha,
        optimal_index=opt_idx, dof_eff=dof_eff, ftest_fc=ftest_fc, gcv=gcv_curve,
        x_norms=x_norms)
    return ContinResult(distribution=dist, lcurve=lcurve,
                        alpha_was_user_supplied=False,
                        alpha_selection_method=alpha_method,
                        ftest_prob_reject=(ftest_prob_reject
                                           if alpha_method == 'ftest' else None),
                        gcv_gamma=(gcv_gamma if alpha_method == 'gcv' else None),
                        alpha_at_ceiling=alpha_at_ceiling,
                        alpha_at_floor=alpha_at_floor,
                        alpha_reliable=alpha_reliable,
                        reliability_note=reliability_note,
                        notes=dist.notes)


# ---------------------------------------------------------------------------
# Rh <-> Gamma axis helper (for the visualization toggle and general use)
# ---------------------------------------------------------------------------

def distribution_axis(distribution: DistributionResult, axis: str = 'rh'):
    """Return (x_values, weights, x_label) for plotting a DistributionResult.

    Supports the Rh <-> Gamma visualization toggle. The same weights are plotted
    against either the hydrodynamic-radius grid or the decay-rate grid; only the
    x-axis changes. Values are returned sorted ascending in the chosen axis.

    Parameters
    ----------
    distribution : DistributionResult
    axis : str
        'rh' (default) -> x is hydrodynamic radius (nm).
        'gamma'        -> x is decay rate (s^-1).

    Returns
    -------
    (x_values, weights, x_label)
    """
    if axis == 'rh':
        x = distribution.rh_grid_nm
        label = 'Hydrodynamic radius Rh (nm)'
    elif axis == 'gamma':
        x = distribution.gamma_grid_s_inv
        label = 'Decay rate Gamma (s^-1)'
    else:
        raise ValueError(f"axis must be 'rh' or 'gamma', got {axis!r}.")
    order = np.argsort(x)
    return x[order], distribution.weights[order], label


# ---------------------------------------------------------------------------
# Distribution peak detection (for multi-population samples)
# ---------------------------------------------------------------------------

@dataclass
class DistributionPeak:
    """One resolved population in a CONTIN/NNLS size distribution.

    `weight_fraction` is the fraction of the total (intensity-weighted) area under
    that peak's lobe -- so the peaks of a distribution sum to ~1. It is an
    intensity fraction, not a number or mass fraction (large particles scatter far
    more). `is_dominant` marks the largest-area peak.

    `rh_lo_nm` / `rh_hi_nm` are the Rh values at the first and last grid points of
    the peak's lobe (the integration bounds the fraction was summed over --
    inclusive on both ends). NaN when the producing path predates the bounds
    (additive fields, defaulted).
    """
    rh_nm: float                      # Rh at the peak maximum
    gamma_s_inv: float                # decay rate at the peak maximum
    weight_fraction: float            # intensity-weighted area of the peak's lobe
    is_dominant: bool
    rh_lo_nm: float = float('nan')    # Rh at the lobe's first grid point
    rh_hi_nm: float = float('nan')    # Rh at the lobe's last grid point


def scan_distribution_peaks(rh_grid_nm, weights, min_weight_fraction: float = 0.05):
    """Resolve the peaks of a weight-vs-Rh distribution, weighting-agnostic.

    Returns a list of ``(peak_index, weight_fraction, lo_idx, hi_idx)`` on the Rh-SORTED
    grid, so the caller can build peak objects in whatever weighting the ``weights`` carry
    (the same scan serves the intensity distribution AND its reweighted volume/surface/
    number views — one definition, per the fix-the-family rule). ``peak_index`` indexes
    the Rh-ascending arrays returned in ``rh_sorted``; ``lo_idx``/``hi_idx`` are the
    peak's lobe bounds as a half-open slice ``[lo_idx, hi_idx)`` of those arrays — the
    exact range the fraction was summed over (mode-resolved scattering integrates a
    lobe between these bounds, so they are returned rather than discarded).

    Returns ``(peaks, rh_sorted, w_sorted)``: peaks is ``[(idx, frac, lo, hi), ...]``
    (empty if the distribution is degenerate), rh_sorted / w_sorted the ascending-Rh
    arrays the indices refer to.
    """
    rh = np.asarray(rh_grid_nm, dtype=float)
    w = np.asarray(weights, dtype=float)
    if rh.size == 0:
        return [], rh, w
    order = np.argsort(rh)
    rh, w = rh[order], w[order]
    total = float(w.sum())
    if not (math.isfinite(total) and total > 0):
        return [], rh, w

    n = w.size
    # Local maxima, including the endpoints if they rise above their neighbor. A
    # flat-topped peak (a run of consecutive EQUAL heights, e.g. [...,1,1,...]) is
    # ONE population, not two: scan by runs of equal weight and report the run's
    # center, so an exact plateau isn't split into two spurious peaks.
    maxima: List[int] = []
    i = 0
    while i < n:
        if not (math.isfinite(w[i]) and w[i] > 0):
            i += 1
            continue
        j = i
        while j + 1 < n and w[j + 1] == w[i]:   # extend the equal-height run [i..j]
            j += 1
        left_ok = (i == 0) or (w[i] >= w[i - 1])
        right_ok = (j == n - 1) or (w[j] >= w[j + 1])
        strictly = ((i > 0 and w[i] > w[i - 1]) or (j < n - 1 and w[j] > w[j + 1]))
        if left_ok and right_ok and strictly:
            maxima.append((i + j) // 2)         # center of the plateau (grid index)
        i = j + 1
    if not maxima:
        maxima = [int(np.argmax(w))]

    # lobe boundaries: the local minimum between each pair of adjacent maxima
    bounds = [0]
    for a, b in zip(maxima[:-1], maxima[1:], strict=True):
        bounds.append(a + int(np.argmin(w[a:b + 1])))
    bounds.append(n)

    peaks = []
    for k, peak_i in enumerate(maxima):
        lo, hi = bounds[k], bounds[k + 1]
        frac = float(w[lo:hi].sum() / total)
        if frac < min_weight_fraction:
            continue
        peaks.append((peak_i, frac, lo, hi))
    return peaks, rh, w


def validate_integration_bounds(bounds_nm):
    """Validate manual integration intervals; returns them as float pairs.

    THE one predicate home (rule 12: one boundary, not two to keep aligned):
    both `integrate_between` (at use) and the controller's `set_mode_bounds`
    (at user-input time) call this, so what the store accepts and what the
    integration refuses cannot drift apart. Raises ValueError on a non-finite
    or inverted interval, and on overlap -- including TOUCHING intervals
    (hi_a == lo_b), because bounds are inclusive both ends, so even touching
    would count a shared grid point twice.
    """
    pairs = [(float(lo), float(hi)) for lo, hi in bounds_nm]
    for lo, hi in pairs:
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(
                f'integration bounds must be finite with lo < hi, got ({lo}, {hi}).')
    for (lo_a, hi_a), (lo_b, hi_b) in zip(sorted(pairs), sorted(pairs)[1:], strict=False):
        if hi_a >= lo_b:
            raise ValueError(
                f'integration bounds overlap: ({lo_a}, {hi_a}) and ({lo_b}, {hi_b}) — '
                'overlapping intervals would count the shared weight twice.')
    return pairs


def integrate_between(rh_grid_nm, weights, bounds_nm):
    """Integrate a weight-vs-Rh distribution over user-set Rh intervals, weighting-agnostic.

    The manual-bounds counterpart of `scan_distribution_peaks`' automatic lobes: when
    overlapping populations defeat the local-minimum split, the user supplies explicit
    ``bounds_nm = [(lo_nm, hi_nm), ...]`` and each interval's summed weight becomes that
    mode's fraction of the total. Works on any weighting (intensity or reweighted views),
    exactly like the scan — one definition for the whole distribution family.

    Interval semantics: grid points with ``lo_nm <= rh <= hi_nm`` belong to the interval
    (inclusive both ends — the values are user-facing Rh, not slice indices). Intervals
    must be well-formed (``lo < hi``) and must not overlap each other: a split that counts
    a grid point twice writes fractions that sum past 1 — a false record, refused with a
    ``ValueError``. An interval that captures NO weight is a *noted* result (fraction 0.0,
    ``rh_at_max_nm`` NaN), not an error: an empty window is honest information about where
    the user cut.

    Returns a list of ``(rh_at_max_nm, weight_fraction, lo_idx, hi_idx)`` in the order of
    ``bounds_nm``, on the Rh-ascending grid — the same tuple shape as the scan, with the
    peak position taken as the weight maximum inside the interval. ``lo_idx``/``hi_idx``
    are the half-open slice of the sorted arrays. Fractions are relative to the FULL
    distribution total (not the covered subset), so a partial cover discloses itself as
    fractions summing below 1. Returns ``(intervals, rh_sorted, w_sorted)``; empty
    ``intervals`` if the distribution is degenerate (empty grid or non-positive total).
    """
    rh = np.asarray(rh_grid_nm, dtype=float)
    w = np.asarray(weights, dtype=float)
    pairs = validate_integration_bounds(bounds_nm)
    order = np.argsort(rh)
    rh, w = rh[order], w[order]
    if rh.size == 0:
        return [], rh, w
    total = float(w.sum())
    if not (math.isfinite(total) and total > 0):
        return [], rh, w
    intervals = []
    for lo_nm, hi_nm in pairs:
        inside = np.nonzero((rh >= lo_nm) & (rh <= hi_nm))[0]
        if inside.size == 0:
            intervals.append((float('nan'), 0.0, 0, 0))
            continue
        lo_idx, hi_idx = int(inside[0]), int(inside[-1]) + 1
        seg = w[lo_idx:hi_idx]      # NaN weights never reach here: a NaN total is degenerate above
        frac = float(seg.sum() / total)
        if frac <= 0.0:             # grid points but no weight — noted, not an error
            intervals.append((float('nan'), 0.0, lo_idx, hi_idx))
            continue
        peak_i = lo_idx + int(np.argmax(seg))
        intervals.append((float(rh[peak_i]), frac, lo_idx, hi_idx))
    return intervals, rh, w


def find_distribution_peaks(
    distribution: DistributionResult,
    min_weight_fraction: float = 0.05,
) -> List[DistributionPeak]:
    """Resolve the distinct populations (peaks) in a size distribution.

    Local maxima of the (intensity-weighted) distribution are found on the Rh grid;
    each peak's lobe runs to the local minima on either side, and the lobe's summed
    weight is its intensity fraction. Peaks contributing less than
    `min_weight_fraction` of the total are dropped as noise. The returned peaks are
    sorted by Rh ascending; exactly one is flagged `is_dominant` (the largest area).

    A monomodal distribution returns a single peak. This is the basis for offering a
    CONTIN/NNLS peak Rh as a rho source and for letting the user pick a peak when a
    sample is multi-population.

    A success=False result has NO peaks: its curve is the seed estimate, and a peak
    of a failed fit is a fit-like number no caller may present. The refusal lives
    here, in the one scan every consumer routes through, so no call site needs its
    own gate.
    """
    if getattr(distribution, 'success', True) is False:
        return []
    gamma = np.asarray(distribution.gamma_grid_s_inv, dtype=float)
    rh_in = np.asarray(distribution.rh_grid_nm, dtype=float)
    # A gamma grid whose length disagrees with the Rh grid is a malformed/degenerate result;
    # the pre-refactor code co-sorted the two and raised IndexError there (caught upstream as
    # "no peaks"). Preserve that: return [] rather than emitting NaN-gamma peaks that could
    # surface a NaN-decay-rate population in the table or as a rho source.
    if gamma.size != rh_in.size:
        return []
    scanned, rh, _w = scan_distribution_peaks(
        rh_in, distribution.weights, min_weight_fraction)
    if not scanned:
        return []
    # Re-sort gamma to match the Rh-ascending arrays the scan returned.
    gamma = gamma[np.argsort(rh_in)]
    peaks: List[DistributionPeak] = []
    for peak_i, frac, lo, hi in scanned:
        peaks.append(DistributionPeak(
            rh_nm=float(rh[peak_i]), gamma_s_inv=float(gamma[peak_i]),
            weight_fraction=frac, is_dominant=False,
            rh_lo_nm=float(rh[lo]), rh_hi_nm=float(rh[hi - 1])))
    if not peaks:
        return []
    dominant = max(range(len(peaks)), key=lambda j: peaks[j].weight_fraction)
    peaks[dominant].is_dominant = True
    return peaks
