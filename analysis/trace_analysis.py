"""
analysis/trace_analysis.py
==========================

Intensity-trace signal-analysis diagnostics. Given a count-rate trace
(IntensityTrace), these assess data quality: summary statistics, robust baseline
estimation, shot-noise outlier flagging (globally, and against a moving window),
baseline normalization, sliding-window running averages, blocking
(standard-error-vs-block-size and Allan-deviation) analysis, count-rate
histograms with Gaussian/Poisson fits and the Fano factor, a formal Augmented
Dickey-Fuller stationarity test, a Welch power spectral density with its
shot-noise floor, and periodicity significance from the plain periodogram.

The three frequency/stability views answer questions the rest cannot. The PSD and
the periodogram ask whether anything in the trace is PERIODIC -- pump or stirrer
vibration, temperature-controller cycling, mains harmonics -- which no
amplitude-domain diagnostic here can see. The Allan deviation asks how long the
trace may usefully be AVERAGED before averaging stops paying, which is the
blocking analysis read on its stability axis rather than its standard-error one.

(Promoted out of analysis/utilities.py into its own module: this cluster has no code
coupling to the rest of utilities -- the SLS I*sin(theta) diagnostic, rho = Rg/Rh,
the scaling power-law, the result-candidate picker, and the synthetic generator
all stay in utilities.py.)

Design contract (consistent with the rest of the platform)
----------------------------------------------------------
Every function here is a PURE function: it takes data objects and parameters and
returns a result object. No function draws a plot, writes a file, or mutates its
inputs.

Units
-----
All inputs are assumed to be in the canonical internal units defined in
core/data_models.py (count rate in cps, time in seconds). The functions operate on
IntensityTrace objects.

Optional dependency
-------------------
test_stationarity_adf() requires statsmodels. It is imported lazily inside that
function, so the rest of the module works without statsmodels installed. If you
want the stationarity test, run:  pip install statsmodels

Why scipy.signal is imported at MODULE level, unlike that one
-------------------------------------------------------------
`scipy.signal` is imported at the top of this file on purpose, and the inversion
matters. A first-time scipy.signal import running on a background thread while
PySide6's shiboken import hook is live can corrupt the heap (0xc0000374 --
reproduced and bisected here; see `preload_stationarity_backend`). The GUI
dispatches these diagnostics to a worker thread, so a lazy import inside
`welch_psd` would put exactly that import on exactly that thread. Importing at
module level means it always happens wherever this module is first imported,
which for the app is the controller on the main thread at startup. That leaves no
preload call site for a future caller to forget -- the gap is removed rather than
policed. statsmodels stays lazy because it is genuinely OPTIONAL; scipy is a hard
runtime dependency either way, so nothing is bought by deferring it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import optimize, signal, stats

from core.data_models import IntensityTrace
from core.notes import Note


# ===========================================================================
# Shared constants
# ===========================================================================

# THE single source for the robust-band multiplier. `app.controller` and the
# Traces tab seed from this rather than restating the number, so the three sites
# cannot drift apart (they had drifted 1.0 / 3.0 / 3.0 once already).
OUTLIER_K_DEFAULT = 3.5

# Scales the MAD to a Gaussian sigma. `flag_outliers` still spells this literal
# inline; that function is deliberately left byte-identical apart from its `k`
# default, so the duplication is a scope decision, not drift.
MAD_TO_SIGMA = 1.4826

# THE single source for the blocking analysis's correlation verdict. A general
# heuristic, not a system-specific threshold: `block_variance` returns the whole
# curve so a reader can judge for themselves. It is here rather than inline
# because the verdict it decides now feeds `periodogram_false_alarm`'s
# `null_assumption_violated`, so a drifted copy would change a reliability flag
# and not just a label -- the same reason OUTLIER_K_DEFAULT sits above.
BLOCKVAR_THRESHOLD_DEFAULT = 1.5

# Moving-window Hampel bounds. Only the FLOOR is derived; read
# `resolve_hampel_window` for what each number rests on before changing either.
HAMPEL_WINDOW_MIN_PTS = 11
HAMPEL_WINDOW_MAX_PTS = 101

# Memory budget for one block of the rolling median/MAD. A PERFORMANCE knob, not
# a guard: every value produces bit-identical output (pinned by
# test_chunked_and_single_pass_window_statistics_agree), so there is no boundary
# here to mis-size. See flag_outliers_windowed.
_HAMPEL_CHUNK_BYTES = 16 << 20

# Elements per block of the point-window running average's interior rows: a tuning
# constant for speed and memory. The per-slice pin named in running_average holds
# the result bitwise at this size.
_RUNNING_AVERAGE_BLOCK_ELEMENTS = 2 ** 16

# Welch PSD segment bounds, in points. Neither is canonical: Welch (1967) treats
# the segment length L purely as "a scale parameter" trading resolution against
# variance and prints NO rule for choosing it. Read `resolve_psd_segment` for
# what each rests on before changing either.
PSD_SEGMENT_MIN_PTS = 16
PSD_SEGMENT_MAX_PTS = 4096

# The three offered data windows; see `_psd_window` for their forms and for why
# 'welch-w2' is named that rather than 'parzen'. 'welch-w1' and 'welch-w2' are
# Welch's own two, so his printed E.D.F. figures apply to them verbatim; 'hann'
# is the modern default and is not one of his, so its E.D.F. comes from his
# general rho(j) formula instead (`welch_edf`).
PSD_WINDOWS = ('hann', 'welch-w1', 'welch-w2')

# Default false-alarm probability for the periodogram peak test. A reporting
# convention, not a Scargle value: he prints the false-alarm formula (Eq. 14,
# p. 839) and no preferred level. 1% is the conventional "significant peak"
# threshold and is user-owned.
FALSE_ALARM_PROBABILITY_DEFAULT = 0.01

# Smallest trace statsmodels' adfuller(..., autolag='AIC') will accept with its
# default regression. Measured, not read from statsmodels' source: probed with
# random, linear, alternating and near-duplicate traces of n = 2..30, autolag
# 'AIC' -- every pattern raised "sample size is too short" below n = 4 and none
# did at or above it, so 4 is the floor for that call shape specifically.
ADF_MIN_POINTS = 4


# ===========================================================================
# Result objects
# ===========================================================================

@dataclass
class TraceStatistics:
    """Summary statistics for an intensity trace."""
    n_points: int
    duration_s: float
    mean_cps: float
    std_cps: float
    cv: float                      # coefficient of variation = std / mean
    minimum_cps: float
    maximum_cps: float
    baseline_cps: float            # from the chosen baseline method
    baseline_method: str
    sigma_clipped_mean_cps: float


@dataclass
class BaselineResult:
    """Estimated baseline intensity and the method used to find it."""
    baseline_cps: float
    method: str                    # 'percentile' or 'sigma_clip'
    parameter: float               # percentile value, or n_sigma
    n_points_used: int             # how many points contributed (sigma_clip)


@dataclass
class OutlierFlags:
    """Points falling outside median +/- k * 1.4826*MAD (the robust band)."""
    flagged_mask: np.ndarray       # bool array, True where flagged
    n_flagged: int
    fraction_flagged: float
    centre_cps: float              # the median count rate (the robust centre)
    scale_cps: float               # envelope half-width at k = 1 (1.4826 * MAD)
    mad_cps: float                 # raw median absolute deviation, before the 1.4826 factor
    scale_is_degenerate: bool      # MAD imploded to 0 -> no scale exists, nothing flagged
    k: float
    lower_bound_cps: float
    upper_bound_cps: float


@dataclass
class WindowedOutlierFlags:
    """Moving-window Hampel decomposition: rolling baseline + impulse flags.

    A keep-and-flag decomposition of the trace against a MOVING baseline, the
    local companion to `flag_outliers`' global band. Nothing is removed,
    replaced or smoothed: the output is curves and flags, never an altered
    trace.

    Where the layer has no local verdict -- no full window fits, or the
    window's MAD imploded -- the rolling arrays carry NaN. That is a RENDERING
    of `local_verdict_mask`, which is the authoritative record; a plot or an
    export consuming the NaNs gets the gap for free.
    """
    # Per-point arrays, all length n and aligned with the trace's own axis.
    times_s: np.ndarray                    # copy of the trace time axis
    rolling_centre_cps: np.ndarray         # window median m_k; NaN where not judged
    rolling_scale_cps: np.ndarray          # MAD_TO_SIGMA * window MAD (S_k); NaN where not judged
    flagged_mask: np.ndarray               # bool; |I - m_k| > k*S_k. False wherever not judged
    local_verdict_mask: np.ndarray         # bool; AUTHORITATIVE record of where a verdict exists
    edge_abstained_mask: np.ndarray        # bool; no full window fits (the first/last K points)
    implosion_abstained_mask: np.ndarray   # bool; a full window fits but its MAD is 0
    # Summary counts.
    n_points: int
    n_judged: int
    n_flagged: int
    # Fraction OF THE JUDGED POINTS, not of n: over n, the flag rate would fall
    # as the layer's coverage degraded, so an abstaining diagnostic would look
    # cleaner than a working one. (`OutlierFlags.fraction_flagged` divides by n
    # because the global band never abstains per-point -- the difference is
    # deliberate.) n_judged and n_points are both here, so either is recoverable.
    fraction_flagged: float
    fraction_abstained: float              # (n - n_judged) / n
    fraction_abstained_edge: float         # of n
    fraction_abstained_implosion: float    # of n
    # Event census: MAXIMAL runs of consecutive flagged samples. Maximal is the
    # honest primitive and it has a visible consequence -- one unflagged sample
    # inside a real event splits it in two, so a 20-sample step can census as
    # two events. Merging across gaps would need a tolerance no source
    # prescribes; `flagged_mask` is on the result, so a consumer that wants one
    # can apply its own.
    n_events: int
    event_start_indices: np.ndarray        # int; first flagged sample of each run
    event_lengths_pts: np.ndarray          # int; samples per run (exact, assumes no cadence)
    event_spans_s: np.ndarray              # float; t[last] - t[first]; a 1-sample event spans 0 s
    longest_event_pts: int
    # Parameters, echoed back for provenance and export.
    k: float
    window_pts: int
    window_half_width_k: int               # Pearson's K = (window_pts - 1) // 2
    window_floor_pts: int
    window_description: str                # e.g. '101 points (K = 50)'
    window_was_resolved: bool              # True when the caller passed no window
    mad_to_sigma: float


@dataclass
class NormalizedTraceResult:
    """A trace normalized by its baseline (fluctuates around 1.0)."""
    times_s: np.ndarray
    normalized: np.ndarray         # count_rates / baseline
    baseline_cps: float
    baseline_method: str


@dataclass
class RunningAverageResult:
    """Sliding-window mean and standard deviation of a trace."""
    times_s: np.ndarray
    running_mean: np.ndarray
    running_std: np.ndarray
    window_description: str        # e.g. '30.0 s' or '15 points'


@dataclass
class BlockVarianceResult:
    """Blocking analysis: standard error of the mean, and Allan deviation, vs block size.

    For an uncorrelated (white-noise) trace the standard error is independent
    of block size. A standard error that rises with block size and plateaus is
    the signature of positive correlations -- e.g. slow-mode fluctuations. The
    plateau value is the corrected standard error of the mean.

    The Allan deviation is the SAME partition read differently. Both views are
    built from one sequence of block means, so they can never disagree about how
    the trace was cut; the standard error asks how precise the mean is at that
    block size, the Allan deviation asks how REPRODUCIBLE consecutive blocks are
    -- which is what says how long the trace may usefully be averaged.
    """
    block_sizes: np.ndarray        # number of points per block
    standard_errors: np.ndarray    # SE of the mean estimated at each block size
    n_blocks: np.ndarray           # how many blocks at each size
    se_ratio: float                # SE(largest block) / SE(smallest block)
    correlation_threshold: float
    correlations_detected: bool    # se_ratio > correlation_threshold
    # --- Allan deviation (Barnes et al. 1971 Eq. (11), p. 109) ---------------
    # sigma_y(tau) is FRACTIONAL: the block means are divided by the trace mean
    # before differencing, matching Barnes's y (a fractional deviation, his
    # Eq. 7 p. 109), so his printed tau-slope mappings apply verbatim. mean_cps
    # is carried so the absolute deviation is recoverable and nothing is hidden.
    tau_s: np.ndarray              # averaging time of each block = block_size * dt_s
    allan_deviation: np.ndarray    # sigma_y(tau), dimensionless (fractional)
    allan_n_pairs: np.ndarray      # m, the number of differences averaged at each tau
    mean_cps: float                # trace mean, the fractional normalizer
    dt_s: float                    # sample spacing used to convert block size -> tau


@dataclass
class HistogramFitResult:
    """Count-rate histogram with optional Gaussian and Poisson fits.

    The histogram is built in count-rate space (cps) for display. The Poisson
    fit, which is physically defined on integer photon counts, is computed by
    converting rates to counts using the integration time, then mapped back
    onto the rate axis for overlay. The Fano factor (variance/mean in count
    space) is the key shot-noise diagnostic: ~1 for an ideal Poisson process,
    >1 for excess fluctuations (slow modes, dust).
    """
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    counts: np.ndarray             # histogram heights (per bin)
    bin_method: str
    n_bins: int
    integration_time_s: float
    integration_time_estimated: bool
    # mean / variance in count space
    mean_counts: float
    variance_counts: float
    fano_factor: float             # variance_counts / mean_counts
    cv: float                      # std/mean of count rates
    shot_noise_cv: float           # 1/sqrt(mean_counts), the Poisson expectation
    # Gaussian fit (None if not requested or failed)
    gaussian_params: Optional[dict] = None         # {'amp','mu_cps','sigma_cps'}
    gaussian_curve: Optional[np.ndarray] = None    # evaluated at bin_centers
    gaussian_chi2_reduced: Optional[float] = None
    # Poisson fit (None if not requested or failed)
    poisson_lambda_counts: Optional[float] = None
    poisson_curve: Optional[np.ndarray] = None     # evaluated at bin_centers (rate space)
    poisson_chi2_reduced: Optional[float] = None


@dataclass
class StationarityResult:
    """Augmented Dickey-Fuller stationarity test result."""
    adf_statistic: float
    p_value: float
    critical_values: dict          # e.g. {'1%': ..., '5%': ..., '10%': ...}
    significance: float
    is_stationary: bool            # p_value < significance


@dataclass
class WelchPSDResult:
    """Welch-averaged power spectral density of a count-rate trace.

    The averaged-modified-periodogram estimator of Welch (1967): the record is
    cut into overlapping segments, each windowed and transformed, and the
    resulting periodograms averaged. Averaging is what makes the estimate
    readable -- for a Gaussian process whose spectrum is flat across the
    estimator's passband, Welch's stated conditions (p. 71), a single modified
    periodogram has variance P^2, 100% relative scatter, no matter how long the
    record, and twice that variance at zero frequency and Nyquist (p. 72) -- and
    `edf` is how much of that scatter is left.

    The flat `shot_floor_psd` is where an ideal, dead-time-free Poisson detector
    would sit. Broadband power ABOVE it is excess fluctuation (the same excess
    the Fano factor totals, resolved by frequency); a narrow peak above it is a
    candidate periodicity, whose significance comes from
    `periodogram_false_alarm` and NOT from this curve.
    """
    frequencies_hz: np.ndarray
    psd: np.ndarray                # one-sided power spectral density, cps^2/Hz
    # Welch's equivalent degrees of freedom of the approximating chi-square
    # distribution (p. 72). Reported rather than a bare "confidence": it is the
    # single number that says how much averaging the estimate actually received.
    edf: float
    ci_lower_factor: float         # multiply psd by these for the confidence band
    ci_upper_factor: float
    confidence_level: float
    # The Poisson reference, cps^2/Hz. NaN when the trace mean is not positive:
    # a shot-noise floor is genuinely undefined with no counts (the same call
    # HistogramFitResult makes for the Fano factor).
    shot_floor_psd: float
    mean_cps: float
    # Parameters, echoed back for provenance and export.
    nperseg: int
    noverlap: int
    n_segments: int                # K, the number of segments averaged
    window: str
    dt_s: float
    segment_was_resolved: bool     # True when nperseg came from the default rule


@dataclass
class PeriodogramResult:
    """Classical periodogram with Scargle's peak false-alarm probability.

    Deliberately NOT the Welch estimate. Scargle's exponential null (his
    Eqs. 12-13, p. 839) holds for the classical periodogram normalized by the
    data variance sigma_0^2, and the false-alarm formula is built on that null;
    the Welch-averaged estimate is chi-square with `edf` degrees of freedom
    instead, so the same line drawn against it would be answering a different
    question. The two views therefore carry two honest statements rather than
    one conflated one.
    """
    frequencies_hz: np.ndarray
    power: np.ndarray              # z = P_X(omega) / sigma_0^2, dimensionless
    n_independent: int             # N = N_0 / 2 (Scargle Eq. 5, p. 836)
    # The strongest peak and what it is worth.
    peak_frequency_hz: float
    peak_power: float              # z_max
    false_alarm_probability: float # Pr{Z > z_max} = 1 - [1 - exp(-z_max)]^N
    # The level a peak must clear for the requested false-alarm probability.
    significance_level: float      # z_0 solving Pr{Z > z_0} = requested
    requested_false_alarm: float
    peak_is_significant: bool      # peak_power > significance_level
    # Scargle's null is white AND Gaussian. A correlated trace violates it and
    # manufactures significant-looking power at low frequency, so the verdict is
    # reported WITH this flag rather than silently. Supplied by the caller (from
    # the blocking analysis) so this function stays pure and single-purpose.
    null_assumption_violated: bool
    variance_cps2: float           # sigma_0^2, the normalizer
    dt_s: float
    # `null_not_white` when the null is violated, about the three readings computed
    # under it (core.notes); () otherwise.
    notes: Tuple[Note, ...] = ()


# ===========================================================================
# Intensity-trace diagnostics
# ===========================================================================

def _sigma_clipped_mean(
    values: np.ndarray,
    n_sigma: float = 3.0,
    max_iters: int = 5,
) -> Tuple[float, int]:
    """Iteratively reject points > n_sigma from the mean, return (mean, n_used).

    A robust mean estimator: on each iteration, points more than n_sigma
    standard deviations from the current mean are removed, and the mean and
    std are recomputed. Iteration stops when no further points are removed or
    max_iters is reached.
    """
    data = np.asarray(values, dtype=float)
    kept = data.copy()
    for _ in range(max_iters):
        if kept.size < 2:
            break
        mu = kept.mean()
        sd = kept.std(ddof=1)
        if not (math.isfinite(sd) and sd != 0):
            break
        mask = np.abs(kept - mu) <= n_sigma * sd
        if mask.all():
            break
        kept = kept[mask]
    return float(kept.mean()) if kept.size else float('nan'), int(kept.size)


def compute_trace_statistics(
    trace: IntensityTrace,
    baseline_method: str = 'percentile',
    baseline_parameter: float = 25.0,
) -> TraceStatistics:
    """Compute summary statistics for an intensity trace.

    Parameters
    ----------
    trace : IntensityTrace
        The count-rate trace to analyze.
    baseline_method : str
        'percentile' (default) or 'sigma_clip'. See identify_baseline.
    baseline_parameter : float
        The single value passed through to identify_baseline for either method;
        its default is 25.0. For 'percentile' that is the percentile. For
        'sigma_clip' it is the number of sigma, so pass a typical ~3.0 explicitly
        (leaving the default gives a very loose 25-sigma clip, not 3).

    Returns
    -------
    TraceStatistics
    """
    y = trace.count_rates_cps
    mean = float(y.mean())
    std = float(y.std(ddof=1)) if y.size > 1 else 0.0
    # A degenerate all-zero (or zero-mean) trace has no meaningful relative spread;
    # report cv = 0 rather than leak a bare nan from 0/0 into the statistics.
    cv = std / mean if mean != 0 else 0.0
    baseline = identify_baseline(trace, method=baseline_method,
                                 parameter=baseline_parameter).baseline_cps
    sc_mean, _ = _sigma_clipped_mean(y)
    duration = float(trace.times_s[-1] - trace.times_s[0]) if y.size > 1 else 0.0
    return TraceStatistics(
        n_points=int(y.size),
        duration_s=duration,
        mean_cps=mean,
        std_cps=std,
        cv=cv,
        minimum_cps=float(y.min()),
        maximum_cps=float(y.max()),
        baseline_cps=baseline,
        baseline_method=baseline_method,
        sigma_clipped_mean_cps=sc_mean,
    )


def identify_baseline(
    trace: IntensityTrace,
    method: str = 'percentile',
    parameter: float = 25.0,
) -> BaselineResult:
    """Estimate the baseline intensity of a trace.

    The baseline is the 'true' scattering level of a well-behaved sample,
    estimated robustly so that upward dust spikes do not bias it.

    Parameters
    ----------
    trace : IntensityTrace
    method : str
        'percentile' (default): the baseline is the given lower percentile of
            the count rates. Resistant to upward outliers.
        'sigma_clip': the baseline is the sigma-clipped mean (iteratively
            reject points more than `parameter` sigma from the mean).
    parameter : float
        For 'percentile': the percentile value (default 25.0). Lower values
            are more conservative (closer to the minimum).
        For 'sigma_clip': the number of sigma for clipping (e.g. 3.0).

    Returns
    -------
    BaselineResult

    Raises
    ------
    ValueError
        If method is unrecognized, or a percentile is outside [0, 100].
    """
    y = trace.count_rates_cps
    if method == 'percentile':
        if not (0.0 <= parameter <= 100.0):
            raise ValueError(
                f"percentile must be in [0, 100], got {parameter!r}."
            )
        baseline = float(np.percentile(y, parameter))
        return BaselineResult(baseline_cps=baseline, method='percentile',
                              parameter=parameter, n_points_used=int(y.size))
    elif method == 'sigma_clip':
        baseline, n_used = _sigma_clipped_mean(y, n_sigma=parameter)
        return BaselineResult(baseline_cps=baseline, method='sigma_clip',
                              parameter=parameter, n_points_used=n_used)
    else:
        raise ValueError(
            f"Unknown baseline method {method!r}. "
            f"Use 'percentile' or 'sigma_clip'."
        )


def flag_outliers(
    trace: IntensityTrace,
    k: float = OUTLIER_K_DEFAULT,
) -> OutlierFlags:
    """Flag points falling outside median +/- k * (robust sigma).

    The centre is the MEDIAN and the scale is 1.4826*MAD (the median absolute
    deviation, scaled so it equals sigma for Gaussian data). A point is flagged
    when |I - median| > k * scale. This is the Hampel identifier, equivalently
    Iglewicz & Hoaglin's modified z-score, whose recommended cutoff D = 3.5 is
    the default k here.

    Both halves earn their place, and both replaced a defect (2026-08-10):

    * The **median centre** is not dragged by the outliers it is looking for. A
      mean centre is: four dust spikes on an otherwise flat trace pull the mean
      past its own envelope, so every CLEAN sample flags as well.
    * The **MAD scale** is set by the trace's own scatter, so the envelope
      adapts to whatever noise the trace actually carries. The Poisson envelope
      this replaced, mean +/- k*sqrt(mean), assumed shot-noise-limited data.
      Real traces frequently are not -- measured across the committed ALV data,
      the observed scatter runs 14-376x the Poisson expectation, and against
      such a trace a Poisson envelope flags essentially every sample.

    IMPLOSION, and why nothing is patched over it. When more than half the
    samples share one value -- a flat, stalled or heavily quantized trace --
    the MAD is exactly 0. Pearson et al. (2016) sec. 3 names this an *implosion
    window* and proves the characterization; in one, they note, the filter's
    output stops depending on the threshold and reduces to the median filter
    (p. 2). They prescribe no treatment for it, neither a scale floor nor a
    substitute estimator, and name the *implosion-free* sequences (min S_k > 0)
    as the ones of greatest interest. Abstaining is this program's own choice:
    the result flags NOTHING and reports `scale_is_degenerate`.

    That has a real cost, and it is deliberate: a single dust spike on a
    PERFECTLY FLAT trace is not flagged either. With over half the samples tied
    there is no scale to judge "stands out" against, and the limit is inherent
    rather than a shortcoming of the MAD -- Rousseeuw & Croux's S_n and Q_n,
    proposed precisely as better scale estimators, carry the same 50% implosion
    breakdown point and collapse on the same input. The alternative is worse:
    a zero-width band marks every sample that is not exactly the median, which
    on a settling trace means flagging a third of it as dust.

    This function briefly floored the scale at sqrt(centre) to dodge that case
    (2026-08-10). The floor was removed the same day: it is dimensionally wrong
    for count RATES (the Poisson sigma of a rate is sqrt(rate/dt), not
    sqrt(rate)), it has no source -- no reference in this project's library
    prescribes one -- and where it bound it produced exactly the spurious flags
    it was meant to prevent. Do not reinstate it.

    Scope, and what this does NOT claim. It is a POINT-outlier test: it asks
    which samples stand out against the rest of THIS trace. It does not say
    whether the trace as a whole is noisier than photon statistics permit --
    the Fano factor (`fit_count_rate_histogram`) is that diagnostic, and a
    trace can be far from shot-noise-limited while flagging nothing here. It
    also flags baseline departures of any duration WITHOUT localizing them: a
    slow drift trips it much as a single dust spike does. That is the gap
    `flag_outliers_windowed` fills -- the same identifier against a MOVING
    baseline, which localizes impulses and shows the baseline it judged them
    against. The two are companions, not alternatives, and they share one k.

    Parameters
    ----------
    trace : IntensityTrace
    k : float
        Multiplier on the robust sigma defining the envelope half-width
        (default 3.5, Iglewicz & Hoaglin's D).

    Returns
    -------
    OutlierFlags

    Raises
    ------
    ValueError
        If k <= 0, if the median count rate is negative or non-finite (not a
        physical count rate, and the centre of the band), or if the envelope
        bounds are not finite (k * scale saturated).
    """
    if not (math.isfinite(k) and k > 0):
        raise ValueError(f"k must be a positive finite number, got {k!r}.")
    y = trace.count_rates_cps
    centre = float(np.median(y))
    if not (math.isfinite(centre) and centre >= 0):
        raise ValueError(
            f"Median count rate must be a non-negative finite number, got {centre}; "
            f"it is the centre of the band and a count rate cannot be negative."
        )
    mad = float(np.median(np.abs(y - centre)))
    scale = 1.4826 * mad
    # Implosion: over half the samples share a value, so the MAD -- and the scale -- is 0.
    # Decided HERE, beside the scale it describes, so no caller has to re-derive it from a
    # float comparison on `scale_cps`; that would be a second predicate to mis-size (rule 12).
    # `not (scale > 0.0)` also routes a NaN scale here, which is the honest destination for
    # it; +inf correctly does NOT come here, and the bounds guard below is what catches that.
    scale_is_degenerate = not (scale > 0.0)
    if scale_is_degenerate:
        lower = upper = centre
        mask = np.zeros_like(y, dtype=bool)
    else:
        # Guard the two numbers the comparison below actually uses, not a proxy for them:
        # at centre ~ 1e308 a finite scale still sends k*scale (and the bounds) to +/-inf,
        # which would silently flag NOTHING. One predicate, no gap between it and the op.
        lower = centre - k * scale
        upper = centre + k * scale
        if not (math.isfinite(lower) and math.isfinite(upper)):
            raise ValueError(
                f"Envelope bounds are not finite (centre={centre}, scale={scale}, k={k}); "
                f"the count rates are too large for a k-sigma band."
            )
        mask = (y < lower) | (y > upper)
    n_flagged = int(mask.sum())
    return OutlierFlags(
        flagged_mask=mask,
        n_flagged=n_flagged,
        fraction_flagged=n_flagged / y.size if y.size else 0.0,
        centre_cps=centre,
        scale_cps=scale,
        mad_cps=mad,
        scale_is_degenerate=scale_is_degenerate,
        k=k,
        lower_bound_cps=lower,
        upper_bound_cps=upper,
    )


def resolve_hampel_window(n_points: int) -> int:
    """The default moving-window width, in points, for a trace of this length.

        window = largest odd integer <= n/8, clamped to
                 [HAMPEL_WINDOW_MIN_PTS, HAMPEL_WINDOW_MAX_PTS]

    A single fixed window cannot serve this platform's traces: the committed
    real ones run from 49 points (a one-minute Brookhaven scan) to 3600 (an
    hour-long run), a 73x range, and dt varies 30x on top of that. So the
    default is resolved from the trace's own length; the user owns the control
    and can override it (invariant 3), and the resolved value is reported.

    THE THREE NUMBERS HAVE THREE DIFFERENT STANDINGS. Only the floor is derived,
    and this docstring is the normative home for all three:

    * FLOOR = 11 -- derived, three independent arguments landing on the same
      number. (a) It is the width Pearson et al.'s (2016) simulation study works
      at throughout, K = 5 in their 2K+1 form (p. 4); they compare K = 3 against
      it on p. 11, and print no window-length rule anywhere. (b) It must be ODD,
      or the median is an interpolated non-data value and the implosion theorem
      this layer's abstention rests on -- stated for a 2K+1 window, p. 3 -- has
      no K to refer to. (c) The uncorrected small-n MAD bias at n = 11 is 7.8% (Croux &
      Rousseeuw's b_n = n/(n - 0.8), 1992 Eq. 4 p. 414), below one 0.5 step of
      the k control it feeds (~14%). That last one is WHY no b_n correction
      ships: the floor keeps the biased regime out of reach instead of
      correcting for it, which would be an unbenchmarked deviation from the
      canonical filter.
    * CAP = 101 -- an EMPIRICAL house choice, not a canonical value. Pearson
      prints no upper-window guidance at all. Measured on the committed
      hour-long traces, the flagged fraction falls 0.39% -> 0.30% -> 0.243% at
      windows 51 -> 101 -> 201 and is flat beyond; there is no corner. 101 was
      chosen where the remaining gain (0.06 points) stopped paying for the
      doubled edge cost (3.1% -> 6.2% of the trace unjudged). Owner-ruled
      2026-08-17 as today's answer, with a carried finding open to replace it
      with something better founded -- most likely a user-owned cap, since the
      window is user-owned and this bound silently is not.
    * n/8 -- a stated coverage budget, not a measurement: at most an eighth of
      any trace goes unjudged at its two edges.

    Evidence limits, so the next reader can weigh the cap honestly: the
    real-trace half rests on eight traces from one solvent system on one
    instrument, and the clean-trace half assumes Gaussian scatter.
    """
    window = int(n_points) // 8
    if window % 2 == 0:                     # step DOWN to odd; a wider window
        window -= 1                         # costs edge coverage, which is the
    return max(HAMPEL_WINDOW_MIN_PTS,       # thing the budget above is spending
               min(HAMPEL_WINDOW_MAX_PTS, window))


def _flag_runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Start index and length of every maximal run of True in `mask`.

    The zero sentinels on both ends are what make a run that touches the first
    or last sample fall out of the same arithmetic as an interior one, with no
    special case to get wrong.
    """
    edges = np.diff(np.asarray(mask, dtype=np.int8), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return starts, ends - starts


def flag_outliers_windowed(
    trace: IntensityTrace,
    k: float = OUTLIER_K_DEFAULT,
    window_pts: Optional[int] = None,
) -> WindowedOutlierFlags:
    """Decompose a trace into a rolling baseline, impulse flags and abstentions.

    The moving-window Hampel filter of Pearson et al. (2016), shipped as a
    DETECTOR rather than a filter: over a window of w = 2K+1 samples centred on
    each point, take the median m_k and the scale
    S_k = 1.4826 * median_j |x_{k-j} - m_k| (their Eq. 4, p. 2), and flag the
    centre sample when |x_k - m_k| > k * S_k. Pearson's filter would then
    REPLACE the flagged sample with m_k (Eq. 3); we never do -- the trace goes
    out exactly as it came in, and the output is curves and flags.

    Why this exists beside `flag_outliers`. The global band judges every sample
    against ONE baseline, so it cannot tell a dust spike from a slow drift. The
    moving window judges each sample against its own neighbourhood, which
    localizes impulses -- and shows the baseline it used, so the user can see
    what was treated as signal.

    THE STRUCTURAL BLIND SPOT, which is a property of the method and not a bug.
    A moving window ABSORBS any feature longer than itself: the rolling median
    simply tracks it as baseline. Pearson names the boundary exactly -- "an
    impulse is a sequence of at most K values" (p. 3) -- so events shorter than
    about half the window are flaggable and longer ones are not. This is why the
    decomposition ships the rolling curve and not just the flags: a persistent
    offset (dust that sits still in a viscous medium, say) shows up as a STEP IN
    THE BASELINE with no flags at all, where a reader can see it. Widening the
    window moves the boundary out.

    ABSTENTION, two causes and one contract. Where this layer has no local
    verdict it says so; it never guesses, and it never floors a scale.

    * No full window fits -- the first and last K points. Edge windows are not
      shrunk: a shrinking window would manufacture the implosion-prone,
      small-n-biased regime that `resolve_hampel_window`'s floor exists to keep
      out of reach, and do it at every trace edge.
    * The window's MAD implodes. More than K of its w samples share a value, so
      S_k = 0 (Pearson's theorem, p. 3) -- a stalled, flat or heavily quantized
      stretch. `flag_outliers`' docstring records why nothing is ever patched
      over this, and that ruling governs here too: no floor, no substitute
      scale, ever.

    Either way the point is simply NOT JUDGED BY THIS LAYER -- the global
    envelope, the running average and the stationarity test still cover it --
    and the abstention is reported (a gap in the band, and the fractions on the
    result), never silent.

    Parameters
    ----------
    trace : IntensityTrace
    k : float
        Multiplier on the local robust sigma (default `OUTLIER_K_DEFAULT`,
        shared with `flag_outliers` -- one control drives both bands). At the
        floor window it is also inside the t = 3.0-6.5 band Pearson measured as
        optimal for K = 5 (p. 5).
    window_pts : int, optional
        Window width in points; must be odd and at least
        `HAMPEL_WINDOW_MIN_PTS`. None resolves it from the trace length via
        `resolve_hampel_window`, which documents what the bounds rest on.

    Returns
    -------
    WindowedOutlierFlags

    Raises
    ------
    ValueError
        If k <= 0 or is not finite; if window_pts is not an odd integer at or
        above the floor; or if a judged window's band bounds are not finite
        (k * S_k saturated). A window LONGER than the trace does not raise --
        that is total edge abstention, which is an outcome and is reported.
    """
    if not (math.isfinite(k) and k > 0):
        raise ValueError(f"k must be a positive finite number, got {k!r}.")

    y = np.asarray(trace.count_rates_cps, dtype=float)
    t = np.asarray(trace.times_s, dtype=float)
    n = int(y.size)

    was_resolved = window_pts is None
    if was_resolved:
        window_pts = resolve_hampel_window(n)
    if isinstance(window_pts, bool) or not isinstance(window_pts, (int, np.integer)):
        raise ValueError(
            f"window_pts must be an integer number of points, got {window_pts!r}."
        )
    window_pts = int(window_pts)
    if window_pts < HAMPEL_WINDOW_MIN_PTS:
        raise ValueError(
            f"window_pts must be at least {HAMPEL_WINDOW_MIN_PTS}, got {window_pts}; "
            f"below that the window's own MAD is biased by more than the k control "
            f"can resolve (see resolve_hampel_window)."
        )
    if window_pts % 2 == 0:
        raise ValueError(
            f"window_pts must be odd, got {window_pts}; an even window has no centre "
            f"sample to judge and no half-width K, which is the form the implosion "
            f"criterion this layer abstains on is stated in."
        )

    half = window_pts // 2
    centre = np.full(n, np.nan)
    scale = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    if n >= window_pts:
        valid[half:n - half] = True
        # One path, always chunked. The temporary that np.median materializes is
        # n*w floats and w is user-owned, so it must be bounded -- but gating the
        # chunking on a size predicate would be a second boundary to mis-size
        # (rule 12) AND would leave one of two code paths unexercised by ordinary
        # input. Chunking unconditionally makes the small case the large case
        # running once, and the chunk size provably cannot change the answer.
        view = sliding_window_view(y, window_pts)
        rows = max(1, _HAMPEL_CHUNK_BYTES // (window_pts * 8))
        for start in range(0, view.shape[0], rows):
            block = view[start:start + rows]
            block_centre = np.median(block, axis=-1)
            block_scale = MAD_TO_SIGMA * np.median(
                np.abs(block - block_centre[:, None]), axis=-1)
            centre[half + start:half + start + block_centre.size] = block_centre
            scale[half + start:half + start + block_scale.size] = block_scale

    # `scale > 0.0` is `flag_outliers`' own phrasing: it routes a NaN scale (a
    # window holding a non-finite sample) to abstention, which is the honest
    # destination for it, and leaves +inf to the bounds guard below. The three
    # masks come off one comparison, so they cannot disagree with each other.
    judged = valid & (scale > 0.0)
    implosion_abstained = valid & ~(scale > 0.0)
    edge_abstained = ~valid

    # Saturation to +/-inf here is the case the guard below EXISTS to catch, so
    # numpy's warning about it is noise on a handled path -- silenced locally,
    # over this expression only, never module-wide.
    with np.errstate(over='ignore', invalid='ignore'):
        lower = centre - k * scale
        upper = centre + k * scale
    if judged.any() and not (np.all(np.isfinite(lower[judged]))
                             and np.all(np.isfinite(upper[judged]))):
        raise ValueError(
            f"Local envelope bounds are not finite somewhere (k={k}, "
            f"window_pts={window_pts}); the count rates are too large for a "
            f"k-sigma band."
        )
    flagged = judged & ((y < lower) | (y > upper))

    # The NaNs are a rendering of `judged`, derived from it rather than tracked
    # alongside it, so a plot's gaps and the mask can never disagree.
    centre[~judged] = np.nan
    scale[~judged] = np.nan

    starts, lengths = _flag_runs(flagged)
    if starts.size:
        spans = t[starts + lengths - 1] - t[starts]
    else:
        spans = np.zeros(0, dtype=float)

    n_judged = int(judged.sum())
    n_flagged = int(flagged.sum())
    return WindowedOutlierFlags(
        times_s=t.copy(),
        rolling_centre_cps=centre,
        rolling_scale_cps=scale,
        flagged_mask=flagged,
        local_verdict_mask=judged,
        edge_abstained_mask=edge_abstained,
        implosion_abstained_mask=implosion_abstained,
        n_points=n,
        n_judged=n_judged,
        n_flagged=n_flagged,
        fraction_flagged=n_flagged / n_judged if n_judged else 0.0,
        fraction_abstained=(n - n_judged) / n if n else 0.0,
        fraction_abstained_edge=int(edge_abstained.sum()) / n if n else 0.0,
        fraction_abstained_implosion=int(implosion_abstained.sum()) / n if n else 0.0,
        n_events=int(starts.size),
        event_start_indices=starts,
        event_lengths_pts=lengths,
        event_spans_s=spans,
        longest_event_pts=int(lengths.max()) if lengths.size else 0,
        k=k,
        window_pts=window_pts,
        window_half_width_k=half,
        window_floor_pts=HAMPEL_WINDOW_MIN_PTS,
        window_description=f'{window_pts} points (K = {half})',
        window_was_resolved=was_resolved,
        mad_to_sigma=MAD_TO_SIGMA,
    )


def normalize_trace(
    trace: IntensityTrace,
    baseline_method: str = 'percentile',
    baseline_parameter: float = 25.0,
) -> NormalizedTraceResult:
    """Normalize a trace by its baseline so it fluctuates around 1.0.

    Useful for comparing traces from different samples or concentrations on the
    same axes, where absolute count-rate differences would otherwise dominate.

    Parameters
    ----------
    trace : IntensityTrace
    baseline_method, baseline_parameter
        Passed to identify_baseline().

    Returns
    -------
    NormalizedTraceResult

    Raises
    ------
    ValueError
        If the estimated baseline is zero (cannot normalize).
    """
    baseline = identify_baseline(trace, method=baseline_method,
                                 parameter=baseline_parameter).baseline_cps
    if not (math.isfinite(baseline) and baseline != 0):
        raise ValueError(
            f"Estimated baseline must be a non-zero finite number, got {baseline}; "
            f"cannot normalize the trace."
        )
    return NormalizedTraceResult(
        times_s=trace.times_s.copy(),
        normalized=trace.count_rates_cps / baseline,
        baseline_cps=baseline,
        baseline_method=baseline_method,
    )


def running_average(
    trace: IntensityTrace,
    window_s: Optional[float] = None,
    window_points: Optional[int] = None,
) -> RunningAverageResult:
    """Sliding-window mean and standard deviation of a trace.

    Exactly one of window_s or window_points must be supplied.

    A time-based window (window_s) correctly handles the irregular sampling
    intervals of Brookhaven count-rate exports (the spacing is only nominally
    1 s). A point-based window (window_points) is simpler but assumes roughly
    uniform sampling.

    Parameters
    ----------
    trace : IntensityTrace
    window_s : float, optional
        Full width of the centered time window, in seconds.
    window_points : int, optional
        Full width of the centered window, in number of points.

    Returns
    -------
    RunningAverageResult
        running_mean and running_std have the same length as the trace.
        Where a window contains a single point, the std is 0.

    Raises
    ------
    ValueError
        If neither or both of window_s / window_points are supplied.
    """
    if (window_s is None) == (window_points is None):
        raise ValueError(
            "Supply exactly one of window_s or window_points (not both, "
            "not neither)."
        )
    t = trace.times_s
    y = trace.count_rates_cps
    n = y.size
    means = np.empty(n, dtype=float)
    stds = np.empty(n, dtype=float)

    if window_s is not None:
        if not (math.isfinite(window_s) and window_s > 0):
            raise ValueError(
                f"window_s must be a positive finite number, got {window_s!r}.")
        half = window_s / 2.0
        # times_s is monotonically increasing, so use searchsorted for O(n log n)
        lo_idx = np.searchsorted(t, t - half, side='left')
        hi_idx = np.searchsorted(t, t + half, side='right')
        # Each window is a contiguous [lo, hi) slice, so its sum and sum-of-squares
        # come from prefix sums in O(1) — O(n) overall instead of the per-point
        # slice .mean()/.std() (which was O(n·w), quadratic on long traces). The
        # variance is computed on data shifted by the global mean (variance is
        # translation-invariant): this keeps the one-pass sum-of-squares formula
        # numerically stable for large count rates, matching np.std(ddof=1).
        shift = float(y.mean()) if n else 0.0
        yc = y - shift
        c1 = np.concatenate(([0.0], np.cumsum(yc)))            # c1[k] = sum(yc[:k])
        c2 = np.concatenate(([0.0], np.cumsum(yc * yc)))       # c2[k] = sum(yc[:k]^2)
        counts = (hi_idx - lo_idx).astype(float)               # window always ⊇ point i
        sw = c1[hi_idx] - c1[lo_idx]
        s2w = c2[hi_idx] - c2[lo_idx]
        means = shift + sw / counts
        with np.errstate(invalid='ignore', divide='ignore'):
            var = (s2w - sw * sw / counts) / (counts - 1.0)
        # ddof=1 is undefined for a single-point window (std=0 there, as before);
        # clamp tiny negative variances from floating-point cancellation.
        stds = np.where(counts > 1.0, np.sqrt(np.maximum(var, 0.0)), 0.0)
        desc = f"{window_s:g} s"
    else:
        if window_points <= 0:
            raise ValueError(
                f"window_points must be positive, got {window_points!r}."
            )
        # An interior window spans EXACTLY window_points samples. Using the
        # symmetric [i-half : i+half+1] made the width 2*half+1, so an even
        # request (e.g. 10) averaged one point too many (11). The half-open
        # [i-half : i-half+w] is w wide for any w (even widths are unavoidably
        # asymmetric by one sample); clamp each end at the trace boundary.
        w = window_points
        half = w // 2
        # The unclamped windows are the rows of a sliding view, each row reduced
        # with the slice's own .mean() / .std(ddof=1), in blocks of rows; the
        # clamped edge windows, and every window when w > n, take the per-slice
        # loop. Pinned to the per-slice values exactly by
        # test_running_average_point_window_matches_the_per_slice_reference.
        n_interior = max(0, n - w + 1)
        if n_interior > 0:
            rows = sliding_window_view(y, w)
            step = max(1, _RUNNING_AVERAGE_BLOCK_ELEMENTS // w)
            for start in range(0, n_interior, step):
                block = rows[start:start + step]
                out = slice(half + start, half + start + block.shape[0])
                means[out] = block.mean(axis=1)
                stds[out] = block.std(axis=1, ddof=1) if w > 1 else 0.0
            edges = (*range(half), *range(half + n_interior, n))
        else:
            edges = range(n)
        for i in edges:
            lo = max(0, i - half)
            hi = min(n, i - half + w)
            window = y[lo:hi]
            means[i] = window.mean()
            stds[i] = window.std(ddof=1) if window.size > 1 else 0.0
        desc = f"{window_points} points"

    return RunningAverageResult(times_s=t.copy(), running_mean=means,
                                running_std=stds, window_description=desc)


def block_variance(
    trace: IntensityTrace,
    n_block_sizes: int = 20,
    min_block_size: int = 1,
    max_block_size: Optional[int] = None,
    correlation_threshold: float = BLOCKVAR_THRESHOLD_DEFAULT,
) -> BlockVarianceResult:
    """Blocking analysis of a trace: standard error and Allan deviation vs block size.

    Partitions the trace into blocks of increasing size, computes the mean of
    each block, and estimates the standard error of the overall mean at each
    block size as std(block_means)/sqrt(n_blocks). For uncorrelated data this
    is flat; for positively correlated data (slow modes) it rises with block
    size and plateaus at the true standard error.

    THE ALLAN DEVIATION RIDES THE SAME PARTITION. Barnes et al. (1971) define
    the two-sample (Allan) variance as the mean squared difference of ADJACENT
    block means, halved -- their Eq. (11), p. 109:

        sigma_y^2(tau) = < (ybar_{k+1} - ybar_k)^2 / 2 >

    for T = tau, which they gloss as "no dead time between measurements". A
    uniformly sampled trace cut into contiguous blocks satisfies that condition
    exactly, and the block means computed here for the standard error ARE the
    ybar_k the definition needs. So both curves come from one partition and one
    set of block means; there is no second cutting of the trace to disagree with
    the first. The finite-m estimator (their abstract, p. 105) averages m
    differences, and m is reported per tau because p. 115 asks that it be stated.

    The deviation is FRACTIONAL -- block means divided by the trace mean before
    differencing. Barnes's y is a fractional deviation (his Eq. 7, p. 109), so
    normalizing this way is what makes his printed tau-slope mappings
    (Eqs. 28-29, p. 111, with Fig. 1) apply verbatim: white noise reads slope
    -1/2 on a log-log sigma_y(tau) plot, a drifting or flicker-limited trace
    flattens or turns up. An absolute sigma(tau) in cps would need every one of
    those mappings re-derived for no interpretive gain. The mean is returned
    alongside, so nothing is hidden and the absolute form is one multiply away.

    One caveat worth carrying to the reader, from Barnes p. 115: discrete
    spectral lines -- he names power-supply frequencies -- make sigma_y^2(tau) a
    rapidly varying function of tau, and at a tau that is an exact multiple of
    the modulation period the value "can be very optimistic". That is why this
    view is read beside the spectral ones rather than alone.

    Parameters
    ----------
    trace : IntensityTrace
    n_block_sizes : int
        Number of distinct block sizes to evaluate (log-spaced).
    min_block_size : int
        Smallest block size (default 1).
    max_block_size : int, optional
        Largest block size. Default is N // 4 so the largest block size still
        yields at least 4 blocks for a stable estimate.
    correlation_threshold : float
        If SE(largest)/SE(smallest) exceeds this, correlations_detected is True.
        This is a general heuristic, not a system-specific threshold; the full
        curve is returned so you can judge for yourself. Defaults to
        BLOCKVAR_THRESHOLD_DEFAULT, which the controller seeds from rather than
        restating.

    Returns
    -------
    BlockVarianceResult

    Raises
    ------
    ValueError
        If the trace has fewer than 8 points (too short for meaningful blocking).
    """
    y = trace.count_rates_cps
    n = y.size
    if n < 8:
        raise ValueError(
            f"Trace has only {n} points; blocking analysis needs at least 8."
        )
    if max_block_size is None:
        max_block_size = max(2, n // 4)
    max_block_size = min(max_block_size, n // 2)

    # Log-spaced, unique integer block sizes
    raw = np.unique(np.round(np.geomspace(
        max(1, min_block_size), max_block_size, n_block_sizes
    )).astype(int))
    raw = raw[raw >= 1]

    # The fractional normalizer for the Allan deviation. A non-positive mean
    # leaves sigma_y genuinely undefined (it is a deviation RELATIVE to the
    # mean), so it becomes NaN rather than a bare 0/0 -- the same call
    # fit_count_rate_histogram makes for the Fano factor.
    mean_cps = float(y.mean()) if n else float('nan')
    mean_is_usable = math.isfinite(mean_cps) and mean_cps > 0.0
    # dt from the time axis, as elsewhere in this module; a single-sample grid
    # has no spacing to read, so tau is left undefined rather than assumed.
    dt_s = (float(np.median(np.diff(trace.times_s)))
            if trace.times_s.size > 1 else float('nan'))

    block_sizes = []
    standard_errors = []
    n_blocks_list = []
    allan_list = []
    allan_pairs_list = []
    for b in raw:
        m = n // b
        if m < 2:
            continue
        trimmed = y[:m * b].reshape(m, b)
        block_means = trimmed.mean(axis=1)
        se = block_means.std(ddof=1) / math.sqrt(m)
        # Allan on the SAME block means: Barnes Eq. (11), finite-m form. With m
        # blocks there are m-1 adjacent differences, and m >= 2 here so there is
        # always at least one.
        if mean_is_usable:
            diffs = np.diff(block_means) / mean_cps
            allan_var = float(np.mean(diffs * diffs) / 2.0)
            allan_dev = (math.sqrt(allan_var)
                         if math.isfinite(allan_var) and allan_var >= 0.0
                         else float('nan'))
        else:
            allan_dev = float('nan')
        block_sizes.append(int(b))
        standard_errors.append(float(se))
        n_blocks_list.append(int(m))
        allan_list.append(allan_dev)
        allan_pairs_list.append(int(m - 1))

    block_sizes = np.array(block_sizes)
    standard_errors = np.array(standard_errors)
    n_blocks_arr = np.array(n_blocks_list)
    allan_deviation = np.array(allan_list, dtype=float)
    allan_n_pairs = np.array(allan_pairs_list, dtype=int)
    tau_s = block_sizes * dt_s

    # Robust ratio: compare the plateau (upper block sizes) to the small-block
    # region, using medians to damp the large estimate noise at big block sizes
    # (where there are few blocks). Taking the raw max here would produce false
    # positives on genuinely uncorrelated data, because the max over many noisy
    # points drifts upward by chance.
    n_se = standard_errors.size
    if n_se >= 4 and standard_errors[0] > 0:
        q = max(1, n_se // 4)
        se_small = float(np.median(standard_errors[:q]))
        se_large = float(np.median(standard_errors[-q:]))
        se_ratio = (se_large / se_small
                    if math.isfinite(se_small) and se_small > 0 else float('nan'))
    elif n_se >= 2 and math.isfinite(standard_errors[0]) and standard_errors[0] > 0:
        se_ratio = float(standard_errors[-1] / standard_errors[0])
    else:
        se_ratio = float('nan')

    return BlockVarianceResult(
        block_sizes=block_sizes,
        standard_errors=standard_errors,
        n_blocks=n_blocks_arr,
        se_ratio=se_ratio,
        correlation_threshold=correlation_threshold,
        correlations_detected=bool(se_ratio > correlation_threshold)
        if not math.isnan(se_ratio) else False,
        tau_s=tau_s,
        allan_deviation=allan_deviation,
        allan_n_pairs=allan_n_pairs,
        mean_cps=mean_cps,
        dt_s=dt_s,
    )


def _freedman_diaconis_bins(values: np.ndarray) -> int:
    """Number of histogram bins by the Freedman-Diaconis rule.

    Bin width = 2 * IQR / n^(1/3). Robust to outliers because it uses the
    interquartile range rather than the standard deviation.
    """
    n = values.size
    q75, q25 = np.percentile(values, [75, 25])
    iqr = q75 - q25
    if not (math.isfinite(iqr) and iqr > 0):
        return 0  # signals caller to fall back
    bin_width = 2.0 * iqr / (n ** (1.0 / 3.0))
    data_range = values.max() - values.min()
    if not (math.isfinite(bin_width) and bin_width > 0) or \
            not (math.isfinite(data_range) and data_range > 0):
        return 0
    return max(1, int(math.ceil(data_range / bin_width)))


def _sturges_bins(values: np.ndarray) -> int:
    """Number of histogram bins by Sturges' rule: ceil(log2(n) + 1)."""
    return max(1, int(math.ceil(math.log2(values.size) + 1)))


def fit_count_rate_histogram(
    trace: IntensityTrace,
    distribution: str = 'both',
    bin_method: str = 'auto',
    integration_time_s: Optional[float] = None,
) -> HistogramFitResult:
    """Build a count-rate histogram and fit Gaussian and/or Poisson models.

    The histogram is built in count-rate space (cps). The Poisson model is
    physically defined on integer photon counts, so rates are converted to
    counts using the integration time (counts = rate * T), the Poisson
    parameter lambda is the mean count, and the model is mapped back onto the
    rate axis for overlay. The Fano factor (variance/mean in count space) is
    reported as the primary shot-noise diagnostic.

    Parameters
    ----------
    trace : IntensityTrace
    distribution : str
        'gaussian', 'poisson', or 'both' (default).
    bin_method : str
        'fd' (Freedman-Diaconis), 'sturges', or 'auto' (default). 'auto' uses
        Freedman-Diaconis, falling back to Sturges if FD fails (e.g. zero IQR
        or very short traces).
    integration_time_s : float, optional
        Sampling integration time in seconds. If None, it is estimated as the
        median spacing between trace timestamps (with a flag set in the result).
        Required conceptually for the Poisson fit; the estimate is usually a
        good approximation for Brookhaven exports.

    Returns
    -------
    HistogramFitResult

    Raises
    ------
    ValueError
        If distribution or bin_method is unrecognized.
    """
    if distribution not in ('gaussian', 'poisson', 'both'):
        raise ValueError(
            f"distribution must be 'gaussian', 'poisson', or 'both', "
            f"got {distribution!r}."
        )
    if bin_method not in ('fd', 'sturges', 'auto'):
        raise ValueError(
            f"bin_method must be 'fd', 'sturges', or 'auto', got {bin_method!r}."
        )

    y = trace.count_rates_cps

    # --- integration time ---
    estimated = False
    if integration_time_s is None:
        if trace.times_s.size > 1:
            integration_time_s = float(np.median(np.diff(trace.times_s)))
        else:
            integration_time_s = 1.0
        estimated = True

    # --- binning ---
    if bin_method == 'fd':
        n_bins = _freedman_diaconis_bins(y)
        method_used = 'fd'
        if n_bins == 0:
            n_bins = _sturges_bins(y)
            method_used = 'sturges (fd failed)'
    elif bin_method == 'sturges':
        n_bins = _sturges_bins(y)
        method_used = 'sturges'
    else:  # auto
        n_bins = _freedman_diaconis_bins(y)
        method_used = 'fd'
        if n_bins == 0:
            n_bins = _sturges_bins(y)
            method_used = 'sturges (auto fallback)'

    counts, bin_edges = np.histogram(y, bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # --- count-space statistics ---
    counts_per_sample = y * integration_time_s
    mean_counts = float(counts_per_sample.mean())
    var_counts = float(counts_per_sample.var(ddof=1)) if y.size > 1 else 0.0
    fano = (var_counts / mean_counts
            if math.isfinite(mean_counts) and mean_counts != 0 else float('nan'))
    mean_rate = float(y.mean())
    std_rate = float(y.std(ddof=1)) if y.size > 1 else 0.0
    # Degenerate zero-mean trace: report cv = 0 rather than a bare nan from 0/0,
    # consistent with compute_trace_statistics. (Fano/shot-noise stay nan: a Poisson
    # statistic is genuinely undefined with no counts, unlike a relative spread.)
    cv = std_rate / mean_rate if mean_rate != 0 else 0.0
    shot_noise_cv = (1.0 / math.sqrt(mean_counts)
                     if math.isfinite(mean_counts) and mean_counts > 0 else float('nan'))

    result = HistogramFitResult(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        counts=counts,
        bin_method=method_used,
        n_bins=len(counts),
        integration_time_s=integration_time_s,
        integration_time_estimated=estimated,
        mean_counts=mean_counts,
        variance_counts=var_counts,
        fano_factor=fano,
        cv=cv,
        shot_noise_cv=shot_noise_cv,
    )

    # Helper: reduced chi-squared using bins with enough counts
    def _reduced_chi2(observed, expected, n_params):
        # chi-squared validity needs >=5 counts per bin; the Pearson chi-squared
        # also divides by the expected count, so bins with expected == 0 (e.g. a
        # Gaussian tail far from the data) must be excluded or they raise a
        # divide-by-zero RuntimeWarning and inject inf into the sum.
        mask = (observed >= 5) & (expected > 0)
        if mask.sum() <= n_params:
            return None
        obs = observed[mask].astype(float)
        exp = expected[mask].astype(float)
        chi2 = np.sum((obs - exp) ** 2 / exp)
        dof = mask.sum() - n_params
        return float(chi2 / dof) if dof > 0 else None

    # --- Gaussian fit ---
    if distribution in ('gaussian', 'both'):
        def gaussian(x, amp, mu, sigma):
            return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        try:
            p0 = [counts.max(), mean_rate,
                  std_rate if math.isfinite(std_rate) and std_rate > 0 else 1.0]
            popt, _ = optimize.curve_fit(gaussian, bin_centers, counts, p0=p0,
                                         maxfev=10000)
            fit_curve = gaussian(bin_centers, *popt)
            result.gaussian_params = {
                'amp': float(popt[0]),
                'mu_cps': float(popt[1]),
                'sigma_cps': float(abs(popt[2])),
            }
            result.gaussian_curve = fit_curve
            result.gaussian_chi2_reduced = _reduced_chi2(counts, fit_curve, 3)
        except (RuntimeError, ValueError, TypeError):
            # TypeError: curve_fit refuses outright (not a convergence failure)
            # when there are fewer data points than the 3 Gaussian parameters
            # to fit -- a degenerate-trace case, not an internal error.
            result.gaussian_params = None  # fit failed; leave curve None

    # --- Poisson fit (in count space, mapped back to rate axis) ---
    if distribution in ('poisson', 'both'):
        lam = mean_counts
        result.poisson_lambda_counts = lam
        # Counts are integer-valued (a rate is count / integration_time), so each
        # rate bin maps to a contiguous range of integer counts [c_lo, c_hi]. The
        # probability mass in a bin is the Poisson PMF summed over that integer
        # range; the overlay is that mass scaled to expected counts-per-bin
        # (N_total * P(bin)). Bins narrower than one count get zero mass (c_hi < c_lo).
        try:
            total = counts.sum()
            # Each rate bin maps to an inclusive integer-count range [c_lo, c_hi]; the
            # Poisson mass over it is cdf(c_hi) - cdf(c_lo - 1). Vectorized over all
            # bins — one pair of cdf() calls instead of a per-bin arange + pmf sum,
            # which looped over every integer count in the bin (a big win for bright
            # samples spanning 10^5-10^6 counts). Bins narrower than one count
            # (c_hi < c_lo) get zero mass; np.where evaluates both cdf branches, but
            # cdf of a below-range (or negative) count is 0, so that is harmless.
            c_lo = np.ceil(bin_edges[:-1] * integration_time_s).astype(np.int64)
            c_hi = np.floor(bin_edges[1:] * integration_time_s).astype(np.int64)
            prob = np.where(
                c_hi >= c_lo,
                stats.poisson.cdf(c_hi, lam) - stats.poisson.cdf(c_lo - 1, lam),
                0.0)
            poisson_curve = total * prob
            result.poisson_curve = poisson_curve
            result.poisson_chi2_reduced = _reduced_chi2(counts, poisson_curve, 1)
        except (ValueError, FloatingPointError):
            result.poisson_curve = None

    return result


def test_stationarity_adf(
    trace: IntensityTrace,
    significance: float = 0.05,
) -> StationarityResult:
    """Augmented Dickey-Fuller test for stationarity of a trace.

    The null hypothesis of the ADF test is that the series has a unit root
    (i.e. is non-stationary). A p-value below `significance` lets you reject
    that null and conclude the trace is stationary. A non-stationary trace
    indicates drift or slow-mode behavior that violates the assumptions of
    standard DLS analysis.

    Parameters
    ----------
    trace : IntensityTrace
    significance : float
        Significance level for the stationarity decision (default 0.05).

    Returns
    -------
    StationarityResult

    Raises
    ------
    ImportError
        If statsmodels is not installed. Install with: pip install statsmodels
    ValueError
        If the trace has fewer than `ADF_MIN_POINTS`, or if the count rates do
        not vary at all -- both cases statsmodels itself refuses too, with its
        own bare message; this raises first with ours.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError as exc:
        raise ImportError(
            "test_stationarity_adf requires statsmodels, which is not "
            "installed. Install it with:  pip install statsmodels"
        ) from exc

    y = trace.count_rates_cps
    n = y.size
    if n < ADF_MIN_POINTS:
        raise ValueError(
            f"Trace has only {n} points; the Augmented Dickey-Fuller test "
            f"needs at least {ADF_MIN_POINTS} (see ADF_MIN_POINTS)."
        )
    if np.ptp(y) == 0:
        raise ValueError(
            "The count rates do not vary, so there is no series for the "
            "Augmented Dickey-Fuller test to examine."
        )
    result = adfuller(y, autolag='AIC')
    adf_stat, p_value, _, _, crit_values, _ = result
    return StationarityResult(
        adf_statistic=float(adf_stat),
        p_value=float(p_value),
        critical_values={k: float(v) for k, v in crit_values.items()},
        significance=significance,
        is_stationary=bool(p_value < significance),
    )


def preload_stationarity_backend() -> bool:
    """Import the statsmodels backend eagerly, on the CALLER'S thread.

    ``test_stationarity_adf`` imports statsmodels lazily so the dependency stays
    optional — but in the GUI that first import would otherwise run on the
    background worker thread the diagnostic is dispatched to, and a first-time
    statsmodels/scipy.signal import cascade on a non-main thread while PySide6's
    shiboken import hook is live can corrupt the heap (0xc0000374 — reproduced
    and bisected in the test harness). The GUI calls this once on
    the main thread at startup; the worker's later import is then a plain
    ``sys.modules`` lookup and no import machinery runs off the main thread.

    Returns True if the backend imported, False otherwise. Never raises — and
    deliberately catches MORE than ImportError: a broken or ABI-mismatched
    statsmodels/scipy install can raise ValueError/RuntimeError from its
    compiled-extension checks, and this preload runs at GUI startup where an
    escape would abort the whole app. The worker path it replaces contained
    any failure (gui/worker.py catches broadly), so a degraded-but-running
    diagnostic must remain the worst case here too; ``test_stationarity_adf``
    keeps the informative per-use error.
    """
    try:
        import statsmodels.tsa.stattools  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 — import probe, docstring
        return False


# ===========================================================================
# Frequency-domain diagnostics
# ===========================================================================

def resolve_running_average_window(n_points: int) -> int:
    """The default running-average width, in points, for a trace of this length.

        window = n / 20, floored at 3

    A house convention, and this docstring is its normative home: a twentieth of
    the record is short enough to follow real structure and long enough to damp
    single-sample scatter, and the floor of 3 keeps the window wide enough for a
    within-window standard deviation to mean anything. `running_average` itself
    still demands an explicit width -- resolving a default is the CALLER's
    decision, and this is simply the one place that decision is written down, so
    the Traces tab's overlay and its export cannot drift apart about what "auto"
    means.
    """
    return max(3, int(n_points) // 20)


def resolve_psd_segment(n_points: int) -> int:
    """The default Welch segment length, in points, for a trace of this length.

        segment = largest power of two <= n/8, clamped to
                  [PSD_SEGMENT_MIN_PTS, PSD_SEGMENT_MAX_PTS] and to n

    THIS IS A HOUSE CHOICE, AND THIS DOCSTRING IS ITS NORMATIVE HOME. Welch
    (1967) prints NO rule for the segment length. He treats L purely as "a scale
    parameter": changing it stretches or shrinks the spectral window (p. 71), so
    it trades frequency resolution against how much averaging the estimate gets,
    and where a given trace should sit on that trade is not something he decides.
    Do not read the number below as canonical -- the check that matters is
    whether the resolved value suits the trace, which is why the user owns the
    control (invariant 3) and the resolved value is reported and exported.

    What each part rests on:

    * n/8 -- the same coverage budget the moving-window Hampel default spends
      (`resolve_hampel_window`), chosen for consistency rather than measured
      here. At 50% overlap it yields K = 2n/L - 1 = 15 segments, so E.D.F. is
      about 25 (`welch_edf`) and the relative scatter of the estimate is about
      sqrt(2/25) ~ 28%. That is enough averaging to read a spectrum by eye while
      still resolving 1/8 of the record.
    * POWER OF TWO -- an FFT-efficiency convention, and it makes the resolution
      ladder predictable as a user steps the control. Nothing statistical rests
      on it; Welch's estimator and its E.D.F. are indifferent to the value.
    * FLOOR = 16 -- below this a segment carries too few frequency bins for the
      curve to mean anything, and the window's own shape dominates the result.
    * CAP = 4096 -- keeps one segment's transform cheap on an hour-long trace and
      keeps at least a few segments to average. Like the Hampel cap this is an
      empirical house bound, not a canonical value.

    The floor is also why very short traces are refused outright rather than
    served a one-segment "average" (see `welch_psd`).
    """
    n = int(n_points)
    target = max(1, n // 8)
    segment = 1 << (int(target).bit_length() - 1)       # largest 2^m <= target
    segment = max(PSD_SEGMENT_MIN_PTS, min(PSD_SEGMENT_MAX_PTS, segment))
    return min(segment, n)


def _psd_window(name: str, nperseg: int) -> np.ndarray:
    """The data window W(j), j = 0 .. nperseg-1, for one of `PSD_WINDOWS`.

    Welch's two are built from his printed forms (p. 71) rather than taken from
    scipy, so that his E.D.F. figures apply to exactly the arrays used here:

        W1(j) = 1 - ((j - (L-1)/2) / ((L+1)/2))^2        the 1 - t^2 window
        W2(j) = 1 - |(j - (L-1)/2) / ((L+1)/2)|          the 1 - |t| window

    W2 is the TRIANGULAR data window. Welch calls it Parzen's because the
    SPECTRAL window it produces is Parzen's, but it is not scipy's 'parzen',
    which is a different (B-spline) function -- hence the name used here, and
    hence building it rather than requesting it by that name.

    'hann' comes from scipy in its periodic (``sym=False``) form, the correct one
    for spectral estimation.
    """
    if name == 'hann':
        return np.asarray(signal.get_window('hann', nperseg, fftbins=True),
                          dtype=float)
    j = np.arange(nperseg, dtype=float)
    t = (j - (nperseg - 1) / 2.0) / ((nperseg + 1) / 2.0)
    if name == 'welch-w1':
        return 1.0 - t * t
    if name == 'welch-w2':
        return 1.0 - np.abs(t)
    raise ValueError(
        f"window must be one of {PSD_WINDOWS}, got {name!r}."
    )


def welch_edf(window: np.ndarray, hop: int, n_segments: int) -> float:
    """Welch's equivalent degrees of freedom for an averaged periodogram.

    Welch (1967) derives, for a Gaussian process whose spectrum is flat over the
    estimator's passband (p. 71-72):

        rho(j)   = [sum_k W(k) W(k + jD)]^2 / [sum_k W^2(k)]^2
        Var{Phat} = (P^2 / K) * {1 + 2 * sum_{j=1}^{K-1} ((K-j)/K) rho(j)}

    and the equivalent degrees of freedom of the approximating chi-square
    distribution follow as E.D.F. = 2 * E^2{Phat} / Var{Phat}, i.e.

        E.D.F. = 2K / {1 + 2 * sum_{j=1}^{K-1} ((K-j)/K) rho(j)}

    Computing rho(j) from the ACTUAL window array and hop, rather than quoting
    one of Welch's worked numbers, is what lets this ship a conventional Hann
    window at 50% overlap without extrapolating a figure he printed for a
    different window. His three printed cases fall out of the same formula and
    are pinned as tests:

    * D >= L (no overlap): rho(j) = 0 for all j, so E.D.F. = 2K exactly (p. 72).
    * D = L/2 with W1: rho(1) ~ 1/9, giving E.D.F. ~ 18K/11 ~ 2.8 N df (p. 72).
    * D = L/2 with W2: rho(1) = 1/16 (p. 72).

    Mind the difference in standing between those last two, established here by
    evaluating the integrals rather than by trusting the page. W2's 1/16 is
    EXACT: for the triangular window the continuous limit of rho(1) is
    (1/6)^2 / (2/3)^2 = 1/16. W1's is not -- the continuous limit there is
    (11/30)^2 / (16/15)^2 = 121/1024 = 0.1182, and Welch's "~1/9" (0.1111) is his
    own rounding of it, low by 6%. He writes it as an approximation and uses it
    to get the round factor 1 + 2/9 = 11/9, so nothing on his page is wrong; but
    a test that pinned W1 to 1/9 tightly would be pinning the rounding, not the
    window. The exact values are what this function returns.

    An independent check the sources do not supply: for a Hann window at 50%
    overlap this formula gives rho(1) = 1/36 exactly, hence E.D.F. = 36K/19, the
    standard textbook result for Welch-Hann. That the general formula reproduces
    a known value for a window Welch never analysed is the evidence that the
    generalization is sound and not a re-derivation of his special cases.

    Reported for the interior frequencies. Welch notes (p. 72) that the variance
    doubles and the E.D.F. halves at f = 0 and at the Nyquist frequency; callers
    that care about those two bins must halve this themselves.

    Parameters
    ----------
    window : np.ndarray
        The data window W(j) actually applied to each segment.
    hop : int
        D, the offset between successive segment starts (nperseg - noverlap).
    n_segments : int
        K, the number of segments averaged.

    Returns
    -------
    float
        The equivalent degrees of freedom. NaN if the window has no energy.
    """
    w = np.asarray(window, dtype=float)
    length = w.size
    k_segments = int(n_segments)
    denom = float(np.dot(w, w))
    if not (math.isfinite(denom) and denom > 0.0) or k_segments < 1:
        return float('nan')
    denom = denom * denom
    correlation_sum = 0.0
    for j in range(1, k_segments):
        shift = j * int(hop)
        if shift >= length:
            break                       # rho(j) = 0 once the segments disjoin
        overlap_product = float(np.dot(w[shift:], w[:length - shift]))
        rho = (overlap_product * overlap_product) / denom
        correlation_sum += ((k_segments - j) / k_segments) * rho
    return 2.0 * k_segments / (1.0 + 2.0 * correlation_sum)


def _trace_sample_spacing_s(trace: IntensityTrace) -> float:
    """The trace's median sample spacing dt in s, for a function that builds a frequency
    axis from it.

    Strict: raises ValueError when the time axis has fewer than two points or the
    median spacing is not positive and finite, since no frequency axis exists then."""
    if trace.times_s.size < 2:
        raise ValueError(
            "The trace has no readable sample spacing (fewer than two time "
            "points), so no frequency axis can be built."
        )
    dt_s = float(np.median(np.diff(np.asarray(trace.times_s, dtype=float))))
    if not (math.isfinite(dt_s) and dt_s > 0.0):
        raise ValueError(
            f"Sample spacing resolves to {dt_s!r}; a frequency axis needs a "
            f"positive, finite dt."
        )
    return dt_s


def welch_psd(
    trace: IntensityTrace,
    *,
    nperseg: Optional[int] = None,
    overlap: float = 0.5,
    window: str = 'hann',
    confidence_level: float = 0.95,
) -> WelchPSDResult:
    """Welch-averaged power spectral density of a count-rate trace.

    Implements Welch's (1967) averaged modified periodogram: cut the record into
    segments of `nperseg` points overlapping by `overlap`, window and transform
    each, and average the resulting periodograms. The averaging is the whole
    point -- under Welch's conditions (a Gaussian process, a spectrum flat across
    the passband; p. 71) a single modified periodogram of ANY record length has
    variance P^2, 100% relative scatter (the variance doubled at zero frequency and
    Nyquist, p. 72), so an unaveraged spectrum cannot be read.

    THE DEFAULT OVERLAP IS WELCH'S OWN RECOMMENDATION for our situation. He notes
    (p. 72) that non-overlapping segments give the most variance reduction per
    unit of computation when N can be made arbitrarily large, but that when the
    record is fixed -- which a measured trace always is -- "a reasonable
    procedure is to overlap the segments by one half their length". The default
    window is Hann, the modern convention; Welch's own W1 and W2 are offered too.

    SHOT-NOISE FLOOR. The returned `shot_floor_psd` is where an ideal Poisson
    detector would sit, derived in two steps from the same count-space reasoning
    the Fano factor uses. Photon counts in a bin of width dt have variance equal
    to their mean, lambda = rbar*dt; the count RATE r = n/dt therefore has
    variance rbar/dt. A white sequence of variance V sampled at fs = 1/dt has
    one-sided spectral density 2V/fs = 2V*dt, so

        S_shot = 2 * (rbar/dt) * dt = 2 * rbar        [cps^2/Hz]

    flat at twice the mean count rate, independent of dt. This assumes
    dead-time-free Poisson detection; a detector with appreciable dead time is
    sub-Poissonian and sits below it. Broadband power above the floor is the
    excess the Fano factor totals, resolved by frequency; a narrow peak above it
    is a candidate periodicity whose SIGNIFICANCE comes from
    `periodogram_false_alarm`, never from this curve.

    Each segment has its mean removed before windowing (scipy's
    ``detrend='constant'``). Without it the mean would put a spike at f = 0 large
    enough to swamp the display; with it, the lowest bins understate genuine slow
    drift, which is the stationarity test's business rather than this view's.

    Parameters
    ----------
    trace : IntensityTrace
    nperseg : int, optional
        Segment length in points. Default resolves from the trace length via
        `resolve_psd_segment` -- a house rule; Welch prints none.
    overlap : float
        Fraction of a segment shared with the next, in [0, 1). Default 0.5.
    window : str
        One of `PSD_WINDOWS`. Default 'hann'.
    confidence_level : float
        Central probability of the reported chi-square band, in (0, 1).

    Returns
    -------
    WelchPSDResult

    Raises
    ------
    ValueError
        If the trace is shorter than one segment at the floor width; if the time
        axis has no readable spacing; or if any parameter is out of range.

    Notes
    -----
    The count rates are not checked for finiteness here: `IntensityTrace` already
    refuses non-finite values at construction, so a second check would be
    unreachable code claiming to guard something it can never see. That guarantee
    is pinned by a test on the data model rather than restated here.
    """
    y = np.asarray(trace.count_rates_cps, dtype=float)
    n = y.size
    if n < PSD_SEGMENT_MIN_PTS:
        raise ValueError(
            f"Trace has only {n} points; a Welch spectrum needs at least "
            f"{PSD_SEGMENT_MIN_PTS} (one segment at the floor width). Below "
            f"that the window's own shape dominates the result."
        )
    dt_s = _trace_sample_spacing_s(trace)
    if not (math.isfinite(overlap) and 0.0 <= overlap < 1.0):
        raise ValueError(
            f"overlap must be a fraction in [0, 1), got {overlap!r}."
        )
    if not (math.isfinite(confidence_level) and 0.0 < confidence_level < 1.0):
        raise ValueError(
            f"confidence_level must lie in (0, 1), got {confidence_level!r}."
        )
    if window not in PSD_WINDOWS:
        raise ValueError(
            f"window must be one of {PSD_WINDOWS}, got {window!r}."
        )

    segment_was_resolved = nperseg is None
    if segment_was_resolved:
        nperseg = resolve_psd_segment(n)
    else:
        if isinstance(nperseg, bool) or not isinstance(nperseg, (int, np.integer)):
            raise ValueError(
                f"nperseg must be an integer number of points, got {nperseg!r}."
            )
        nperseg = int(nperseg)
        if nperseg < PSD_SEGMENT_MIN_PTS:
            raise ValueError(
                f"nperseg must be at least {PSD_SEGMENT_MIN_PTS}, got "
                f"{nperseg}; below that the window's own shape dominates the "
                f"estimate (see resolve_psd_segment)."
            )
        if nperseg > n:
            raise ValueError(
                f"nperseg is {nperseg} but the trace has only {n} points; a "
                f"segment cannot be longer than the record."
            )

    win = _psd_window(window, nperseg)
    noverlap = int(round(overlap * nperseg))
    noverlap = min(noverlap, nperseg - 1)          # a hop of zero never advances
    hop = nperseg - noverlap
    n_segments = 1 + (n - nperseg) // hop

    frequencies_hz, psd = signal.welch(
        y, fs=1.0 / dt_s, window=win, nperseg=nperseg, noverlap=noverlap,
        detrend='constant', return_onesided=True, scaling='density',
    )

    edf = welch_edf(win, hop, n_segments)
    if math.isfinite(edf) and edf > 0.0:
        alpha = 1.0 - confidence_level
        # The chi-square band for a variance-like estimate with `edf` degrees of
        # freedom. Welch names the approximating chi-square distribution and
        # gives its degrees of freedom; the interval built from it is standard
        # and is ours, not his.
        ci_lower_factor = float(edf / stats.chi2.ppf(1.0 - alpha / 2.0, edf))
        ci_upper_factor = float(edf / stats.chi2.ppf(alpha / 2.0, edf))
    else:
        ci_lower_factor = float('nan')
        ci_upper_factor = float('nan')

    mean_cps = float(y.mean())
    shot_floor_psd = (2.0 * mean_cps
                      if math.isfinite(mean_cps) and mean_cps > 0.0
                      else float('nan'))

    return WelchPSDResult(
        frequencies_hz=np.asarray(frequencies_hz, dtype=float),
        psd=np.asarray(psd, dtype=float),
        edf=float(edf),
        ci_lower_factor=ci_lower_factor,
        ci_upper_factor=ci_upper_factor,
        confidence_level=float(confidence_level),
        shot_floor_psd=shot_floor_psd,
        mean_cps=mean_cps,
        nperseg=int(nperseg),
        noverlap=int(noverlap),
        n_segments=int(n_segments),
        window=str(window),
        dt_s=dt_s,
        segment_was_resolved=bool(segment_was_resolved),
    )


def periodogram_false_alarm(
    trace: IntensityTrace,
    *,
    false_alarm_probability: float = FALSE_ALARM_PROBABILITY_DEFAULT,
    correlations_detected: bool = False,
) -> PeriodogramResult:
    """Classical periodogram of a trace, with Scargle's peak false-alarm test.

    Answers one question: could the strongest peak in this spectrum have arisen
    from noise alone? Scargle (1982) sets it up in three printed steps.

    First the CLASSICAL PERIODOGRAM, his Eq. (4), p. 836, for evenly spaced data:

        P_X(omega) = (1/N0) * |sum_j X_j exp(-i j omega)|^2

    evaluated at N = N0/2 evenly spaced frequencies (his Eq. 5, p. 836). Then the
    NULL: for pure Gaussian noise the power at a given frequency is
    exponentially distributed, p(z) dz = exp(-z) dz (Eqs. 12-13, p. 839), where
    -- and this is the detail the whole test turns on -- "the noise variance will
    be taken to be unity, i.e. P will be measured in units of sigma_0^2". Then
    the FALSE-ALARM PROBABILITY of the largest of N independent powers, his
    Eq. (14), p. 839:

        Pr{Z > z} = 1 - [1 - exp(-z)]^N,     Z = max_n P(omega_n)

    which is the statistical penalty for having inspected many frequencies at
    once rather than one nominated in advance.

    WHY THIS IS NOT RUN ON THE WELCH ESTIMATE. The exponential null belongs to
    the single, unaveraged periodogram normalized by sigma_0^2. Welch's averaged
    estimate is chi-square with its E.D.F. degrees of freedom instead, so drawing
    this line against that curve would be testing the wrong null. The two views
    therefore carry two separate honest statements: `welch_psd` reports its
    E.D.F., and the significance verdict comes from here.

    WHAT THE NULL ASSUMES, AND WHEN IT FAILS. Scargle's model (his Eq. 1, p. 836)
    is a signal plus errors that are independent between samples and normally
    distributed with constant variance -- white AND Gaussian. A drifting or
    otherwise correlated trace violates the whiteness half, and the violation is
    not benign: correlated noise piles power into the lowest frequencies, where
    it can clear a line computed for a white null and read as a confident
    periodicity that is not there. So the verdict is reported together with
    `null_assumption_violated`, which the caller sets from the blocking
    analysis's correlation verdict. It is a flag, never a suppression: the number
    still carries information, and this platform flags rather than filters.

    The Gaussian half is an approximation for photon counts, good at the count
    rates DLS runs at (thousands per sample) and progressively worse as the
    counts per sample fall towards single digits.

    Two bins are excluded from the returned spectrum and the peak search. Zero
    frequency is identically zero once the mean is removed and carries no
    information. For even N0 the Nyquist bin is real-valued rather than complex,
    making it chi-square with ONE degree of freedom where the exponential null
    assumes two -- so it is not a fair member of the maximum. `n_independent`
    stays at Scargle's printed N0/2 regardless, which is the conservative
    direction: it counts at least as many trials as were actually searched.

    THE POWER RATIO IS FORMED BY SCALING, NOT BY DIVIDING AT THE END. Because z is
    a power divided by a variance, both computed from the same data, the two can
    be separately unrepresentable while their ratio is perfectly ordinary: at
    count rates near 1e-300 the variance underflows to zero and at 1e+300 it
    overflows to infinity, and in both cases a guard on the variance would refuse
    a trace whose z is finite and correct. So the centred data is divided by its
    own peak magnitude FIRST. The ratio is invariant under that scaling, no
    intermediate can leave float range, and the only input still refused is one
    that genuinely does not vary -- where the refusal is true as stated. (The
    reported `variance_cps2` is reconstructed in the original units afterwards
    and may legitimately be infinite; it is provenance, not something z depends
    on.)

    Parameters
    ----------
    trace : IntensityTrace
    false_alarm_probability : float
        The false-alarm probability the reported `significance_level` is drawn
        for, in (0, 1). Default 1%, a reporting convention -- Scargle prints the
        formula and no preferred level.
    correlations_detected : bool
        The blocking analysis's verdict for this trace, passed in so this
        function stays pure and single-purpose. Recorded as
        `null_assumption_violated`, and when True as the `null_not_white` note.

    Returns
    -------
    PeriodogramResult

    Raises
    ------
    ValueError
        If the trace is too short to have any non-trivial frequency; if the time
        axis has no readable spacing; if the count rates do not vary at all; or if
        the requested probability is out of range.

    Notes
    -----
    The count rates are not checked for finiteness here: `IntensityTrace` already
    refuses non-finite values at construction, so a second check would be
    unreachable code claiming to guard something it can never see. That guarantee
    is pinned by a test on the data model rather than restated here.
    """
    y = np.asarray(trace.count_rates_cps, dtype=float)
    n0 = y.size
    if n0 < 4:
        raise ValueError(
            f"Trace has only {n0} points; a periodogram needs at least 4 to "
            f"have any frequency below Nyquist to test."
        )
    dt_s = _trace_sample_spacing_s(trace)
    if not (math.isfinite(false_alarm_probability)
            and 0.0 < false_alarm_probability < 1.0):
        raise ValueError(
            f"false_alarm_probability must lie in (0, 1), got "
            f"{false_alarm_probability!r}."
        )

    centred = y - y.mean()
    # Scale first, divide never -- see the docstring. The peak magnitude is the
    # natural scale: it makes every sample land in [-1, 1], so neither the
    # variance nor the squared transform can leave float range.
    peak = float(np.max(np.abs(centred)))
    if not (math.isfinite(peak) and peak > 0.0):
        raise ValueError(
            "The count rates do not vary, so there is no scale for Scargle's "
            "null to measure power in units of."
        )
    scaled = centred / peak
    scaled_variance = float(np.var(scaled, ddof=1))
    # Reported in the original units for provenance; may be infinite for absurdly
    # large count rates, which does not affect the power ratio above.
    with np.errstate(over='ignore'):
        variance_cps2 = scaled_variance * peak * peak

    # Scargle Eq. (4) for even spacing. numpy's rfft computes exactly the sum in
    # that equation at omega_n = 2*pi*n/N0, so the 1/N0 prefactor is all that is
    # needed; z is then that power in units of sigma_0^2 (his Eqs. 12-13).
    spectrum = np.fft.rfft(scaled)
    power_all = (np.abs(spectrum) ** 2) / (n0 * scaled_variance)
    freq_all = np.fft.rfftfreq(n0, d=dt_s)
    # Drop bin 0 (identically zero after centring) and, when present, the
    # real-valued Nyquist bin -- see the docstring.
    last = power_all.size - 1 if n0 % 2 == 0 else power_all.size
    power = power_all[1:last]
    frequencies_hz = freq_all[1:last]
    if power.size == 0:
        raise ValueError(
            f"Trace of {n0} points leaves no testable frequency once the zero "
            f"and Nyquist bins are excluded."
        )

    n_independent = max(1, n0 // 2)          # Scargle Eq. (5), p. 836
    peak_index = int(np.argmax(power))
    peak_power = float(power[peak_index])
    peak_frequency_hz = float(frequencies_hz[peak_index])

    # Eq. (14). Written through log1p/expm1 so a strong peak, where exp(-z) is
    # tiny and the naive form would round 1 - (1 - eps)^N straight to zero,
    # still yields the correct ~N*eps.
    with np.errstate(over='ignore', invalid='ignore'):
        survival = math.exp(-peak_power) if peak_power < 700.0 else 0.0
    if survival <= 0.0:
        false_alarm = 0.0
    elif survival >= 1.0:
        false_alarm = 1.0
    else:
        false_alarm = -math.expm1(n_independent * math.log1p(-survival))

    # Invert Eq. (14) for the level a peak must clear:
    #     z_0 = -ln(1 - (1 - p)^(1/N))
    tail = -math.expm1(math.log1p(-false_alarm_probability) / n_independent)
    significance_level = (-math.log(tail)
                          if 0.0 < tail < 1.0 else float('inf'))

    return PeriodogramResult(
        frequencies_hz=np.asarray(frequencies_hz, dtype=float),
        power=np.asarray(power, dtype=float),
        n_independent=int(n_independent),
        peak_frequency_hz=peak_frequency_hz,
        peak_power=peak_power,
        false_alarm_probability=float(false_alarm),
        significance_level=float(significance_level),
        requested_false_alarm=float(false_alarm_probability),
        peak_is_significant=bool(peak_power > significance_level),
        null_assumption_violated=bool(correlations_detected),
        variance_cps2=variance_cps2,
        dt_s=dt_s,
        notes=((Note('null_not_white', ('false_alarm_probability', 'significance_level',
                                        'peak_is_significant')),)
               if correlations_detected else ()),
    )
