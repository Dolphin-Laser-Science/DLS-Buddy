"""
analysis/trace_analysis.py
==========================

Intensity-trace signal-analysis diagnostics. Given a count-rate trace
(IntensityTrace), these assess data quality: summary statistics, robust baseline
estimation, shot-noise outlier flagging, baseline normalization, sliding-window
running averages, blocking (standard-error-vs-block-size) analysis, count-rate
histograms with Gaussian/Poisson fits and the Fano factor, and a formal Augmented
Dickey-Fuller stationarity test.

(Promoted out of analysis/utilities.py in Session 59: this cluster has no code
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
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import optimize, stats

from core.data_models import IntensityTrace


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
    """Points falling outside mean +/- k*sqrt(mean)."""
    flagged_mask: np.ndarray       # bool array, True where flagged
    n_flagged: int
    fraction_flagged: float
    mean_cps: float
    sqrt_mean: float
    k: float
    lower_bound_cps: float
    upper_bound_cps: float


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
    """Blocking analysis: standard error of the mean vs block size.

    For an uncorrelated (white-noise) trace the standard error is independent
    of block size. A standard error that rises with block size and plateaus is
    the signature of positive correlations -- e.g. slow-mode fluctuations. The
    plateau value is the corrected standard error of the mean.
    """
    block_sizes: np.ndarray        # number of points per block
    standard_errors: np.ndarray    # SE of the mean estimated at each block size
    n_blocks: np.ndarray           # how many blocks at each size
    se_ratio: float                # SE(largest block) / SE(smallest block)
    correlation_threshold: float
    correlations_detected: bool    # se_ratio > correlation_threshold


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
        if sd == 0:
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
    cv = std / mean if mean != 0 else float('nan')
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
    k: float = 1.0,
) -> OutlierFlags:
    """Flag points falling outside mean +/- k*sqrt(mean).

    For a Poisson (shot-noise-limited) process, the standard deviation equals
    sqrt(mean), so mean +/- sqrt(mean) is the expected +/-1-sigma envelope.
    Points outside k*sqrt(mean) are candidate dust events or slow-mode
    fluctuations. The fraction flagged is a compact quality metric.

    Note: sqrt(mean) equals the Poisson standard deviation only when the values
    are raw photon counts. For count *rates* (cps), this envelope is a useful
    heuristic but is not the literal shot-noise band unless the rate happens to
    be numerically close to the count. Interpret accordingly.

    Parameters
    ----------
    trace : IntensityTrace
    k : float
        Multiplier on sqrt(mean) defining the envelope half-width (default 1.0).

    Returns
    -------
    OutlierFlags

    Raises
    ------
    ValueError
        If the mean count rate is negative (sqrt undefined) or k <= 0.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k!r}.")
    y = trace.count_rates_cps
    mean = float(y.mean())
    if mean < 0:
        raise ValueError(
            f"Mean count rate is negative ({mean}); sqrt(mean) is undefined."
        )
    sqrt_mean = math.sqrt(mean)
    lower = mean - k * sqrt_mean
    upper = mean + k * sqrt_mean
    mask = (y < lower) | (y > upper)
    n_flagged = int(mask.sum())
    return OutlierFlags(
        flagged_mask=mask,
        n_flagged=n_flagged,
        fraction_flagged=n_flagged / y.size if y.size else 0.0,
        mean_cps=mean,
        sqrt_mean=sqrt_mean,
        k=k,
        lower_bound_cps=lower,
        upper_bound_cps=upper,
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
    if baseline == 0:
        raise ValueError(
            "Estimated baseline is zero; cannot normalize the trace."
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
        if window_s <= 0:
            raise ValueError(f"window_s must be positive, got {window_s!r}.")
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
        for i in range(n):
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
    correlation_threshold: float = 1.5,
) -> BlockVarianceResult:
    """Blocking analysis of a trace: standard error of the mean vs block size.

    Partitions the trace into blocks of increasing size, computes the mean of
    each block, and estimates the standard error of the overall mean at each
    block size as std(block_means)/sqrt(n_blocks). For uncorrelated data this
    is flat; for positively correlated data (slow modes) it rises with block
    size and plateaus at the true standard error.

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
        curve is returned so you can judge for yourself. Default 1.5.

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

    block_sizes = []
    standard_errors = []
    n_blocks_list = []
    for b in raw:
        m = n // b
        if m < 2:
            continue
        trimmed = y[:m * b].reshape(m, b)
        block_means = trimmed.mean(axis=1)
        se = block_means.std(ddof=1) / math.sqrt(m)
        block_sizes.append(int(b))
        standard_errors.append(float(se))
        n_blocks_list.append(int(m))

    block_sizes = np.array(block_sizes)
    standard_errors = np.array(standard_errors)
    n_blocks_arr = np.array(n_blocks_list)

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
        se_ratio = se_large / se_small if se_small > 0 else float('nan')
    elif n_se >= 2 and standard_errors[0] > 0:
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
    )


def _freedman_diaconis_bins(values: np.ndarray) -> int:
    """Number of histogram bins by the Freedman-Diaconis rule.

    Bin width = 2 * IQR / n^(1/3). Robust to outliers because it uses the
    interquartile range rather than the standard deviation.
    """
    n = values.size
    q75, q25 = np.percentile(values, [75, 25])
    iqr = q75 - q25
    if iqr <= 0:
        return 0  # signals caller to fall back
    bin_width = 2.0 * iqr / (n ** (1.0 / 3.0))
    data_range = values.max() - values.min()
    if bin_width <= 0 or data_range <= 0:
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
    fano = var_counts / mean_counts if mean_counts != 0 else float('nan')
    mean_rate = float(y.mean())
    std_rate = float(y.std(ddof=1)) if y.size > 1 else 0.0
    cv = std_rate / mean_rate if mean_rate != 0 else float('nan')
    shot_noise_cv = 1.0 / math.sqrt(mean_counts) if mean_counts > 0 else float('nan')

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
            p0 = [counts.max(), mean_rate, std_rate if std_rate > 0 else 1.0]
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
        except (RuntimeError, ValueError):
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
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError as exc:
        raise ImportError(
            "test_stationarity_adf requires statsmodels, which is not "
            "installed. Install it with:  pip install statsmodels"
        ) from exc

    y = trace.count_rates_cps
    result = adfuller(y, autolag='AIC')
    adf_stat, p_value, _, _, crit_values, _ = result
    return StationarityResult(
        adf_statistic=float(adf_stat),
        p_value=float(p_value),
        critical_values={k: float(v) for k, v in crit_values.items()},
        significance=significance,
        is_stationary=bool(p_value < significance),
    )
