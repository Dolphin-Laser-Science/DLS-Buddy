"""
app/analysis_region.py
======================

The DLS analysis region -- the delay window and the baseline span the user edits on the
DLS tab's Correlogram sub-tab -- as the fit inputs one fit of one measurement reads.

ONE READING, for every consumer. The Correlogram and Distribution sub-tabs and every
path that re-runs a fit behind the scenes (Γ vs q², D vs c, the Mode Map, DDLS and
replicate averaging, each through its "use the Correlogram tab's delay window and
baseline" checkbox) turn the region into engine kwargs HERE, so a region cannot mean one
thing on the tab that shows a fit and another on the path that promotes, pools, averages
or maps it. Qt-free (invariant 5): the GUI keeps the handles and the touched flags and
hands this module a `RegionState` snapshot.

The rule (owner-ruled 2026-09-03 for the baseline span, 2026-09-22/23 for the window and
the in-window reading; this docstring is the rule's home in the code):

- **Nothing the user has not set is applied.** An untouched delay window sends no window,
  so each run is fitted over its own full lag range; an untouched baseline span sends no
  baseline, so each fit uses its own existing estimate -- the parametric fits none, the
  nonlinear cumulant its index tail, an inversion the engine's tail estimate over the
  live channels inside the window. A region seeded from one run and carried to another
  therefore reaches nothing.
- **Nothing outside the delay window counts.** A placed span is measured over the
  channels the fit itself reads: past the skipped leading channels, inside the delay
  window this call sends (the run's full lag range when the window is untouched), and
  live (`g2 > 0`) -- the channels `analysis.dls` fits and measures its own baseline over.
  A span with no such channel counts as not placed.
- **Each fit is sent the one baseline input it reads.** The nonlinear cumulant takes the
  span itself (`baseline_window`), which it measures its floating baseline over and
  refuses below two channels; every other parametric arm takes a value to subtract
  (`baseline`) and gets one only when two or more channels survive, the nonlinear arm's
  own floor, so the two readings of one span agree on whether it was placed. An inversion
  takes a value (`baseline`) from one channel up: it normalizes by the baseline and has
  no floor of its own, and a span the user placed is theirs (invariant 4 clause 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from analysis.dls._common import _apply_tau_window, _live_channels

#: The fits that re-express the region as a subtracted value or a floating-baseline span.
PARAMETRIC_METHODS = ('cumulant', 'single', 'double', 'kww')
#: The inversions, which normalize by a baseline value.
INVERSION_METHODS = ('nnls', 'contin', 'lognormal')


@dataclass(frozen=True)
class RegionState:
    """A snapshot of the region, in seconds, with the two touched flags.

    `window_touched` / `baseline_touched` are True once the user moved a handle of that
    kind and False while it still holds the values the tab seeded from the data -- the
    Data tab's proposed-versus-asserted distinction. Only a touched part is applied.
    """
    tau_min_s: Optional[float] = None
    tau_max_s: Optional[float] = None
    base_lo_s: Optional[float] = None
    base_hi_s: Optional[float] = None
    window_touched: bool = False
    baseline_touched: bool = False

    def window(self) -> Optional[Tuple[Optional[float], Optional[float]]]:
        """The delay window a fit is sent, or None while the window is untouched."""
        if not self.window_touched:
            return None
        return (self.tau_min_s, self.tau_max_s)

    def span(self) -> Optional[Tuple[float, float]]:
        """The baseline span, ends ordered, or None while it is not placed."""
        if not self.baseline_touched or None in (self.base_lo_s, self.base_hi_s):
            return None
        lo, hi = sorted((float(self.base_lo_s), float(self.base_hi_s)))
        return (lo, hi)

    def applied(self) -> tuple:
        """What of the region a fit can read -- the key a checked run's staleness compares.

        Built from the two applied parts alone, so re-seeding an untouched handle (which
        reaches no fit) never flags a result stale, and every move of a touched one does.
        """
        return (self.window(), self.span())

    def is_set(self) -> bool:
        return self.window() is not None or self.span() is not None


def _fit_channels(tau: np.ndarray, g2m1: np.ndarray, window,
                  skip_initial_channels: int) -> Optional[np.ndarray]:
    """Boolean mask over the RAW axis of the channels a fit reads: past the skipped
    leading channels, inside `window`, and live.

    Computed through the engine's own `_apply_tau_window` and `_live_channels`, so the
    count a span is judged by is the count the fit takes (rule 12: one computation). The
    window is run over the channel INDICES, which comes back as the indices it keeps.
    None when the engine refuses the window or the skip outright -- the fit then raises
    that same refusal itself, and no baseline is worth sending to it.
    """
    tmin, tmax = window if window is not None else (None, None)
    try:
        _t, idx = _apply_tau_window(tau, np.arange(tau.size, dtype=float), tmin, tmax,
                                    min_points=0,
                                    skip_initial_channels=skip_initial_channels)
    except ValueError:
        return None
    mask = np.zeros(tau.size, dtype=bool)
    kept = idx.astype(int)
    mask[kept] = _live_channels(g2m1[kept])
    return mask


def region_fit_kwargs(tau, g2m1, method: str, region: Optional[RegionState], *,
                      cumulant_method: str = 'linear',
                      skip_initial_channels: int = 0) -> Dict:
    """The engine kwargs one `method` fit of one measurement reads from `region`.

    `tau` / `g2m1` are that measurement's raw delay axis (s) and correlogram (g2 - 1);
    `cumulant_method` decides which baseline input a cumulant fit reads, and
    `skip_initial_channels` is the fit's own, so the channels counted are the fit's.
    `region` None -- the checkbox unticked -- sends nothing, as does an untouched region.
    The rule is the module docstring's.
    """
    if method not in PARAMETRIC_METHODS + INVERSION_METHODS:
        raise ValueError(f"unknown DLS fit method {method!r}.")
    kw: Dict = {}
    if region is None:
        return kw
    window = region.window()
    if window is not None:
        if window[0] is not None:
            kw['tau_min_s'] = float(window[0])
        if window[1] is not None:
            kw['tau_max_s'] = float(window[1])
    span = region.span()
    if span is None:
        return kw
    tau = np.asarray(tau, dtype=float)
    g2m1 = np.asarray(g2m1, dtype=float)
    channels = _fit_channels(tau, g2m1, window, skip_initial_channels)
    if channels is None:
        return kw
    inside = channels & (tau >= span[0]) & (tau <= span[1])
    n = int(np.count_nonzero(inside))
    if method in INVERSION_METHODS:
        if n < 1:
            return kw
        kw['baseline'] = float(np.mean(g2m1[inside]))
        return kw
    if n < 2:
        return kw
    if method == 'cumulant' and cumulant_method == 'nonlinear':
        kw['baseline_window'] = span
    else:
        kw['baseline'] = float(np.mean(g2m1[inside]))
    return kw


def _seconds(x: float) -> str:
    return f'{float(x):.4g} s'


#: How every region note begins -- the one spelling a recorded mode set finds the map's
#: region note by, so the writer and the reader cannot drift apart.
REGION_NOTE_PREFIX = "Read the Correlogram tab's delay window and baseline"


def describe_region(region: Optional[RegionState]) -> str:
    """What a checked run read, in one sentence: the delay window and the baseline span
    it was sent, or that each was not set. '' for None (the box unticked), which runs on
    each run's own defaults exactly as before the box existed -- the silent default.

    ONE computation for every result's note, so a result, its export and a mode set
    recorded from it cannot describe the region differently."""
    if region is None:
        return ''
    window, span = region.window(), region.span()
    if window is None and span is None:
        return (f"{REGION_NOTE_PREFIX}: neither is set there, so each run used its own "
                f"full lag range and its fit's own baseline estimate.")
    if window is None:
        wtxt = 'window not set (each run used its full lag range)'
    else:
        lo = '—' if window[0] is None else _seconds(window[0])
        hi = '—' if window[1] is None else _seconds(window[1])
        wtxt = f'window {lo} to {hi}'
    if span is None:
        btxt = "baseline not placed (each fit used its own estimate)"
    else:
        btxt = (f'baseline measured over {_seconds(span[0])} to {_seconds(span[1])} '
                f'where that lies inside each run\'s delay window')
    return f"{REGION_NOTE_PREFIX}: {wtxt}; {btxt}."


def region_summary(region: Optional[RegionState]) -> str:
    """The checkbox's short label of what the region holds now -- whether or not the box
    is ticked, so the user sees what ticking it would apply."""
    window = region.window() if region is not None else None
    span = region.span() if region is not None else None
    if window is None and span is None:
        return 'Nothing set there yet: each run keeps its own lag range and baseline.'
    parts = []
    if window is not None:
        lo = '—' if window[0] is None else f'{float(window[0]):.3g}'
        hi = '—' if window[1] is None else f'{float(window[1]):.3g}'
        parts.append(f'window {lo}–{hi} s')
    else:
        parts.append('full lag range')
    if span is not None:
        parts.append(f'baseline {span[0]:.3g}–{span[1]:.3g} s')
    else:
        parts.append('no baseline placed')
    return 'Set there: ' + ', '.join(parts) + '.'


#: The words a Summary row or an averaging label appends for a checked run.
REGION_LABEL = "Correlogram window and baseline"
