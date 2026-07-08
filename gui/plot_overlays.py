"""Shared GUI plot-overlay helpers (code-review D5).

Excluded/masked-point markers and their legend re-issue are GUI-owned overlays —
drawn on top of the analysis figure, never baked into the clean figures the
plotting layer saves (see the locked "flags are GUI-owned overlays" decision). The
DLS γ–q²/D–c plot, the DDLS plot, and the SLS Zimm/Berry/Debye/Guinier/Rayleigh
plot each add these overlays after their analysis layer has already built the
legend. The two small, style-parameterised helpers here keep those three draw paths
from drifting apart by hand (they had already diverged on fontsize / marker style).

Each caller keeps its own point-selection logic and its own style arguments, so the
rendering is identical to the hand-written blocks these replaced — the helpers only
own the shared *mechanics* (grey-× drawing with a single legend label; re-issuing
the legend while preserving any title)."""
from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

#: Grey used for the DLS/DDLS "excluded (unticked)" × markers.
EXCLUDED_GREY = '#999999'


def draw_excluded_markers(ax: Any, xy_points: Iterable[Tuple[float, float]], *,
                          zorder: int, label: str = 'excluded (unticked)',
                          color: str = EXCLUDED_GREY) -> bool:
    """Plot each ``(x, y)`` in ``xy_points`` as a grey × on ``ax``.

    Only the first marker is labelled, so the legend shows a single "excluded" entry
    rather than one per point. Returns ``True`` if anything was drawn (the caller
    then re-issues the legend to surface that entry). The marker style (``'x'`` /
    grey / ms=9 / mew=2) matches the historical DLS and DDLS overlays exactly;
    ``zorder`` is per-caller (DLS 4, DDLS 5)."""
    drew = False
    for x, y in xy_points:
        ax.plot([x], [y], 'x', color=color, ms=9, mew=2, zorder=zorder,
                label=label if not drew else None)
        drew = True
    return drew


def reissue_legend_preserving_title(ax: Any, *, fontsize: int,
                                    frameon: bool = False) -> None:
    """Re-issue ``ax``'s legend so overlay markers added *after* the analysis layer
    built it get an entry, preserving any existing legend title (e.g. the D-vs-c
    ``k_D`` title). If there is no current legend the title is simply ``None``. Style
    (``fontsize`` / ``frameon``) is per-caller so each plot renders as before."""
    leg = ax.get_legend()
    title: Optional[str] = leg.get_title().get_text() if leg is not None else None
    ax.legend(frameon=frameon, fontsize=fontsize, title=title or None)
