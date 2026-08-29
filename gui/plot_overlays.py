"""Shared GUI plot-overlay helpers (code-review D5).

Excluded/masked-point markers and their legend re-issue are GUI-owned overlays —
drawn on top of the analysis figure, never baked into the clean figures the
plotting layer saves (see the locked "flags are GUI-owned overlays" decision). The
DLS γ–q²/D–c plot, the DDLS plot, and the SLS Zimm/Berry/Debye/Guinier/Rayleigh
plot each add these overlays after their analysis layer has already built the
legend. The two small, style-parameterized helpers here keep those three draw paths
from drifting apart by hand (they had already diverged on fontsize / marker style).

Each caller keeps its own point-selection logic and its own style arguments, so the
rendering is identical to the hand-written blocks these replaced — the helpers only
own the shared *mechanics* (gray-× drawing with a single legend label; re-issuing
the legend while preserving any title)."""
from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

#: Gray used for the DLS/DDLS "excluded (unticked)" × markers.
EXCLUDED_GRAY = '#999999'


def draw_excluded_markers(ax: Any, xy_points: Iterable[Tuple[float, float]], *,
                          zorder: int, label: str = 'excluded (unticked)',
                          color: str = EXCLUDED_GRAY) -> bool:
    """Plot each ``(x, y)`` in ``xy_points`` as a gray × on ``ax``.

    Only the first marker is labeled, so the legend shows a single "excluded" entry
    rather than one per point. Returns ``True`` if anything was drawn (the caller
    then re-issues the legend to surface that entry). The marker style (``'x'`` /
    gray / ms=9 / mew=2) matches the historical DLS and DDLS overlays exactly;
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


#: Neutral grey for an empty-plot placeholder message. A Matplotlib canvas colour, not a
#: Qt theme token (theme tokens don't reach the canvas). Same value the SLS tab used.
PLACEHOLDER_GRAY = '#999999'


def show_plot_placeholder(figure: Any, canvas: Any, message: str, *,
                          axes: Iterable[Any] = (),
                          clear_figure: bool = False) -> Any:
    """Render a plot area's empty-state as a single centred grey message with no visible
    plot box — the shared "nothing to show yet" look used by every DLS/SLS plot tab.

    Two paths, one look. A tab that keeps **persistent axes** passes them in ``axes`` and
    they are hidden (so no blank 0–1 box reads as broken); a tab that **rebuilds its axes
    on every draw** (SLS) passes ``clear_figure=True`` to wipe them. Either way the message
    is drawn with :meth:`figure.text` — a single Text artist, **no new Axes** — which is
    deliberate: recreating an Axes on each empty-state refresh churned Matplotlib/Agg
    teardown enough to trip a Windows heap fault across the many windows the gui-smoke
    suite builds.

    Returns the Text artist so a persistent-axes caller can ``.remove()`` it (and re-show
    its axes) before the next real draw. A ``clear_figure`` caller need not keep it — the
    next ``figure.clf()`` wipes it."""
    if clear_figure:
        figure.clf()
    else:
        for ax in axes:
            ax.set_visible(False)
    txt = figure.text(0.5, 0.5, message, ha='center', va='center', color=PLACEHOLDER_GRAY)
    canvas.draw_idle()
    return txt
