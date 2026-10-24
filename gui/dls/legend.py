"""
The DLS plot keys: the marker, method and measurement legend entries, the key drawn beside
the Correlogram and Distribution plots, and the dressing a saved image's legend gets.
"""
from __future__ import annotations

from matplotlib.lines import Line2D

from gui.dls.fits import _METHOD_LINESTYLE, _fit_label, _neutral_ink


# THE LEGEND IS SPLIT BY DESTINATION (ui_style_guide R10.6). The SCREEN gets a
# compact key -- one entry per METHOD drawn plus the window/baseline markers, constant
# in the selection size -- and the SAVED file gets that plus one entry per MEASUREMENT,
# grouped under its sample, in a figure grown to hold it. `_draw_key` and
# `export_legend_dressing` carry the measurements behind each half.

#: Point size of the on-screen key, and of the export legend.
_KEY_FONT_PT = 7.0


_EXPORT_FONT_PT = 7.0


#: Legend `labelspacing`, in units of the font size.
_LEGEND_LABELSPACING = 0.3


# The on-screen key shortens two marker labels. `tau window (fit range)` and
# `baseline region` render 229 px wide together in the 160 px corner cell the key
# lives in (measured); at these they measure 93 px and fit with room. The full
# wording goes to the export legend, which has the width.
_KEY_SHORT_LABELS = {'τ window (fit range)': 'τ window',
                     'baseline region': 'baseline'}


#: A saved figure grows by its legend's own height, capped so a large selection
#: cannot emit a strip. 2.0 in is about 40 rows at 7 pt -- past any real selection,
#: and short of an image whose legend dwarfs its plot.
_EXPORT_GROW_CAP_IN = 2.0


#: The labels `_draw_markers` puts on the correlogram's own chrome artists -- DERIVED
#: from the shortening map above rather than typed again, so the two cannot drift.
#: Matched EXACTLY: a substring test on 'window'/'baseline' also caught any
#: measurement whose name contains either word -- `baseline run`, `baseline_90.asc`
#: -- pulling its data scatter into a key that is supposed to be constant in the
#: selection size, and into the exported legend a second time.
_MARKER_LABELS = tuple(_KEY_SHORT_LABELS)


def _marker_legend_entries(ax):
    """The window/baseline marker entries a correlogram axes has labelled itself."""
    handles, labels = ax.get_legend_handles_labels()
    return [(h, lab) for h, lab in zip(handles, labels, strict=True)
            if lab in _MARKER_LABELS]


def _method_legend_entries(methods):
    """One neutral-ink entry per method drawn, carrying that method's line style."""
    ink = _neutral_ink()
    return [(Line2D([], [], color=ink, lw=1.4, ls=_METHOD_LINESTYLE.get(k, '-')),
             _fit_label(k)) for k in methods]


def _measurement_legend_entries(selection, names_by_sample):
    """One entry per measurement, grouped under a bold sample row.

    `names_by_sample` is `{sample label: [(iid, unprefixed name)]}` in draw order.
    The sample is a GROUP ROW rather than a prefix on every entry -- the shape the
    measurement picker and the DLS Summary table already use, and what takes the
    longest entry from 76 characters to 44. Returns `(entries, group_labels)`; the
    caller bolds the group rows.
    """
    out, groups = [], []
    for sample, rows in names_by_sample.items():
        out.append((Line2D([], [], ls='none'), sample))
        groups.append(sample)
        for iid, name in rows:
            out.append((Line2D([], [], color=selection.color_for(iid),
                               marker=selection.marker_for(iid) or 'o',
                               markerfacecolor='none', ls='none'),
                        '   ' + name))
    return out, groups


def _draw_key(figure, entries, *, bbox=None, ncol=None):
    """Draw the on-screen KEY and return it, or None when there is nothing to show.

    `bbox` places it inside a region of the figure (the Correlogram tab's empty
    bottom-right gridspec cell); without one it sits below the panel, which is what
    the Distribution tab does -- that figure is a plain stack with no free cell.
    Clears any previous legend first: a figure legend outlives `ax.clear()`.
    """
    figure.legends.clear()
    if not entries:
        return None
    handles = [h for h, _lab in entries]
    labels = [_KEY_SHORT_LABELS.get(lab, lab) for _h, lab in entries]
    if bbox is not None:
        # BOTTOM-aligned inside the cell: the cell is empty by construction, but the
        # side view above spills its x tick labels down into it, so a top-aligned key
        # prints over them (measured).
        return figure.legend(handles, labels, loc='lower left',
                             bbox_to_anchor=(bbox.x0, bbox.y0, bbox.width,
                                             bbox.height),
                             frameon=False, fontsize=_KEY_FONT_PT, handlelength=2.0,
                             handletextpad=0.5, borderaxespad=0.0,
                             labelspacing=_LEGEND_LABELSPACING)
    return figure.legend(handles, labels, loc='outside lower center', frameon=False,
                         fontsize=_KEY_FONT_PT, handlelength=2.0,
                         ncol=ncol or len(labels), columnspacing=1.6,
                         labelspacing=_LEGEND_LABELSPACING)


def export_legend_dressing(figure, entries, groups):
    """A `save_figure_via` dressing: put the FULL legend on the written figure.

    Returns the zero-argument callable that hook wants, which itself returns an undo.
    The figure grows by the legend's own height (capped at `_EXPORT_GROW_CAP_IN`) so
    nothing is taken from either panel -- the saved plot is LARGER than the on-screen
    one, not rationed against its legend.

    ONE column, deliberately: matplotlib fills columns top-to-bottom, so a sample
    group splits across a column boundary and the right column clips the margin
    (both measured). One column needs no straddling rule at all.
    """
    def dress():
        keep = list(figure.legends)
        figure.legends.clear()
        size_in = figure.get_size_inches().copy()
        grow = min(len(entries) * _EXPORT_FONT_PT * 1.45 / figure.dpi + 0.12,
                   _EXPORT_GROW_CAP_IN)
        figure.set_size_inches(size_in[0], size_in[1] + grow)
        leg = figure.legend([h for h, _lab in entries],
                            [lab for _h, lab in entries],
                            loc='outside lower center', frameon=False,
                            fontsize=_EXPORT_FONT_PT, handlelength=2.0,
                            handletextpad=0.6, ncol=1,
                            labelspacing=_LEGEND_LABELSPACING)
        for text in leg.get_texts():
            if text.get_text() in groups:
                text.set_fontweight('bold')

        def undo():
            leg.remove()
            figure.legends.extend(keep)
            figure.set_size_inches(*size_in)
        return undo
    return dress
