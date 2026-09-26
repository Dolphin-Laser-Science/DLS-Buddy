"""
gui/plot_controls.py
====================

A small reusable widget that puts axis controls *next to a plot* — per-axis
linear/log scale, an explicit min/max for each axis, and an autoscale button —
so the user can adjust a plot's axes directly without digging into
matplotlib's hidden "figure options" dialog.

Usage: build one under a figure canvas, then call `attach(ax)` after each
(re)draw so the controls reflect the current axes (an axes object created by
`figure.clf()` + `add_subplot` is a new object, so re-attach each draw):

    self.axis_bar = AxisControlBar(self.canvas)
    ... after drawing ...
    self.axis_bar.attach(self.ax)
"""

from __future__ import annotations

import math
import os
from typing import Optional

from PySide6 import QtCore, QtWidgets
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

from gui.widgets import GripSplitter

# Color of the drawn residual-resize grip line. Plots render on a white background
# (plot-background theming is deferred), so a fixed mid-gray reads on both themes.
_GRIP_COLOR = '#888888'

from gui.theme import ThemedLabel


class _WhiteSaveToolbar(NavigationToolbar2QT):
    """A matplotlib navigation toolbar whose Save button always writes a WHITE figure,
    regardless of the on-screen "match app theme" setting (locked export policy, style
    guide R10.3). The recolour/save/restore lives in `plotting.plots.save_figure_via`
    (the policy's single implementation, shared with `save_figure_white`); this class
    only wraps matplotlib's own save dialog in it. When the plot is already white (the
    default) the recolour is a harmless no-op.

    When built with a controller it also ties matplotlib's save dialog into the
    app's remembered export folder (`settings.last_export_dir`): the dialog starts
    there, and the folder the user actually saves into is persisted back. The
    channel is matplotlib's own `rcParams['savefig.directory']`, which its Qt save
    dialog reads before and writes after a save (only when started non-empty —
    guaranteed here by the Documents fallback in `export_start_dir`)."""

    def __init__(self, canvas, parent, controller=None):
        super().__init__(canvas, parent)
        self._controller = controller

    def save_figure(self, *args):
        import matplotlib as mpl
        from plotting.plots import save_figure_via
        before = None
        if self._controller is not None:
            from gui.export_helper import export_start_dir, remember_export_dir
            before = export_start_dir(self._controller)
            mpl.rcParams['savefig.directory'] = before
        parent_save = super().save_figure       # bind here; a lambda has no __class__ cell
        save_figure_via(self.canvas.figure, lambda: parent_save(*args),
                        dressing=getattr(self.canvas, '_export_dressing', None))
        if self._controller is not None:
            after = mpl.rcParams['savefig.directory']
            if after and after != before:
                # matplotlib wrote the chosen folder back -> the user saved somewhere
                # new; remember it (append a dummy basename: remember_* takes a path).
                remember_export_dir(self._controller, os.path.join(after, 'x'))


def themed_navtoolbar(canvas, parent, controller=None):
    """A matplotlib `NavigationToolbar` whose Save always exports white (see
    `_WhiteSaveToolbar`). Drop-in replacement for `NavigationToolbar(canvas, parent)`
    at every embedded-plot site. Pass the controller to seed/remember the shared
    export folder across CSV exports and plot saves."""
    return _WhiteSaveToolbar(canvas, parent, controller)


def set_plot_export_name(canvas, stem: str) -> None:
    """Give this canvas's save dialog a meaningful suggested filename (instead of
    matplotlib's `image.png` fallback for embedded canvases). `stem` may carry an
    extension (e.g. a CSV suggested name — shared with the Export button); it is
    dropped in favour of the user's default plot filetype. Call at figure-refresh
    time with the analysis on screen."""
    bare = stem.rsplit('.', 1)[0] if '.' in stem else stem
    canvas.get_default_filename = \
        lambda: f'{bare}.{canvas.get_default_filetype()}'


def set_plot_export_dressing(canvas, dressing) -> None:
    """Give this canvas's Save button extra content to put on the WRITTEN figure.

    `dressing` is a zero-argument callable returning an undo callable; it runs
    inside the locked recolour in `plotting.plots.save_figure_via`, which is where
    the screen-versus-export split already lives. `None` clears it.

    Sibling of `set_plot_export_name`, and set at the same moment: figure-refresh
    time, with the analysis on screen. The one consumer is the DLS co-plot tabs,
    whose on-screen legend is a compact method key while the saved figure carries
    every measurement by name.
    """
    canvas._export_dressing = dressing


def make_canvas_expanding(canvas, min_height: int = 170,
                          max_height: Optional[int] = None):
    """Let a matplotlib FigureCanvas grow/shrink vertically with its container so
    the plot fills the available height (and resizes when the window or a splitter
    is dragged) instead of being pinned to its figsize.
    A small minimum keeps it usable; the figsize now only sets the default aspect.
    `max_height` optionally caps it so a plot does not balloon on a tall pane
    on a tall pane."""
    canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                         QtWidgets.QSizePolicy.Policy.Expanding)
    canvas.setMinimumHeight(min_height)
    if max_height is not None:
        canvas.setMaximumHeight(max_height)
    return canvas


# Object name of the control column's scroll area, the one scroll area a column's
# primary action may sit in (`pin_under_column` keeps the rest outside it).
CONTROL_COLUMN_SCROLL = 'control_column_scroll'


def make_split_panels(parent, left_min_width: Optional[int] = None,
                      sizes=(340, 760)):
    """Build a horizontal control-panel / plot splitter as `parent`'s sole child.

    Replaces the old fixed QHBoxLayout(control | plot) so the user can drag the
    divider to resize the control panel against the plot. Returns
    (splitter, left_layout, right_layout); callers add their controls to
    `left_layout` and the figure/canvas to `right_layout` exactly as before.
    """
    outer = QtWidgets.QHBoxLayout(parent)
    outer.setContentsMargins(0, 0, 0, 0)
    splitter = GripSplitter(QtCore.Qt.Orientation.Horizontal)   # visible drag grip
    outer.addWidget(splitter)

    left_widget = QtWidgets.QWidget()
    left = QtWidgets.QVBoxLayout(left_widget)
    left.setContentsMargins(0, 0, 0, 0)
    # Give the control column its OWN vertical scroll area so a tall set of controls
    # scrolls internally instead of forcing the whole tab past the viewport — that
    # overflow is what made the plot pane balloon and the page scroll.
    # Horizontal scrolling is off (forms wrap instead).
    left_scroll = QtWidgets.QScrollArea()
    left_scroll.setObjectName(CONTROL_COLUMN_SCROLL)
    left_scroll.setWidgetResizable(True)
    left_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    # Horizontal bar only when a panel is genuinely wider than its column, so an
    # over-wide control (e.g. a long checkbox label) is still reachable rather than
    # clipped. That vertical-scroll fix was about the VERTICAL page scroll, not this.
    left_scroll.setHorizontalScrollBarPolicy(
        QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    left_scroll.setWidget(left_widget)
    if left_min_width:
        left_scroll.setMinimumWidth(left_min_width + 18)   # room for the scrollbar

    right_widget = QtWidgets.QWidget()
    right = QtWidgets.QVBoxLayout(right_widget)
    right.setContentsMargins(0, 0, 0, 0)

    splitter.addWidget(left_scroll)
    splitter.addWidget(right_widget)
    splitter.setStretchFactor(0, 0)            # control panel keeps its width
    splitter.setStretchFactor(1, 1)            # plot takes the extra space
    splitter.setSizes(list(sizes))
    splitter._initial_sizes = list(sizes)      # re-applied by `pin_under_column`
    return splitter, left, right


def pin_under_column(splitter, *widgets):
    """Pin `widgets` under the control column's scroll area, outside it, so they never
    scroll out of view: a tab's primary action, its Export, its status line and its
    reliability flag (style guide R7.1, R8.6 and the §3 column template).

    `splitter` is the one `make_split_panels` returned. The column scroll area is
    re-parented into a new pane with the widgets stacked below it, in order, and that
    pane takes the scroll area's place in the splitter. Call it once per tab, after the
    column's contents are built. Returns the pane's layout, so a caller can pin more
    later.
    """
    pane = QtWidgets.QWidget()
    pane_layout = QtWidgets.QVBoxLayout(pane)
    pane_layout.setContentsMargins(0, 0, 0, 0)
    pane_layout.addWidget(splitter.widget(0), 1)   # re-parents the column scroll area
    for w in widgets:
        pane_layout.addWidget(w)
    splitter.insertWidget(0, pane)
    splitter.setStretchFactor(0, 0)                # the control pane keeps its width
    splitter.setStretchFactor(1, 1)
    splitter.setSizes(getattr(splitter, '_initial_sizes', splitter.sizes()))
    return pane_layout


def scrolling_notes(*labels, min_height: int = 40):
    """Stack word-wrapping flag/notes labels in a frameless scroll area for a results
    pane, and return it.

    A splitter pane's floor ignores a wrapped label's real height, so notes that name
    every ticked run would be clipped at the pane's bottom edge. In here they scroll
    instead, and the bar shows only when the notes are longer than the space the pane
    gives them. `min_height` keeps two lines in view.
    """
    box = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    for label in labels:
        lay.addWidget(label)
    lay.addStretch(1)
    sa = QtWidgets.QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    sa.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    sa.setWidget(box)
    sa.setMinimumHeight(min_height)
    return sa


# Stretch factors for a control column's three-pane stack (picker, settings, results;
# the §3 column template in the GUI style guide): extra window height goes mostly to the
# settings form, which scrolls until it fits, rather than to a results pane that stays
# empty until a Run.
COLUMN_STACK_STRETCH = (1, 2, 1)


def make_vertical_plot_stack(widgets, sizes=None, min_heights=None, stretch=None):
    """Stack several widgets in a draggable vertical QSplitter (with a visible grip
    handle) so each can be resized independently.

    Two uses: (1) genuinely INDEPENDENT plots with different x-axes (e.g. the Utilities
    trace vs its histogram diagnostic) — NOT a fit + its residual, which stay on one
    canvas via `attach_residual_resizer` so they keep their shared-x alignment; and (2)
    the stacked sections of a control column, so the user can resize the checklist /
    results table / etc. against each other. `min_heights` keeps each pane usable;
    `stretch` sets how extra height is shared out (one factor per pane).
    Returns the splitter."""
    splitter = GripSplitter(QtCore.Qt.Orientation.Vertical)     # visible drag grip
    for i, w in enumerate(widgets):
        splitter.addWidget(w)
        if min_heights is not None and i < len(min_heights):
            # Floor the requested minimum at the widget's own content minimum so a caller
            # can never specify a value that lets the splitter squeeze a pane UNDER its
            # children (which made the Distribution picker's bulk-select rows overlap the
            # checklist — style guide R3.2).
            w.setMinimumHeight(max(min_heights[i], w.minimumSizeHint().height()))
    if stretch is not None:
        for i, factor in enumerate(stretch):
            splitter.setStretchFactor(i, factor)
    if sizes is not None:
        splitter.setSizes(list(sizes))
    return splitter


def attach_residual_resizer(canvas, fig, main_ax, resid_ax, apply_ratio,
                            margin_bottom=0.10, margin_top=0.93):
    """Make the boundary between a fit axes (`main_ax`) and its residual axes
    (`resid_ax`) — both on the SAME figure, sharing one x-axis — draggable, so the
    residual can be resized vertically while staying perfectly aligned under the fit
    (owner's request).

    The fit + residual keep their guaranteed `sharex` alignment because they live on
    one canvas; only their height split changes. `apply_ratio(resid_fraction)` is a
    caller-supplied callback that updates the gridspec height ratios and redraws
    (resid_fraction is the residual's share of the stacked plotting area, clamped to
    a sensible band). `margin_bottom/top` are the figure-fraction extents of the
    stacked plotting area (set the same on the figure via subplots_adjust).

    Returns the controller object (kept alive by the caller, e.g. on the tab).
    """
    return _ResidualResizer(canvas, fig, main_ax, resid_ax, apply_ratio,
                            margin_bottom, margin_top)


class _ResidualResizer:
    _BAND_PX = 7                  # grab tolerance around the gap, in pixels
    _MIN_FRAC, _MAX_FRAC = 0.12, 0.50

    def __init__(self, canvas, fig, main_ax, resid_ax, apply_ratio,
                 margin_bottom, margin_top):
        self.canvas = canvas
        self.fig = fig
        self.main_ax = main_ax
        self.resid_ax = resid_ax
        self.apply_ratio = apply_ratio
        self.mb = margin_bottom
        self.mt = margin_top
        self._drag = False
        # A visible grip line drawn in the gap so users can tell the boundary is
        # draggable. It is `animated` (excluded from the normal
        # draw) and painted on top in the draw_event via blitting, so it always tracks
        # the current gap position without triggering a redraw.
        self._grip = Line2D([0.45, 0.55], [0.5, 0.5], transform=fig.transFigure,
                            color=_GRIP_COLOR, lw=3.0, solid_capstyle='round',
                            zorder=12, animated=True, visible=False)
        fig.add_artist(self._grip)
        canvas.mpl_connect('button_press_event', self._press)
        canvas.mpl_connect('motion_notify_event', self._motion)
        canvas.mpl_connect('button_release_event', self._release)
        canvas.mpl_connect('draw_event', self._on_draw)

    def _on_draw(self, _event) -> None:
        """Draw the grip line on top of the finished frame, centered in the fit/residual
        gap. Uses draw_artist + blit (not draw_idle) so it never recurses."""
        info = self._gap_y_px()
        if info is None:
            return
        y_px, x0_px, x1_px = info
        w = self.fig.bbox.width
        h = self.fig.bbox.height
        if not (w and h):
            return
        cx = 0.5 * (x0_px + x1_px) / w
        half = 0.035
        self._grip.set_data([cx - half, cx + half], [y_px / h, y_px / h])
        self._grip.set_visible(True)
        self.fig.draw_artist(self._grip)
        self.canvas.blit(self.fig.bbox)

    def _gap_y_px(self):
        """Pixel y of the grip line and the x-pixel span of the fit, or None if the
        axes have no valid position yet.

        The grip sits in the gap between the residual (below) and the fit (above).
        When the fit axes carries its OWN x-axis label — the Distribution tab, where
        fit and residual do not share x — that label (ticks + title) hangs down into
        the gap, so centering the grip on the geometric gap midpoint drops it onto the
        label. When we can measure the fit's x-axis extent at draw
        time we put the grip just below it; otherwise (no renderer yet, or nothing on
        the fit's x-axis reaches into the gap) we fall back to the geometric midpoint."""
        # No gap to grip while the plot is in its empty-state (axes hidden) — suppress the
        # grip line and its drag band so no stray dash floats over the placeholder message.
        if not (self.main_ax.get_visible() and self.resid_ax.get_visible()):
            return None
        try:
            mpos = self.main_ax.get_position()
            rpos = self.resid_ax.get_position()
        except Exception:  # noqa: BLE001 — geometry guard, no grip
            return None
        h = self.fig.bbox.height
        w = self.fig.bbox.width
        top_px = mpos.y0 * h            # fit axes bottom edge (figure pixels)
        resid_top_px = rpos.y1 * h      # residual axes top edge
        # If the fit's x-axis (tick labels + axis title) extends into the gap, use its
        # bottom instead of the axes edge so the grip clears it.
        try:
            renderer = self.canvas.get_renderer()
            bb = (self.main_ax.xaxis.get_tightbbox(renderer)
                  if renderer is not None else None)
        except Exception:  # noqa: BLE001 — geometry guard, axes edge used
            bb = None
        if bb is not None and bb.y0 < top_px:
            top_px = bb.y0
        y_gap = 0.5 * (resid_top_px + top_px)
        # Stay strictly inside the gap: a few px clear of the residual top, never above
        # the fit's bottom edge.
        y_gap = min(max(y_gap, resid_top_px + 3.0), mpos.y0 * h)
        return y_gap, mpos.x0 * w, mpos.x1 * w

    def _on_gap(self, event):
        info = self._gap_y_px()
        if info is None or event.x is None or event.y is None:
            return False
        y_gap, x0, x1 = info
        return abs(event.y - y_gap) <= self._BAND_PX and x0 <= event.x <= x1

    def _press(self, event):
        # Only a left-press squarely on the gap band starts a residual resize.
        if event.button == 1 and self._on_gap(event):
            self._drag = True

    def _motion(self, event):
        if not self._drag:
            # Hint that the boundary is draggable when hovering it.
            if self._on_gap(event):
                self.canvas.setCursor(QtCore.Qt.CursorShape.SplitVCursor)
            else:
                self.canvas.unsetCursor()
            return
        if event.y is None:
            return
        # Drag y (figure fraction) -> residual share of the [bottom, top] band.
        fy = event.y / self.fig.bbox.height
        span = self.mt - self.mb
        frac = (fy - self.mb) / span if span > 0 else 0.25
        frac = min(max(frac, self._MIN_FRAC), self._MAX_FRAC)
        self.apply_ratio(frac)

    def _release(self, event):
        self._drag = False

    def consumes_press(self, event) -> bool:
        """True if a press should be handled as a residual resize (so a host canvas
        with its own press handler can defer to this first)."""
        return event.button == 1 and self._on_gap(event)


class AxisControlBar(QtWidgets.QWidget):
    """On-plot axis controls: lin/log + min/max per axis + autoscale."""

    def __init__(self, canvas, parent=None) -> None:
        super().__init__(parent)
        self._canvas = canvas
        self._ax = None
        self._suppress = False

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        self.x_scale = QtWidgets.QComboBox(); self.x_scale.addItems(['linear', 'log'])
        self.y_scale = QtWidgets.QComboBox(); self.y_scale.addItems(['linear', 'log'])
        self.x_min, self.x_max = self._edit(), self._edit()
        self.y_min, self.y_max = self._edit(), self._edit()
        self.x_scale.currentTextChanged.connect(lambda s: self._set_scale('x', s))
        self.y_scale.currentTextChanged.connect(lambda s: self._set_scale('y', s))
        auto = QtWidgets.QPushButton('Autoscale')
        auto.clicked.connect(self._autoscale)

        row.addWidget(QtWidgets.QLabel('X'))
        row.addWidget(self.x_scale)
        row.addWidget(self.x_min); row.addWidget(self.x_max)
        row.addSpacing(10)
        row.addWidget(QtWidgets.QLabel('Y'))
        row.addWidget(self.y_scale)
        row.addWidget(self.y_min); row.addWidget(self.y_max)
        row.addStretch(1)
        row.addWidget(auto)
        self.note = ThemedLabel('', role='error', size=11)
        row.addWidget(self.note)
        # One passive tooltip on every control: the bar starts DISABLED on each plot
        # tab and stays so until a plot is drawn, and a greyed control with no word
        # of why is the widget audit's most-hit class. Per control, not on the bar:
        # a tooltip is read off the hovered widget itself.
        for w in (self.x_scale, self.x_min, self.x_max,
                  self.y_scale, self.y_min, self.y_max, auto):
            w.setToolTip('Axis scale and limits of the plot above; '
                         'available once a plot is drawn.')
        self.setEnabled(False)

    def _edit(self) -> QtWidgets.QLineEdit:
        e = QtWidgets.QLineEdit()
        e.setMaximumWidth(72)
        e.setPlaceholderText('auto')
        e.editingFinished.connect(self._apply_limits)
        return e

    # ---- public ----
    def attach(self, ax) -> None:
        """Point the bar at the current axes (None disables it) and sync controls."""
        self._ax = ax
        self.setEnabled(ax is not None)
        self.sync()

    def sync(self, *, clear_note: bool = True) -> None:
        """Repopulate the controls from the current axes' scales and limits.
        ``clear_note=False`` lets a caller resync the combo/edits to the actual axes
        state without wiping a note it just set (e.g. a rejected scale change)."""
        if self._ax is None:
            return
        self._suppress = True
        self.x_scale.setCurrentText(self._ax.get_xscale())
        self.y_scale.setCurrentText(self._ax.get_yscale())
        (x0, x1), (y0, y1) = self._ax.get_xlim(), self._ax.get_ylim()
        self.x_min.setText(f'{x0:.4g}'); self.x_max.setText(f'{x1:.4g}')
        self.y_min.setText(f'{y0:.4g}'); self.y_max.setText(f'{y1:.4g}')
        if clear_note:
            self.note.clear()
        self._suppress = False

    # ---- handlers ----
    def _set_scale(self, axis: str, scale: str) -> None:
        if self._suppress or self._ax is None:
            return
        self.note.clear()
        if scale == 'log':
            lo, _ = self._ax.get_xlim() if axis == 'x' else self._ax.get_ylim()
            if lo <= 0:
                self.note.setText(f'{axis}: log min must be > 0')
                self.sync(clear_note=False)   # revert the combo to the unchanged scale
                return
        try:
            (self._ax.set_xscale if axis == 'x' else self._ax.set_yscale)(scale)
            self._canvas.draw_idle()
        except Exception:  # noqa: BLE001 — surfaced in the note
            self.note.setText(f'cannot set {axis} {scale}')
            self.sync(clear_note=False)
            return
        self.sync()

    def _apply_limits(self) -> None:
        if self._suppress or self._ax is None:
            return
        self.note.clear()
        try:
            x_lo, x_hi = self._validate_axis(
                'x', self.x_min, self.x_max, self.x_scale.currentText())
            y_lo, y_hi = self._validate_axis(
                'y', self.y_min, self.y_max, self.y_scale.currentText())
        except ValueError as exc:
            self.note.setText(str(exc))
            return
        self._ax.set_xlim(x_lo, x_hi)
        self._ax.set_ylim(y_lo, y_hi)
        self._canvas.draw_idle()

    def _validate_axis(self, axis: str, lo_edit, hi_edit, scale: str) -> tuple:
        """Parse + validate one axis's edited limits WITHOUT applying them, so both
        axes can be checked before either is mutated (a valid X must not be applied
        if Y then turns out invalid)."""
        cur = self._ax.get_xlim() if axis == 'x' else self._ax.get_ylim()
        lo = self._parse(lo_edit, cur[0])
        hi = self._parse(hi_edit, cur[1])
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f'{axis}: min must be < max, both finite numbers')
        if scale == 'log' and lo <= 0:
            raise ValueError(f'{axis}: log min must be > 0')
        return lo, hi

    @staticmethod
    def _parse(edit: QtWidgets.QLineEdit, default: float) -> float:
        text = edit.text().strip()
        if not text:
            return float(default)
        try:
            return float(text)
        except ValueError:
            raise ValueError('limits must be numbers') from None

    def _autoscale(self) -> None:
        if self._ax is None:
            return
        self._ax.relim()
        self._ax.autoscale(enable=True)
        self._canvas.draw_idle()
        self.sync()
