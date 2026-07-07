"""
gui/plot_controls.py
====================

A small reusable widget that puts axis controls *next to a plot* — per-axis
linear/log scale, an explicit min/max for each axis, and an autoscale button —
so the user can adjust a plot's axes directly (feedback B7) without digging into
matplotlib's hidden "figure options" dialog.

Usage: build one under a figure canvas, then call `attach(ax)` after each
(re)draw so the controls reflect the current axes (an axes object created by
`figure.clf()` + `add_subplot` is a new object, so re-attach each draw):

    self.axis_bar = AxisControlBar(self.canvas)
    ... after drawing ...
    self.axis_bar.attach(self.ax)
"""

from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtWidgets
from matplotlib.lines import Line2D

from gui.widgets import GripSplitter

# Colour of the drawn residual-resize grip line. Plots render on a white background
# (plot-background theming is deferred), so a fixed mid-grey reads on both themes.
_GRIP_COLOUR = '#888888'

from gui.theme import ThemedLabel


def make_canvas_expanding(canvas, min_height: int = 170,
                          max_height: Optional[int] = None):
    """Let a matplotlib FigureCanvas grow/shrink vertically with its container so
    the plot fills the available height (and resizes when the window or a splitter
    is dragged) instead of being pinned to its figsize (feedback 2026-06-26 #10).
    A small minimum keeps it usable; the figsize now only sets the default aspect.
    `max_height` optionally caps it so a plot does not balloon on a tall pane
    (feedback 2026-06-29 #9)."""
    canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                         QtWidgets.QSizePolicy.Policy.Expanding)
    canvas.setMinimumHeight(min_height)
    if max_height is not None:
        canvas.setMaximumHeight(max_height)
    return canvas


def make_split_panels(parent, left_min_width: Optional[int] = None,
                      sizes=(340, 760)):
    """Build a horizontal control-panel / plot splitter as `parent`'s sole child.

    Replaces the old fixed QHBoxLayout(control | plot) so the user can drag the
    divider to resize the control panel against the plot (feedback A3/B3). Returns
    (splitter, left_layout, right_layout); callers add their controls to
    `left_layout` and the figure/canvas to `right_layout` exactly as before.
    """
    outer = QtWidgets.QHBoxLayout(parent)
    outer.setContentsMargins(0, 0, 0, 0)
    splitter = GripSplitter(QtCore.Qt.Orientation.Horizontal)   # visible drag grip (#5/#9)
    outer.addWidget(splitter)

    left_widget = QtWidgets.QWidget()
    left = QtWidgets.QVBoxLayout(left_widget)
    left.setContentsMargins(0, 0, 0, 0)
    # Give the control column its OWN vertical scroll area so a tall set of controls
    # scrolls internally instead of forcing the whole tab past the viewport — that
    # overflow is what made the plot pane balloon and the page scroll (feedback
    # 2026-06-29 #9). Horizontal scrolling is off (forms wrap instead, #8).
    left_scroll = QtWidgets.QScrollArea()
    left_scroll.setWidgetResizable(True)
    left_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    # Horizontal bar only when a panel is genuinely wider than its column, so an
    # over-wide control (e.g. a long checkbox label) is still reachable rather than
    # clipped. The #9 fix was about the VERTICAL page scroll, not this.
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
    return splitter, left, right


def make_vertical_plot_stack(widgets, sizes=None, min_heights=None):
    """Stack several widgets in a draggable vertical QSplitter (with a visible grip
    handle) so each can be resized independently (feedback 2026-06-29 #9, 2026-06-30 #5/#9).

    Two uses: (1) genuinely INDEPENDENT plots with different x-axes (e.g. the Utilities
    trace vs its histogram diagnostic) — NOT a fit + its residual, which stay on one
    canvas via `attach_residual_resizer` so they keep their shared-x alignment; and (2)
    the stacked sections of a control column, so the user can resize the checklist /
    results table / etc. against each other. `min_heights` keeps each pane usable.
    Returns the splitter."""
    splitter = GripSplitter(QtCore.Qt.Orientation.Vertical)     # visible drag grip (#5/#9)
    for i, w in enumerate(widgets):
        splitter.addWidget(w)
        if min_heights is not None and i < len(min_heights):
            w.setMinimumHeight(min_heights[i])
    if sizes is not None:
        splitter.setSizes(list(sizes))
    return splitter


def attach_residual_resizer(canvas, fig, main_ax, resid_ax, apply_ratio,
                            margin_bottom=0.10, margin_top=0.93):
    """Make the boundary between a fit axes (`main_ax`) and its residual axes
    (`resid_ax`) — both on the SAME figure, sharing one x-axis — draggable, so the
    residual can be resized vertically while staying perfectly aligned under the fit
    (feedback 2026-06-29 #9, owner's request).

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
        # draggable (feedback 2026-06-30 #5). It is `animated` (excluded from the normal
        # draw) and painted on top in the draw_event via blitting, so it always tracks
        # the current gap position without triggering a redraw.
        self._grip = Line2D([0.45, 0.55], [0.5, 0.5], transform=fig.transFigure,
                            color=_GRIP_COLOUR, lw=3.0, solid_capstyle='round',
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
        label (feedback 2026-07-07). When we can measure the fit's x-axis extent at draw
        time we put the grip just below it; otherwise (no renderer yet, or nothing on
        the fit's x-axis reaches into the gap) we fall back to the geometric midpoint."""
        try:
            mpos = self.main_ax.get_position()
            rpos = self.resid_ax.get_position()
        except Exception:
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
        except Exception:
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

    def sync(self) -> None:
        """Repopulate the controls from the current axes' scales and limits."""
        if self._ax is None:
            return
        self._suppress = True
        self.x_scale.setCurrentText(self._ax.get_xscale())
        self.y_scale.setCurrentText(self._ax.get_yscale())
        (x0, x1), (y0, y1) = self._ax.get_xlim(), self._ax.get_ylim()
        self.x_min.setText(f'{x0:.4g}'); self.x_max.setText(f'{x1:.4g}')
        self.y_min.setText(f'{y0:.4g}'); self.y_max.setText(f'{y1:.4g}')
        self.note.clear()
        self._suppress = False

    # ---- handlers ----
    def _set_scale(self, axis: str, scale: str) -> None:
        if self._suppress or self._ax is None:
            return
        self.note.clear()
        try:
            (self._ax.set_xscale if axis == 'x' else self._ax.set_yscale)(scale)
            self._canvas.draw_idle()
        except Exception:
            self.note.setText(f'cannot set {axis} {scale}')
        self.sync()

    def _apply_limits(self) -> None:
        if self._suppress or self._ax is None:
            return
        self.note.clear()
        try:
            self._apply_axis('x', self.x_min, self.x_max, self.x_scale.currentText())
            self._apply_axis('y', self.y_min, self.y_max, self.y_scale.currentText())
        except ValueError as exc:
            self.note.setText(str(exc))
            return
        self._canvas.draw_idle()

    def _apply_axis(self, axis: str, lo_edit, hi_edit, scale: str) -> None:
        cur = self._ax.get_xlim() if axis == 'x' else self._ax.get_ylim()
        lo = self._parse(lo_edit, cur[0])
        hi = self._parse(hi_edit, cur[1])
        if not (lo < hi):
            raise ValueError(f'{axis}: min must be < max')
        if scale == 'log' and lo <= 0:
            raise ValueError(f'{axis}: log min must be > 0')
        (self._ax.set_xlim if axis == 'x' else self._ax.set_ylim)(lo, hi)

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
