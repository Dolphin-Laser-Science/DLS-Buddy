"""
gui/utilities_module.py
=======================

The Utilities tab: data-quality diagnostics and tools. **Sample-scoped** — the
shell's sidebar navigator selects the sample, exactly like the DLS and SLS tabs.

Inner tabs (all built):
  * **Traces** — intensity-trace diagnostics (a separate trace store; Absolute /
    Relative scale, outlier + running-average overlays, a histogram+Fano /
    block-variance sub-plot, ADF stationarity in the summary line).
  * **I·sin θ** — an optical-quality / alignment check over a sample's SLS
    measurements, with an Absolute / Relative (normalized) scale toggle. For an
    ideal isotropic, dust-free scattering volume the curve is flat across angle.
  * **Synthetic data** — the synthetic-dataset generator (correlogram / trace /
    multi-angle DLS / SLS slices) with per-artifact Preview + Save.
  * **Solvent Explorer** — the global, display-only solvent-property calculator
    (`gui/solvent_explorer_module.py`). It ignores the sidebar selection by
    design (nothing is ever written into a measurement); hosting it here keeps
    the shell at the six locked top-level tabs.

All analysis lives in the controller/engine; this widget only drives the controller
and displays what comes back.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from PySide6 import QtCore, QtWidgets

from matplotlib.figure import Figure

from analysis.synthetic_dataset import DDLS_SHAPES
from analysis.trace_analysis import (FALSE_ALARM_PROBABILITY_DEFAULT,
                                     HAMPEL_WINDOW_MIN_PTS, OUTLIER_K_DEFAULT,
                                     PSD_WINDOWS)
from app import synthetic
from app.controller import (TRACE_DIAGNOSTIC_PLOT_VIEWS,
                            TRACE_DIAGNOSTIC_VIEWS)
from app.loading import detect_trace, load_trace_files
from core import units as U
from core.naming import sample_display_name
from gui.plot_controls import (
    LazyCanvas, make_split_panels, make_canvas_expanding, make_vertical_plot_stack,
    set_plot_export_name, themed_navtoolbar,
)
from gui.export_helper import (
    EXPORT_STATUS_MS, export_start_dir, export_to_csv, open_start_dir, remember_open_dir,
    sample_suggested_name, trace_suggested_name)
from gui.help import add_help_to_groupbox, section_header
from gui.solvent_explorer_module import SolventExplorerModule
from gui.theme import ThemedLabel, set_flag
from gui.widgets import (roomy_tabs, SampleSelector, show_parse_notes,
                         value_unit_row as _value_unit_row)


from gui.worker import RunSlot, busy_notice, run_when_idle, runner

from plotting.plots import (
    plot_i_sin_theta, plot_synthetic_correlogram, plot_intensity_trace,
    plot_count_rate_histogram, plot_block_variance, plot_allan_deviation,
    plot_welch_psd, plot_periodogram,
    plot_synthetic_multi_dls, plot_synthetic_sls_set,
    cycle as _plot_cycle,
    display_factor as _plot_factor, display_unit as _plot_unit,
)
# The generic trace fallback (`_load_generic_trace`) parses here, behind its units prompt;
# the instrument parsers and the load itself are `app/loading.py`'s.
from parsers.base_parser import ParseError
from parsers.generic_trace import GenericTraceParser

class _HampelWindowSpinBox(QtWidgets.QSpinBox):
    """Width spinbox for the windowed-Hampel overlay.

    The engine's admissible set is ``{0 (auto)}`` together with the ODD widths at or
    above `HAMPEL_WINDOW_MIN_PTS` -- two disjoint pieces, which a contiguous Qt range
    cannot express. Stepping is therefore handled here rather than by snapping a value
    after the fact: snapping alone trapped the down arrow at the floor (stepping below
    it snapped straight back up), leaving `auto` unreachable once the user had left it.
    Typed input is still snapped, on commit, by `_snap_hampel_window`.
    """

    def stepBy(self, steps: int) -> None:
        value = self.value()
        if steps < 0 and value <= HAMPEL_WINDOW_MIN_PTS:
            self.setValue(0)                     # down out of the range -> auto
        elif steps > 0 and value == 0:
            self.setValue(HAMPEL_WINDOW_MIN_PTS)  # up out of auto -> the floor
        else:
            super().stepBy(steps)


_OUTLIER_COLOR = '#D55E00'   # Okabe-Ito vermilion, for the GUI-owned flag overlay
# The windowed layer's band and impulse markers. A SECOND frozen literal for the
# same reason as the one above, and it must stay distinguishable from it: both
# overlays can be on at once, so the two flag sets have to be told apart by
# colour as well as by marker (o vs x) -- Okabe-Ito bluish green against
# vermilion, a pairing that survives the common colour-vision deficiencies.
_HAMPEL_COLOR = '#009E73'
# Trace-overlay colours come LIVE from plotting.plots.cycle() at each redraw
# (spec engineering-04): the old frozen Okabe-Ito literal here never followed
# a Settings palette swap, which is exactly the staleness cycle() exists to
# prevent.

# Default seeds for the intensity-trace controls. These used to live in Settings;
# they are now plain in-tab fields seeded each launch
# from these fixed code constants (not persisted).
_DEFAULT_OUTLIER_K = OUTLIER_K_DEFAULT   # seeded from the engine, never restated
_DEFAULT_RUNAVG_WINDOW = 0     # 0 → auto (≈ n/20 points)
_DEFAULT_HAMPEL_WINDOW = 0     # 0 → auto (resolved from the trace length)
# Percent, for a spinbox a user reads in percent; the engine takes a fraction.
_DEFAULT_FALSE_ALARM_PERCENT = 100.0 * FALSE_ALARM_PROBABILITY_DEFAULT

# The Diagnostic combo's contents, seeded from the controller's single source so
# the tab and the export dispatch cannot disagree about which views exist.
_DIAG_VIEW_LABELS = TRACE_DIAGNOSTIC_VIEWS
_DIAG_PLOT_VIEWS = TRACE_DIAGNOSTIC_PLOT_VIEWS

# Display names for the engine's window keys, in the engine's own order.
_PSD_WINDOW_LABELS = dict(zip(
    PSD_WINDOWS, ('Hann', 'Welch W1 (1 − t²)', 'Welch W2 (1 − |t|)'),
    strict=True))

# View key -> plotter. Every key in _DIAG_PLOT_VIEWS must appear here; a missing
# one raises rather than drawing some other view's curve (pinned by a test).
_DIAG_PLOTTERS = {
    'histogram': plot_count_rate_histogram,
    'block_variance': plot_block_variance,
    'allan': plot_allan_deviation,
    'psd': plot_welch_psd,
    'periodogram': plot_periodogram,
}

# At/above this sample count the focus trace's histogram + block-variance
# diagnostics are computed on the worker thread (like the ADF test) so a long
# trace never freezes the UI; shorter traces stay synchronous — backgrounding a
# cheap compute would only add latency/flicker.
_HEAVY_TRACE_POINTS = 20_000


# Synthetic-data generator field groups: (key, label, default).
# Conditions are shared by DLS + SLS; the size populations drive DLS Rh, while the
# SLS sample block carries Mw/Rg/A2 (a different set of physical parameters).
# Temperature + viscosity are entered with selectable units (defaults °C and
# mPa·s) and handled apart; only the unitless conditions go
# through this simple list.
_SYNTH_CONDITIONS = [
    ('wavelength_nm', 'Wavelength (nm)', '532'),
    ('solvent_refractive_index', 'n (solvent)', '1.33'),
]
# Noise level default is the committed Synthetic-Clean corpus's floor (corr_noise /
# sls_noise = 0.005), never 0.0 — an exact set gives ~1e-15 relative SEs, printing a ± too long to show.
_SYNTH_DLS = [
    ('angle_deg', 'Angle, single (°)', '90'),
    ('beta', 'β (coherence)', '0.8'),
    ('noise_level', 'Noise level', '0.005'),
    # The residual instrument baseline: a flat offset on g2-1, left behind when a
    # correlator's MEASURED baseline misses its calculated one. Surfaced because the
    # condition was reachable only from the test suite -- the fixture could build it and
    # the shipped program could not.
    ('baseline', 'Baseline offset (g₂−1)', '0.0'),
    ('n_points', 'Points', '200'),
]
# Noise-realism fields: (key, label, default). The count rate is read only by the full
# noise model (enabled with it, the shown-but-disabled idiom the multi-tau fields
# use); the correlation applies whatever the model. Both default to "off": 300 kcps is a
# typical header value but is inert until the full model is chosen, and 0 correlation is
# today's independent draw -- so an untouched panel produces exactly the call it always did.
_SYNTH_NOISE = [
    ('mean_count_rate_kcps', 'Mean count rate (kcps)', '300'),
    ('noise_correlation', 'Channel noise correlation (lag 1)', '0'),
]
# Multi-tau lag-axis fields: (key, label, default). Shown always, enabled only when the
# lag axis is Multi-tau — a real correlator's axis is set by its sample time and block
# count, not by a point count, so `Points` is what these replace.
_SYNTH_MULTITAU = [
    ('multitau_dt0_us', 'Δt₀ (µs)', '0.025'),
    ('multitau_channels_per_octave', 'Channels/octave (L)', '8'),
    ('multitau_n_blocks', 'Blocks (octaves)', '22'),
]
# Count-rate trace fields: (key, label, default). These feed the "Count-rate
# trace" artifact, which until now generated at builder defaults with nothing
# surfaced -- so the trace diagnostics had no ground truth reachable from the UI.
# The offset trio plants a step that PERSISTS, as distinct from the point spikes:
# it is what makes the windowed-Hampel blind spot demonstrable (a step longer
# than the analysis window lands in the rolling baseline and raises no flags).
_SYNTH_TRACE = [
    ('duration_s', 'Duration (s)', '120'),
    ('dt_s', 'Δt (s)', '0.1'),
    ('mean_kHz', 'Mean count rate (kHz)', '300'),
    ('trace_cv', 'CV (fractional)', '0.03'),
    ('drift', 'Drift (fraction over run)', '0.0'),
    ('spikes', 'Dust spikes', '0'),
    ('offset_start_s', 'Offset start (s)', ''),
    ('offset_duration_s', 'Offset duration (s)', '0'),
    ('offset_amplitude', 'Offset amplitude (× mean)', '0.0'),
    ('sine_frequency_hz', 'Sine frequency (Hz)', '0.0'),
    ('sine_amplitude', 'Sine amplitude (× mean)', '0.0'),
]
_SYNTH_SLS = [
    ('mw', 'M<sub>w</sub> (g/mol)', '1e6'),
    ('rg_nm', 'R<sub>g</sub> (nm)', '55'),
    ('a2', 'A₂ (mol·mL/g²)', '9.5e-5'),
    ('dn_dc', 'dn/dc (mL/g)', '0.135'),
]
# Artifacts the generator can produce: (key, label, can_preview, can_save).
# "Full Mw-series test set" and "Architecture series" are multi-file/multi-sample batches
# (like a whole test-data folder), so they save to a subfolder rather than a single file.
_SYNTH_ARTIFACTS = [
    ('correlogram', 'Correlogram (one angle)', True, True),
    ('trace', 'Count-rate trace', True, True),
    ('multi_dls', 'Multi-angle DLS (all angles)', True, True),
    ('two_mode', 'Two-mode DLS (fast + slow, crossing)', True, True),
    ('mode_resolved', 'Mode-resolved linked set (two-mode + traces)', False, True),
    ('depolarized', 'Depolarized (VV/VH)', True, True),
    ('sls_zimm', 'SLS — full Zimm set', True, True),
    ('sls_ff_sphere', 'SLS — form factor P(q), sphere', True, True),
    ('sls_ff_debye', 'SLS — form factor P(q), coil', True, True),
    ('sls_ff_ellipsoid', 'SLS — form factor P(q), ellipsoid', True, True),
    ('sls_ff_mie', 'SLS — form factor P(q), Mie sphere', True, True),
    ('sls_ff_core_shell', 'SLS — form factor P(q), core-shell', True, True),
    ('sls_single_conc', 'SLS — single concentration (angular)', True, True),
    ('sls_single_angle', 'SLS — single angle (conc. series)', True, True),
    ('architecture', 'Architecture series (linear + branched)', True, True),
    ('form_factor', 'Form-factor series (SLS + DLS)', True, True),
    ('full_set', 'Full Mw-series test set', False, True),
]
# Depolarization/shape input rows: (key, label, default). Rod dims are (L, d);
# ellipsoid dims are (a, b) — the two dim labels relabel when the shape changes.
_SYNTH_DEPOL = [
    ('rho_v', 'ρ<sub>v</sub> (depolarization ratio)', '0.05'),
    ('vh_noise', 'VH noise level', '0.02'),
]


class UtilitiesModule(QtWidgets.QWidget):
    """Sample-scoped data-quality diagnostics and tools."""

    # Emitted after the synthetic generator injects data into the workspace, so the
    # shell can rebuild the sidebar/Data tab (the utilities module has no reference
    # to the main window).
    workspaceChanged = QtCore.Signal()
    selectionChanged = QtCore.Signal()   # emitted when the I·sinθ sample changes (mirror)
    # Emitted when the USER picks a sample in the I·sinθ dropdown, so the shell makes it
    # the single active sample and fans it out to every tab + the sidebar.
    sampleFocusRequested = QtCore.Signal(str)

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.sample_id: Optional[str] = None
        # The ADF test's run slot. Its epoch is the staleness token for every
        # backgrounded trace-view computation (the ADF stationarity test and, for long
        # traces, the histogram/block-variance diagnostics); bumped whenever the trace
        # view refreshes so a late result from a superseded view is dropped. A busy
        # ADF press is queued, retried at idle while its view is still the one shown.
        self._adf_slot = RunSlot(
            available=lambda target: target[0] == self._adf_slot.epoch,
            rerun=lambda target: self._dispatch_adf(*target[1:]))
        self._generate_slot = RunSlot()      # a busy Generate is refused
        # Per-trace diagnostics memo: (trace_id, 'hist'|'bv') -> result. The
        # histogram and block-variance are pure functions of the (immutable) trace
        # data, so each is computed ONCE and shared by the text line and the
        # sub-plot (they were recomputed for each). Cleared when the trace list is
        # rebuilt (refresh_traces).
        # Messages from overlays that could not be drawn, rebuilt on every redraw.
        self._overlay_notes: list = []
        self._diag_cache: dict = {}
        self._isin_refresh_pending = False   # dedup deferred I·sinθ refreshes
        # Warm the statsmodels import HERE, on the main thread, before any trace
        # view can dispatch the ADF stationarity test to the worker thread — a
        # first-time statsmodels/scipy.signal import on the worker under
        # PySide6's shiboken import hook can corrupt the heap (see the docstring
        # of analysis.trace_analysis.preload_stationarity_backend). Costs ~1 s once
        # at startup; a missing statsmodels is not an error here (the diagnostic
        # reports it when actually used).
        self.controller.preload_trace_backend()
        self._build_ui()
        self.set_measurement(None)

    # ------------------------------------------------------------------ UI ---
    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        self.inner = roomy_tabs(QtWidgets.QTabWidget())   # roomier tabs so labels don't clip
        self.inner.setMovable(True)              # drag to reorder sub-tabs
        outer.addWidget(self.inner)
        self.inner.addTab(self._build_traces_tab(), 'Traces')
        self.inner.addTab(self._build_isin_tab(), 'I·sin θ')
        self.inner.addTab(self._build_synth_tab(), 'Synthetic Generator')
        # The Solvent Explorer is a self-contained global calculator; it seeds its
        # own solvent in its __init__ and ignores the sample selection by design.
        self.solvent_explorer = SolventExplorerModule(self.controller)
        self.inner.addTab(self.solvent_explorer, 'Solvent Explorer')
        self.refresh_traces()

    def _build_isin_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        # The I·sin θ dropdown is a mirror/override of the shell's single active sample
        # picking here sets the active sample everywhere; the sidebar drives
        # it too. Only SLS-bearing samples can produce an I·sin θ plot (it needs the c = 0
        # angular intensities); an incompatible active sample shows a named empty-state.
        self.isin_selector = SampleSelector(
            self.controller,
            # Sized to run_i_sin_theta's own precondition (a c = 0 curve), not to
            # "has SLS" -- sls_item_ids holds only non-zero concentrations.
            predicate=lambda s: bool(
                s.solvent_reference_item_id or s.extra_solvent_reference_item_ids),
            label_fn=sample_display_name, title='Sample',
            help_text='Choose which sample to plot I·sin θ for.',
            help_bullets=['Only samples with a c = 0 (solvent-reference) curve appear.',
                          'I·sin θ uses the c = 0 (solvent-reference) curves.'])
        self.isin_selector.sampleChanged.connect(self._on_isin_sample)
        v.addWidget(self.isin_selector)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Scale:'))
        self.isin_abs = QtWidgets.QRadioButton('Absolute')
        self.isin_rel = QtWidgets.QRadioButton('Relative (normalized)')
        self.isin_abs.setChecked(True)
        group = QtWidgets.QButtonGroup(self)
        group.addButton(self.isin_abs)
        group.addButton(self.isin_rel)
        # Connect only one of the pair: in a 2-button group, that button's state
        # flips on every toggle, so this fires exactly once per change.
        self.isin_abs.toggled.connect(self._update_isin)
        row.addWidget(self.isin_abs)
        row.addWidget(self.isin_rel)
        row.addStretch(1)
        v.addLayout(row)

        self.isin_fig = Figure(figsize=(4.7, 3.4))
        self.isin_canvas = make_canvas_expanding(LazyCanvas(self.isin_fig))
        self.isin_ax = self.isin_fig.add_subplot(111)
        v.addWidget(themed_navtoolbar(self.isin_canvas, w, self.controller))
        set_plot_export_name(self.isin_canvas, 'i_sin_theta')
        v.addWidget(self.isin_canvas, 1)

        self.isin_note = ThemedLabel('', role='hint', size=11)
        self.isin_note.setWordWrap(True)
        v.addWidget(self.isin_note)
        return w

    # ---------------------------------------------- intensity-trace tab ---
    def _build_traces_tab(self) -> QtWidgets.QWidget:
        """Count-rate trace diagnostics. Traces live in the workspace's separate
        store (not sample-scoped); load via the generic or ALV trace parsers."""
        w = QtWidgets.QWidget()
        _, left, right = make_split_panels(w, left_min_width=220, sizes=(220, 580))

        left.addWidget(section_header('Traces to Plot', 'Inspect count-rate traces '
                                      'for drift, dust and stationarity.', bullets=[
            'This list picks what is <b>plotted</b>; load, attach and remove traces '
            'in the <b>Workspace</b> sidebar.',
            'Tick traces to overlay them; the <b>focused</b> one (highlighted if '
            'ticked, else the first ticked) drives the stats line, the diagnostic '
            'sub-plot and every <b>Export …</b> button.',
            '<b>Scale</b>: <b>Absolute</b> or <b>Relative (baseline-normalized)</b>.',
            'Overlays: <b>Outliers</b> judges each point against the whole trace, '
            '<b>Windowed Hampel</b> against a moving window, both with one <b>k</b>; '
            '<b>Running average</b> draws a sliding mean.',
            'Widen the <b>Hampel window</b> to catch longer events; narrow it to '
            'follow a faster-moving baseline.',
            '<b>Diagnostic</b> picks the sub-plot and <b>Export diagnostic…</b> '
            'writes its numbers; reading each view: User-Manual §11.1.5.',
        ]))
        # A multi-select checklist (Select all/none), matching the DLS correlogram
        # tab's measurement selector: tick several traces to
        # overlay them. The focused trace drives the stats line + secondary diagnostic.
        self.trace_list = QtWidgets.QListWidget()
        self.trace_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        # ONE signal drives the redraw. itemSelectionChanged (not currentItemChanged)
        # is the right one: the plot overlays the whole selection, and Select all /
        # none change the selection set without moving the current item. Connecting
        # both fired _update_trace twice per click (double redraw + a doubled epoch
        # bump); the epoch guard kept the output right but the work was wasted.
        self.trace_list.itemSelectionChanged.connect(self._update_trace)
        left.addWidget(self.trace_list, 1)
        selrow = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton('Select all')
        btn_none = QtWidgets.QPushButton('Select none')
        btn_all.clicked.connect(lambda: self.trace_list.selectAll())
        btn_none.clicked.connect(lambda: self.trace_list.clearSelection())
        selrow.addWidget(btn_all)
        selrow.addWidget(btn_none)
        selrow.addStretch(1)
        left.addLayout(selrow)
        # No Load/Remove buttons here: membership lives on the workspace, like every
        # other kind of loaded data. Where those controls went is said in the two
        # places that have room for a full sentence -- the section's ? badge and the
        # statistics line opposite -- rather than in a hint under this list, whose
        # column has no vertical budget left and clips it to one truncated line.

        # scale toggle
        trow = QtWidgets.QHBoxLayout()
        trow.addWidget(QtWidgets.QLabel('Scale:'))
        self.trace_abs = QtWidgets.QRadioButton('Absolute')
        self.trace_rel = QtWidgets.QRadioButton('Relative (baseline-normalized)')
        self.trace_abs.setChecked(True)
        scale_group = QtWidgets.QButtonGroup(self)
        scale_group.addButton(self.trace_abs)
        scale_group.addButton(self.trace_rel)
        self.trace_abs.toggled.connect(self._update_trace)
        trow.addWidget(self.trace_abs)
        trow.addWidget(self.trace_rel)
        trow.addStretch(1)
        right.addLayout(trow)

        # overlay toggles (GUI-owned flags, never baked into the figure data)
        # 'k' tooltip: k multiplies the trace's OWN robust sigma, not the Poisson one.
        _k_help = ('k — the band multiplier shared by both outlier overlays: a point '
                   'is flagged outside median ± k·σ, σ being the trace’s own robust '
                   'scatter (1.4826·MAD). It finds single samples that stand out; '
                   'whether the whole trace is noisier than photon statistics allow '
                   'is the Fano factor’s question, in the summary line '
                   '(Theory-and-Equations-Guide §2.1.1 and §2.1.3).')
        _window_help = (
            'Width of the moving window in points (auto: sized from the trace '
            'length); events up to about half the window can be flagged, longer ones '
            'are tracked as baseline and show as a step in the rolling curve. A '
            'wider window flags less spuriously and leaves a wider gap not judged '
            'at each end (Theory-and-Equations-Guide §2.1.2).')
        # Two rows: the shared-k robust bands first, the unrelated smoother second.
        # Both rows would otherwise carry a control labelled 'window (pts)' meaning
        # two different things.
        orow = QtWidgets.QHBoxLayout()
        self.ov_outliers = QtWidgets.QCheckBox('Outliers')
        self.ov_outliers.toggled.connect(self._update_trace)
        orow.addWidget(self.ov_outliers)
        k_lbl = QtWidgets.QLabel('k:')
        k_lbl.setToolTip(_k_help)
        orow.addWidget(k_lbl)
        self.outlier_k = QtWidgets.QDoubleSpinBox()
        self.outlier_k.setRange(0.5, 6.0)
        self.outlier_k.setSingleStep(0.5)
        self.outlier_k.setValue(_DEFAULT_OUTLIER_K)        # seed (in-tab, not Settings)
        self.outlier_k.setToolTip(_k_help)
        self.outlier_k.valueChanged.connect(self._update_trace)
        orow.addWidget(self.outlier_k)
        self.ov_whampel = QtWidgets.QCheckBox('Windowed Hampel')
        self.ov_whampel.setToolTip(
            'Rolling median with its ±k·σ band, plus markers on the points that '
            'leave it. A gap in the band is a stretch not judged: no full window '
            'fits there, or over half the window shares one value; the summary line '
            'reports how much (Theory-and-Equations-Guide §2.1.2).')
        self.ov_whampel.toggled.connect(self._update_trace)
        orow.addWidget(self.ov_whampel)
        hw_lbl = QtWidgets.QLabel('Hampel window (pts):')
        hw_lbl.setToolTip(_window_help)
        orow.addWidget(hw_lbl)
        self.hampel_window = _HampelWindowSpinBox()
        # 0 = auto, then the engine's admissible range. Qt ranges are contiguous, so
        # 1..floor-1 are reachable by typing; _snap_hampel_window rounds them back up
        # ON COMMIT. Keyboard tracking is OFF so that snap cannot fire mid-word: with
        # it on, typing '51' snapped the intermediate '5' up to the floor and then
        # appended the '1', silently running the analysis at 111 points.
        # The engine still raises on a bad width — this only keeps the GUI from asking.
        self.hampel_window.setRange(0, 100_001)
        self.hampel_window.setKeyboardTracking(False)
        self.hampel_window.setSingleStep(2)                # stays odd on the arrows
        self.hampel_window.setSpecialValueText('auto')     # shown when value == 0
        self.hampel_window.setValue(_DEFAULT_HAMPEL_WINDOW)
        self.hampel_window.setToolTip(_window_help)
        self.hampel_window.valueChanged.connect(self._snap_hampel_window)
        self.hampel_window.valueChanged.connect(self._update_trace)
        orow.addWidget(self.hampel_window)
        orow.addStretch(1)
        right.addLayout(orow)

        orow = QtWidgets.QHBoxLayout()
        self.ov_running = QtWidgets.QCheckBox('Running average')
        self.ov_running.toggled.connect(self._update_trace)
        orow.addWidget(self.ov_running)
        # User-adjustable running-average window. The
        # sliding window spans this many points; 0 = auto (≈ n/20).
        win_lbl = QtWidgets.QLabel('window (pts):')
        win_lbl.setToolTip('Number of points in the sliding window; 0 = auto (≈ n/20).')
        orow.addWidget(win_lbl)
        self.runavg_window = QtWidgets.QSpinBox()
        self.runavg_window.setRange(0, 1_000_000)
        self.runavg_window.setSpecialValueText('auto')     # shown when value == 0
        self.runavg_window.setValue(_DEFAULT_RUNAVG_WINDOW)
        self.runavg_window.setToolTip(
            'Number of points in the sliding window; 0 = auto (≈ n/20).')
        self.runavg_window.valueChanged.connect(self._update_trace)
        orow.addWidget(self.runavg_window)
        orow.addStretch(1)
        # The running average is an overlay rather than a Diagnostic sub-plot, so
        # its export sits beside its own controls -- still one click from the view
        # it exports, like the Diagnostic one.
        self.runavg_export_button = QtWidgets.QPushButton('Export running average…')
        self.runavg_export_button.setToolTip(
            'Export the running mean and its within-window standard deviation at '
            'the window set here.')
        self.runavg_export_button.clicked.connect(
            self._export_focused_running_average)
        orow.addWidget(self.runavg_export_button)
        right.addLayout(orow)

        # The main trace plot and the secondary diagnostic are independent plots
        # (count rate vs time; histogram / block variance) with different x-axes, so
        # they go in a draggable vertical splitter — each resizable on its own.
        # (A fit + its residual, which must stay aligned,
        # share one canvas instead — see the DLS tabs.)
        top = QtWidgets.QWidget()
        tlay = QtWidgets.QVBoxLayout(top); tlay.setContentsMargins(0, 0, 0, 0)
        self.trace_fig = Figure(figsize=(5.2, 3.4))
        self.trace_canvas = make_canvas_expanding(LazyCanvas(self.trace_fig))
        self.trace_ax = self.trace_fig.add_subplot(111)
        tlay.addWidget(themed_navtoolbar(self.trace_canvas, w, self.controller))
        set_plot_export_name(self.trace_canvas, 'count_rate_trace')
        tlay.addWidget(self.trace_canvas, 1)
        # An overlay that could not be drawn says so here. It gets its own label
        # because `trace_stats` below is rewritten wholesale when the backgrounded
        # ADF verdict lands, which would silently eat the notice.
        self.trace_overlay_note = ThemedLabel('', role='error', size=11)
        self.trace_overlay_note.setWordWrap(True)
        self.trace_overlay_note.setVisible(False)
        tlay.addWidget(self.trace_overlay_note)
        self.trace_stats = ThemedLabel(
            'No trace selected. Load traces in the Workspace panel; '
            'tick one here to see its statistics.',
            role='muted', size=11)
        self.trace_stats.setWordWrap(True)
        tlay.addWidget(self.trace_stats)

        bottom = QtWidgets.QWidget()
        blay = QtWidgets.QVBoxLayout(bottom); blay.setContentsMargins(0, 0, 0, 0)
        drow = QtWidgets.QHBoxLayout()
        drow.addWidget(QtWidgets.QLabel('Diagnostic:'))
        self.diag_combo = QtWidgets.QComboBox()
        # Seeded from the controller's single source, so a view can never appear
        # here without an export behind it (see TRACE_DIAGNOSTIC_VIEWS). The key
        # rides on the item as user data, which is what the dispatch reads --
        # never the label, which is wording and may be restyled.
        for key in _DIAG_PLOT_VIEWS:
            self.diag_combo.addItem(_DIAG_VIEW_LABELS[key], key)
        self.diag_combo.setToolTip(
            'Secondary diagnostic plotted below the trace. The histogram and '
            'block variance describe the trace’s amplitudes; the Allan deviation '
            'says how long it can usefully be averaged; the spectrum and '
            'periodogram look for periodic contamination '
            '(Theory-and-Equations-Guide §2.1).')
        # Switching the sub-plot type reuses the cached diagnostics (or backgrounds
        # them for a long trace) rather than recomputing on the UI thread.
        self.diag_combo.currentIndexChanged.connect(self._on_diag_view_changed)
        drow.addWidget(self.diag_combo)

        # Per-view parameters. Only the selected view's controls are shown, so the
        # row stays one line: five views' worth of knobs on screen at once would
        # cost more space than the sub-plot itself.
        self.psd_params = QtWidgets.QWidget()
        prow = QtWidgets.QHBoxLayout(self.psd_params)
        prow.setContentsMargins(0, 0, 0, 0)
        _seg_help = (
            'Points per averaged segment. Longer segments resolve finer frequency '
            'detail but leave fewer of them to average, so the curve gets noisier; '
            '0 = auto (a power of two near an eighth of the trace). No value is '
            'prescribed — the trade is yours to make.')
        seg_lbl = QtWidgets.QLabel('segment (pts):')
        seg_lbl.setToolTip(_seg_help)
        prow.addWidget(seg_lbl)
        self.psd_nperseg = QtWidgets.QSpinBox()
        self.psd_nperseg.setRange(0, 1_000_000)
        self.psd_nperseg.setSpecialValueText('auto')       # shown when value == 0
        self.psd_nperseg.setValue(0)
        # Commit on focus-out/Enter, not per keystroke. Typing "512" otherwise
        # emits 5, then 51, then 512, and the first two are refused widths -- so
        # the view errored twice on the way to a value the user never asked for.
        # Same reason the Hampel width spinbox turns it off.
        self.psd_nperseg.setKeyboardTracking(False)
        self.psd_nperseg.setToolTip(_seg_help)
        self.psd_nperseg.valueChanged.connect(self._on_diag_view_changed)
        prow.addWidget(self.psd_nperseg)
        win_lbl = QtWidgets.QLabel('window:')
        _win_help = (
            'Data window applied to each segment before transforming. Hann is the '
            'modern default; W1 (1 − t²) and W2 (1 − |t|) are the two classical '
            'Welch windows, whose degrees-of-freedom figures the confidence band '
            'uses directly (Theory-and-Equations-Guide §2.1.6).')
        win_lbl.setToolTip(_win_help)
        prow.addWidget(win_lbl)
        self.psd_window = QtWidgets.QComboBox()
        for key, label in _PSD_WINDOW_LABELS.items():
            self.psd_window.addItem(label, key)
        self.psd_window.setToolTip(_win_help)
        self.psd_window.currentIndexChanged.connect(self._on_diag_view_changed)
        prow.addWidget(self.psd_window)
        drow.addWidget(self.psd_params)

        self.periodogram_params = QtWidgets.QWidget()
        frow = QtWidgets.QHBoxLayout(self.periodogram_params)
        frow.setContentsMargins(0, 0, 0, 0)
        _fap_help = (
            'False-alarm probability the significance line is drawn for: the chance '
            'that pure noise would produce a peak that tall somewhere in the '
            'spectrum. Lower means a stricter line. No level is prescribed; 1% is '
            'the usual convention (Theory-and-Equations-Guide §2.1.8).')
        fap_lbl = QtWidgets.QLabel('false alarm (%):')
        fap_lbl.setToolTip(_fap_help)
        frow.addWidget(fap_lbl)
        self.periodogram_fap = QtWidgets.QDoubleSpinBox()
        self.periodogram_fap.setDecimals(2)
        self.periodogram_fap.setRange(0.01, 50.0)
        self.periodogram_fap.setSingleStep(0.5)
        self.periodogram_fap.setValue(_DEFAULT_FALSE_ALARM_PERCENT)
        self.periodogram_fap.setKeyboardTracking(False)     # as psd_nperseg above
        self.periodogram_fap.setToolTip(_fap_help)
        self.periodogram_fap.valueChanged.connect(self._on_diag_view_changed)
        frow.addWidget(self.periodogram_fap)
        drow.addWidget(self.periodogram_params)

        drow.addStretch(1)
        # Export sits beside the view it exports, so the diagnostic's own data is
        # one click from the diagnostic (the sixth-block export ruling).
        self.diag_export_button = QtWidgets.QPushButton('Export diagnostic…')
        self.diag_export_button.setToolTip(
            'Export the selected diagnostic’s own data — the numbers behind the '
            'sub-plot, with its parameters and caveats in the header.')
        self.diag_export_button.clicked.connect(self._export_focused_diagnostic)
        drow.addWidget(self.diag_export_button)
        blay.addLayout(drow)
        self._sync_diag_params()
        self.diag_fig = Figure(figsize=(5.2, 2.6))
        self.diag_canvas = make_canvas_expanding(LazyCanvas(self.diag_fig), 130)
        self.diag_ax = self.diag_fig.add_subplot(111)
        blay.addWidget(self.diag_canvas, 1)

        right.addWidget(make_vertical_plot_stack(
            [top, bottom], sizes=[360, 220], min_heights=[160, 120]), 1)
        # Export lives at the bottom of the RESULTS column, the same place as on
        # every other tab. It is a results action, not a membership one, so the move
        # of load/remove to the workspace does not take it along -- and putting it on
        # this side keeps it from reading as the Remove button coming back.
        erow = QtWidgets.QHBoxLayout()
        erow.addStretch(1)
        self.trace_export_button = QtWidgets.QPushButton('Export CSV…')
        self.trace_export_button.setToolTip(
            'Export the focused trace as (time, count rate) columns.')
        self.trace_export_button.clicked.connect(self._export_focused_trace)
        erow.addWidget(self.trace_export_button)
        self.hampel_export_button = QtWidgets.QPushButton('Export decomposition…')
        self.hampel_export_button.setToolTip(
            'Export the windowed-Hampel decomposition of the focused trace: the '
            'rolling median and scale, the impulse flags, and the stretches not '
            'judged, with both bands’ settings in the header.')
        self.hampel_export_button.clicked.connect(self._export_focused_decomposition)
        erow.addWidget(self.hampel_export_button)
        right.addLayout(erow)
        return w

    def _snap_hampel_window(self, value: int) -> None:
        """Round a COMMITTED typed width onto the widths the engine accepts: 0
        (auto), or an odd width at or above the floor. Typing 1 or 12 is reachable
        because a Qt range is contiguous; this rounds rather than rejects, so the
        control never hands the engine a width it will refuse.

        Only ever sees committed values -- the spinbox has keyboard tracking off, so
        a half-typed number never reaches here. Arrow steps are handled by
        `_HampelWindowSpinBox.stepBy`, which owns the auto/floor boundary."""
        if value == 0:
            return
        snapped = max(HAMPEL_WINDOW_MIN_PTS, value)
        if snapped % 2 == 0:
            snapped += 1
        if snapped != value:
            with QtCore.QSignalBlocker(self.hampel_window):
                self.hampel_window.setValue(snapped)

    def _show_export_status(self, status: Optional[str]) -> None:
        """Report an export's outcome on the window's status bar, where the sidebar's
        trace export reports it: these tabs have no status line of their own."""
        win = self.window()
        if status and isinstance(win, QtWidgets.QMainWindow):
            win.statusBar().showMessage(status, EXPORT_STATUS_MS)

    def _export_focused_trace(self) -> None:
        """Export the focused count-rate trace as Origin-style (t, cps) columns."""
        tid = self._current_trace_id()
        if tid is None:
            return
        self._show_export_status(export_to_csv(
            self, self.controller, trace_suggested_name(self.controller, tid),
            lambda p: self.controller.export_trace(tid, p)))

    def _export_focused_decomposition(self) -> None:
        """Export the focused trace's windowed-Hampel decomposition."""
        tid = self._current_trace_id()
        if tid is None:
            return
        k, wp = self._windowed_params()
        name = trace_suggested_name(self.controller, tid)
        stem, _, ext = name.rpartition('.')
        self._show_export_status(export_to_csv(
            self, self.controller,
            f'{stem or name}_decomposition{("." + ext) if stem else ""}',
            lambda p: self.controller.export_trace_windowed_hampel(
                tid, p, k=k, window_pts=wp)))

    def _suggested_diag_name(self, tid: str, view: str) -> str:
        """The trace's export name with the view spliced in before the extension."""
        name = trace_suggested_name(self.controller, tid)
        stem, _, ext = name.rpartition('.')
        return f'{stem or name}_{view}{("." + ext) if stem else ""}'

    def _export_focused_diagnostic(self) -> None:
        """Export the selected diagnostic's own data for the focused trace.

        Parameters are read from the controls HERE and passed down, so the file
        records what is on screen rather than whatever the controller would have
        defaulted to — the same rule the decomposition export follows."""
        tid = self._current_trace_id()
        if tid is None:
            return
        view = self._current_diag_view()
        kw = {}
        if view == 'psd':
            nperseg, window = self._psd_params()
            kw = {'nperseg': nperseg, 'window': window}
        elif view == 'periodogram':
            (fap,) = self._periodogram_params()
            kw = {'false_alarm_probability': fap}
        self._show_export_status(export_to_csv(
            self, self.controller, self._suggested_diag_name(tid, view),
            lambda p: self.controller.export_trace_diagnostic(tid, view, p, **kw)))

    def _export_focused_running_average(self) -> None:
        """Export the focused trace's running average at the window on screen."""
        tid = self._current_trace_id()
        if tid is None:
            return
        # Resolved through the same helper the overlay draws with, so the file
        # holds the curve that is on screen rather than one at a nearby width.
        window = self._running_window(self.controller.trace_point_count(tid))
        self._show_export_status(export_to_csv(
            self, self.controller,
            self._suggested_diag_name(tid, 'running_average'),
            lambda p: self.controller.export_trace_diagnostic(
                tid, 'running_average', p, window_points=window)))

    def refresh_traces(self, select_id: Optional[str] = None) -> None:
        """Rebuild the trace list from the workspace (call after a load or a
        session load). Selects `select_id` if given; else keeps the current trace
        and the ticked set by id (a passive workspace refresh must not move the
        focus the exports and diagnostics follow); else the first row."""
        self._diag_cache.clear()             # trace set changed — drop the memo
        role = QtCore.Qt.ItemDataRole.UserRole
        cur = self.trace_list.currentItem()
        prev_current = cur.data(role) if cur is not None else None
        prev_selected = set(self._selected_trace_ids())
        # Signals stay blocked through the programmatic setCurrentRow so it does not
        # fire itemSelectionChanged; the single explicit _update_trace() below does
        # the one redraw (and always redraws, even if the current row is unchanged
        # but the underlying data reloaded).
        self.trace_list.blockSignals(True)
        self.trace_list.clear()
        row_of = {}
        for t in self.controller.traces():
            label = t.sample_label or t.trace_id
            item = QtWidgets.QListWidgetItem(f'{label}  [{t.trace_id}]')
            item.setData(role, t.trace_id)
            row_of[t.trace_id] = self.trace_list.count()
            self.trace_list.addItem(item)
        if select_id in row_of:
            self.trace_list.setCurrentRow(row_of[select_id])
        elif prev_current in row_of:
            self.trace_list.setCurrentRow(
                row_of[prev_current], QtCore.QItemSelectionModel.SelectionFlag.NoUpdate)
            for tid in prev_selected:
                if tid in row_of:
                    self.trace_list.item(row_of[tid]).setSelected(True)
        elif prev_selected & set(row_of):
            # The current trace is gone but ticked ones survive: keep them, and make
            # the first survivor current rather than clearing the overlay to row 0.
            survivors = [t for t in self.controller.traces()
                         if t.trace_id in prev_selected]
            self.trace_list.setCurrentRow(
                row_of[survivors[0].trace_id],
                QtCore.QItemSelectionModel.SelectionFlag.NoUpdate)
            for t in survivors:
                self.trace_list.item(row_of[t.trace_id]).setSelected(True)
        elif self.trace_list.count():
            self.trace_list.setCurrentRow(0)
        self.trace_list.blockSignals(False)
        self._update_trace()

    def _current_trace_id(self) -> Optional[str]:
        """The ONE focused trace every consumer reads — the plot's single-trace
        detail, the statistics line, the diagnostic sub-plot and the four
        'Export …' handlers. The list's current item when it is among the ticked
        traces, else the first ticked one: Qt keeps a Ctrl-deselected row current,
        so the raw item can name a trace that is neither plotted nor described."""
        it = self.trace_list.currentItem()
        cur = it.data(QtCore.Qt.ItemDataRole.UserRole) if it is not None else None
        tids = self._selected_trace_ids()
        if not tids or cur in tids:
            return cur
        return tids[0]

    def _selected_trace_ids(self) -> List[str]:
        out = []
        for it in self.trace_list.selectedItems():
            tid = it.data(QtCore.Qt.ItemDataRole.UserRole)
            if tid is not None:
                out.append(tid)
        return out

    def _trace_label(self, tid: str) -> str:
        for t in self.controller.traces():
            if t.trace_id == tid:
                return t.sample_label or t.trace_id
        return tid

    def _running_window(self, n_points: int) -> int:
        """Sliding-window length in points: the user value, or auto (≈ n/20).

        The auto rule is the engine's, not restated here, so the overlay and the
        export cannot disagree about what "auto" means."""
        w = int(self.runavg_window.value())
        return self.controller.running_average_window(n_points) if w == 0 else max(3, w)

    def _update_trace(self) -> None:
        # A refresh supersedes any in-flight backgrounded work (ADF, or a long
        # trace's diagnostics) from the previous view.
        self._adf_slot.bump()
        self.trace_ax.clear()
        self._overlay_notes = []
        self.trace_overlay_note.clear()
        self.trace_overlay_note.setVisible(False)
        tids = self._selected_trace_ids()
        focus = self._current_trace_id()     # already resolved into the ticked set
        if not tids and focus is not None:
            tids = [focus]
        if not tids:
            # Names the control by its real label, and where it now lives: loading
            # and removal moved to the Workspace panel with everything else.
            self.trace_stats.setText(
                'No trace ticked. Load count-rate traces in the Workspace panel '
                '(an instrument file or a two-column CSV), then tick one here.')
            self.trace_canvas.draw_idle()
            self._update_diag()
            return
        # Per refresh with a target-specific stem, like every DLS/SLS canvas
        # (gui/dls_module.py's set_plot_export_name(self.canvas, sample_suggested_name(...))).
        set_plot_export_name(self.trace_canvas, trace_suggested_name(self.controller, focus))
        mode = 'relative' if self.trace_rel.isChecked() else 'absolute'
        single = len(tids) == 1
        # Gather the buildable traces + stats first, so a relative view can fall back
        # to absolute UNIFORMLY when any shown trace lacks a usable baseline — B8:
        # never plot absolute data under a 'relative' axis (which the per-trace
        # `if baseline` fallback in _cr_transform did silently), and keep every
        # overlaid curve on the same scale.
        shown = []
        focus_stats = None
        for tid in tids:
            try:
                stats = self.controller.run_trace_statistics(tid)
                trace = self.controller.build_trace(tid)
            except Exception as exc:  # noqa: BLE001 — per-trace; surfaced when single
                if single:
                    self.trace_stats.setText(str(exc))
                continue
            if tid == focus:
                focus_stats = stats
            shown.append((tid, trace, stats))

        # Relative only when EVERY shown trace has a positive baseline; otherwise the
        # axis (and its label) fall back to absolute so the two never disagree.
        relative_requested = mode == 'relative'
        relative_ok = relative_requested and all(s.baseline_cps for _, _, s in shown)
        eff_mode = 'relative' if relative_ok else 'absolute'

        cyc = _plot_cycle()                      # live palette cycle (spec 04)
        for idx, (tid, trace, stats) in enumerate(shown):
            if single:
                plot_intensity_trace(trace, mode=eff_mode,
                                     baseline_cps=stats.baseline_cps, ax=self.trace_ax)
            else:
                # Overlay raw curves only (per-trace flag/running overlays are
                # single-trace detail — they'd be unreadable stacked).
                disp = self._cr_transform(eff_mode, stats.baseline_cps)
                t = np.asarray(trace.times_s, dtype=float)
                cr = np.asarray(trace.count_rates_cps, dtype=float)
                self.trace_ax.plot(t, disp(cr), lw=0.8,
                                   color=cyc[idx % len(cyc)],
                                   label=self._trace_label(tid))
        if not single:
            self.trace_ax.set_xlabel('Time (s)')
            downgraded = '' if relative_ok or not relative_requested else ' (no baseline)'
            self.trace_ax.set_ylabel(
                'Count rate / baseline' if eff_mode == 'relative'
                else f'Count rate ({_plot_unit("intensity")}){downgraded}')
            self.trace_ax.legend(frameon=False, fontsize=8)
        self.trace_fig.tight_layout()
        self.trace_canvas.draw_idle()
        # Diagnostics (text line + sub-plot) for the focus trace, sharing ONE cached
        # histogram/block-variance; for a long trace that compute goes to the worker.
        # The GUI-owned overlays draw inside the same continuation rather than in the
        # curve loop above: the windowed-Hampel overlay reads a memo `_ensure_focus_diag`
        # fills, so drawing it earlier computed it inline on the GUI thread and the
        # backgrounding never ran on the single-trace path -- the only path an overlay
        # draws on. Drawing after means a long trace's band lands when the worker does,
        # on axes that already hold the curve.
        if focus is not None and focus_stats is not None:
            prefix = ('' if single
                      else f'{len(tids)} traces · focused: {self._trace_label(focus)} — ')
            focus_trace = next((tr for tid, tr, _ in shown if tid == focus), None)
            self._ensure_focus_diag(
                focus,
                lambda: (self._draw_trace_overlays(focus, focus_trace, focus_stats,
                                                   eff_mode)
                         if single and focus_trace is not None else None,
                         self.trace_canvas.draw_idle(),
                         self._set_diag_text(focus, focus_stats, prefix=prefix),
                         self._update_diag()))
        else:
            # Fallback (no focus, or its statistics raised): _update_diag warms the
            # histogram/block-variance on the GUI thread, skipping the heavy-trace
            # worker route above. Confirmed safe: run_trace_statistics
            # cannot raise for a HEAVY trace -- it only raises on an empty array
            # (mutually exclusive with heavy) or a bad baseline method/parameter (both
            # hard-coded valid), and a build() failure makes _update_diag fail fast
            # rather than compute heavily. So this branch never runs a heavy synchronous
            # compute in practice; left synchronous deliberately.
            self._update_diag()

    def _cr_transform(self, mode: str, baseline_cps):
        """Map a canonical (cps) count-rate array to the plot's current y-scale: in
        relative mode divide by the baseline; in absolute mode convert to the active
        count-rate display unit (kcps by default), matching plot_intensity_trace."""
        if mode == 'relative' and baseline_cps:
            return lambda arr: np.asarray(arr, float) / baseline_cps
        ifac = _plot_factor('intensity')
        return lambda arr: np.asarray(arr, float) * ifac

    def _draw_trace_overlays(self, tid, trace, stats, mode: str) -> None:
        """Per-trace GUI-owned overlays (outlier flags + running average), drawn in
        the plot's current scale. Only used when a single trace is shown."""
        disp = self._cr_transform(mode, stats.baseline_cps)
        t = np.asarray(trace.times_s, dtype=float)
        cr = np.asarray(trace.count_rates_cps, dtype=float)
        if self.ov_outliers.isChecked():
            try:
                fl = self.controller.flag_trace_outliers(tid, k=self.outlier_k.value())
                mask = np.asarray(fl.flagged_mask, dtype=bool)
                # A degenerate trace flags nothing because it HAS no scale, which is a
                # different statement from "nothing stands out" — say which one it is,
                # or a stalled detector reads as a clean sample. NB the condition is over
                # half the samples sharing ANY value, not zeros specifically.
                lbl = ('no usable scale (over half the samples identical)'
                       if fl.scale_is_degenerate
                       else f'{fl.n_flagged} outliers (±{fl.k:g}·robust σ)')
                self.trace_ax.plot(t[mask], disp(cr[mask]), 'o',
                                   color=_OUTLIER_COLOR, ms=4, label=lbl)
                self.trace_ax.legend(frameon=False, fontsize=8)
            except Exception as exc:  # noqa: BLE001 — surfaced via _overlay_note
                self._overlay_note('Outliers overlay not drawn', exc)
        if self.ov_whampel.isChecked() and not self._windowed_ready(tid):
            # A cold memo here means the compute that should have filled it did not
            # finish -- `_ensure_focus_diag`'s `fail` branch runs this same
            # continuation. Computing it now would put the whole cost back on the GUI
            # thread, on the one path where the computation has already gone wrong.
            # The draw READS the memo and never fills it, so no caller can put a
            # synchronous compute back.
            self._overlay_note('Windowed Hampel not drawn',
                               'its diagnostics did not finish for this trace')
        elif self.ov_whampel.isChecked():
            try:
                wh = self._focus_windowed(tid)      # a memo hit, by the guard above
                centre = disp(wh.rolling_centre_cps)
                half_band = disp(wh.k * wh.rolling_scale_cps)
                mask = np.asarray(wh.flagged_mask, dtype=bool)
                # The engine already wrote NaN wherever it had no verdict, so both
                # artists break there on their own — the abstention gap is drawn,
                # not computed here (one representation, not two to keep in step).
                self.trace_ax.plot(
                    t, centre, '-', color=_HAMPEL_COLOR, lw=1.2,
                    label=f'rolling median, {wh.window_pts}-pt window '
                          f'(K = {wh.window_half_width_k})')
                self.trace_ax.fill_between(t, centre - half_band, centre + half_band,
                                           color=_HAMPEL_COLOR, alpha=0.15)
                self.trace_ax.plot(t[mask], disp(cr[mask]), 'x',
                                   color=_HAMPEL_COLOR, ms=6, mew=1.4,
                                   label=f'{wh.n_events} impulse events (windowed)')
                self.trace_ax.legend(frameon=False, fontsize=8)
            except Exception as exc:  # noqa: BLE001 — surfaced via _overlay_note
                self._overlay_note('Windowed Hampel overlay not drawn', exc)
        if self.ov_running.isChecked():
            try:
                wp = self._running_window(len(t))
                ra = self.controller.trace_running_average(tid, window_points=wp)
                tt = np.asarray(ra.times_s, dtype=float)
                mean = disp(ra.running_mean)
                std = disp(ra.running_std)        # disp is multiplicative → scales SD
                self.trace_ax.plot(tt, mean, '-', color='#000000', lw=1.2)
                self.trace_ax.fill_between(tt, mean - std, mean + std,
                                           color='#000000', alpha=0.15)
            except Exception as exc:  # noqa: BLE001 — surfaced via _overlay_note
                self._overlay_note('Running-average overlay not drawn', exc)

    def _overlay_note(self, what: str, detail) -> None:
        """Say that a requested overlay is missing, and why. `detail` is a written
        reason or the exception that stopped it.

        These three draws used to swallow every exception silently, which left a
        ticked checkbox with nothing under it and no way to tell an empty result
        from a failed one. A diagnostic whose contract is that it discloses when
        it cannot judge must not fail invisibly."""
        # A ValueError from the engine carries a written reason (a window below the
        # floor, a saturated band); anything else is a defect, so name its type
        # rather than paste an internal message into the panel.
        if isinstance(detail, BaseException):
            detail = (str(detail) if isinstance(detail, ValueError)
                      else type(detail).__name__)
        self._overlay_notes.append(f'{what}: {detail}')
        # One set_flag call over the joined text: the shared renderer prepends one
        # tier glyph, so accumulating through it would scatter glyphs mid-string.
        set_flag(self.trace_overlay_note, '   ·   '.join(self._overlay_notes))
        self.trace_overlay_note.setVisible(True)

    # -- shared / backgrounded trace diagnostics (histogram + block variance) --
    def _focus_histogram(self, tid: str):
        """Count-rate histogram for `tid`, memoized per trace (computed once, shared
        by the Fano text and the histogram sub-plot). Computes on the calling thread
        on a miss — the background path pre-warms this cache for long traces."""
        key = (tid, 'hist')
        if key not in self._diag_cache:
            self._diag_cache[key] = self.controller.trace_histogram(tid)
        return self._diag_cache[key]

    def _focus_block_variance(self, tid: str):
        """Block-variance analysis for `tid`, memoized per trace (shared by the
        correlation text and the block-variance sub-plot)."""
        key = (tid, 'bv')
        if key not in self._diag_cache:
            self._diag_cache[key] = self.controller.trace_block_variance(tid)
        return self._diag_cache[key]

    def _windowed_params(self):
        """(k, window_pts) as the controls currently request them; 0 → None (auto),
        which lets the engine resolve the width from the trace length."""
        w = int(self.hampel_window.value())
        return float(self.outlier_k.value()), (w if w else None)

    def _windowed_ready(self, tid: str) -> bool:
        """True when the memo already holds a decomposition for the CURRENT knobs
        (or none is wanted). Unlike the histogram and block variance, this result
        is not a pure function of the trace, so trace identity alone is not enough
        to say the cache is fresh."""
        if not self.ov_whampel.isChecked():
            return True
        entry = self._diag_cache.get((tid, 'wh'))
        return entry is not None and entry[:2] == self._windowed_params()

    def _focus_windowed(self, tid: str):
        """Windowed-Hampel decomposition for `tid`, memoized per trace AND per
        (k, window) — a trace-keyed memo would serve a stale band after the user
        moved either knob. One entry per trace, replaced rather than accumulated,
        so spinning a control does not grow the cache."""
        params = self._windowed_params()
        entry = self._diag_cache.get((tid, 'wh'))
        if entry is None or entry[:2] != params:
            k, wp = params
            entry = (k, wp, self.controller.flag_trace_outliers_windowed(
                tid, k=k, window_pts=wp))
            self._diag_cache[(tid, 'wh')] = entry
        return entry[2]

    # -- the parameter-bearing spectral views ---------------------------------
    # These follow the windowed-Hampel memo pattern, not the histogram one: each
    # depends on controls the user can move, so a trace-keyed cache would serve a
    # curve computed for settings that are no longer on screen.
    def _current_diag_view(self) -> str:
        """The selected Diagnostic view's KEY (never its label)."""
        key = self.diag_combo.currentData()
        return key if key else _DIAG_PLOT_VIEWS[0]

    def _psd_params(self):
        """(nperseg, window) as the controls request them; 0 → None (auto)."""
        n = int(self.psd_nperseg.value())
        return (n if n else None), str(self.psd_window.currentData())

    def _periodogram_params(self):
        """(false_alarm_probability,) as a fraction; the spinbox reads percent."""
        return (float(self.periodogram_fap.value()) / 100.0,)

    def _focus_psd(self, tid: str):
        """Welch spectrum for `tid`, memoized per trace AND per (segment, window)."""
        params = self._psd_params()
        entry = self._diag_cache.get((tid, 'psd'))
        if entry is None or entry[0] != params:
            nperseg, window = params
            entry = (params, self.controller.trace_welch_psd(
                tid, nperseg=nperseg, window=window))
            self._diag_cache[(tid, 'psd')] = entry
        return entry[1]

    def _focus_periodogram(self, tid: str):
        """Periodogram + false-alarm test for `tid`, memoized per trace and level.

        The controller supplies the correlation verdict that marks the result
        unreliable, so nothing here needs to know about the blocking analysis.
        """
        params = self._periodogram_params()
        entry = self._diag_cache.get((tid, 'pgram'))
        if entry is None or entry[0] != params:
            (fap,) = params
            entry = (params, self.controller.trace_periodogram(
                tid, false_alarm_probability=fap))
            self._diag_cache[(tid, 'pgram')] = entry
        return entry[1]

    def _focus_view(self, tid: str, view: str):
        """The result behind one Diagnostic view, from the memo.

        Allan and block variance deliberately share the block-variance entry:
        they are two readings of one computation, so cutting the trace twice
        would be wasted work and could not disagree usefully.
        """
        if view == 'histogram':
            return self._focus_histogram(tid)
        if view in ('block_variance', 'allan'):
            return self._focus_block_variance(tid)
        if view == 'psd':
            return self._focus_psd(tid)
        if view == 'periodogram':
            return self._focus_periodogram(tid)
        raise ValueError(f'Unknown diagnostic view {view!r}.')

    def _sync_diag_params(self) -> None:
        """Show only the selected view's parameter controls."""
        view = self._current_diag_view()
        self.psd_params.setVisible(view == 'psd')
        self.periodogram_params.setVisible(view == 'periodogram')

    def _trace_is_heavy(self, tid: str) -> bool:
        """True if the trace is long enough that its diagnostics should compute on
        the worker thread rather than block the UI."""
        try:
            return self.controller.trace_point_count(tid) >= _HEAVY_TRACE_POINTS
        except Exception:  # noqa: BLE001 — unknown trace is not heavy
            return False

    def _spectral_ready(self, tid: str) -> bool:
        """True when the memo already holds the SELECTED spectral view for the
        current knobs (or the selected view is not a spectral one).

        Only the selected view is checked, and only the spectral ones are checked
        at all. The histogram and block variance are always wanted because the
        stats line reports the Fano factor and the correlation verdict from them;
        a spectrum the user is not looking at is pure cost."""
        view = self._current_diag_view()
        if view == 'psd':
            entry = self._diag_cache.get((tid, 'psd'))
            return entry is not None and entry[0] == self._psd_params()
        if view == 'periodogram':
            entry = self._diag_cache.get((tid, 'pgram'))
            return entry is not None and entry[0] == self._periodogram_params()
        return True

    def _ensure_focus_diag(self, tid: Optional[str], then) -> None:
        """Ensure `tid`'s cached diagnostics are current, then run `then`
        (which renders the diagnostics text + sub-plot from that cache).

        Short traces compute synchronously; a long trace's heavy diagnostics are
        computed together on the worker thread — busy-guarded and epoch-dropped like
        the ADF test — so the UI never freezes. `then` reads only the memo, so it is
        identical whether reached synchronously or from the worker's completion.

        The histogram and block variance are always computed (the stats line needs
        both); the windowed decomposition and the spectral views are computed only
        when they are actually on screen."""
        if tid is None:
            then()
            return
        if ((tid, 'hist') in self._diag_cache and (tid, 'bv') in self._diag_cache
                and self._windowed_ready(tid) and self._spectral_ready(tid)):
            then()                       # already computed — nothing to wait for
            return
        if not self._trace_is_heavy(tid):
            self._focus_histogram(tid)   # warm the memo synchronously (cheap)
            self._focus_block_variance(tid)
            if self.ov_whampel.isChecked():
                self._focus_windowed(tid)
            view = self._current_diag_view()
            if view in ('psd', 'periodogram'):
                try:
                    self._focus_view(tid, view)
                except Exception:      # noqa: BLE001 — surfaced by _update_diag
                    # Leave the memo cold and let `then` run. `_update_diag`
                    # re-enters the same call inside its own try/except and
                    # renders the reason on the axes -- the handler that already
                    # exists for this. Letting it escape here instead killed the
                    # whole redraw, so the sub-plot went on showing the PREVIOUS
                    # spectrum while the controls read the new setting: the one
                    # outcome worse than an error message. The heavy path always
                    # had this, via its `fail` branch.
                    pass
            then()
            return

        epoch = self._trace_epoch
        controller = self.controller
        # Read the controls HERE, on the GUI thread: `compute` runs on the worker,
        # where touching a widget is not safe.
        want_windowed = self.ov_whampel.isChecked()
        k, wp = self._windowed_params()
        view = self._current_diag_view()
        nperseg, psd_window = self._psd_params()
        (fap,) = self._periodogram_params()

        def compute():
            spectral = None
            if view == 'psd':
                spectral = controller.trace_welch_psd(
                    tid, nperseg=nperseg, window=psd_window)
            elif view == 'periodogram':
                spectral = controller.trace_periodogram(
                    tid, false_alarm_probability=fap)
            return (controller.trace_histogram(tid),
                    controller.trace_block_variance(tid),
                    controller.flag_trace_outliers_windowed(tid, k=k, window_pts=wp)
                    if want_windowed else None,
                    spectral)

        def done(res) -> None:
            if epoch != self._trace_epoch:
                return                   # the view moved on — drop this result
            hist, bv, windowed, spectral = res
            self._diag_cache[(tid, 'hist')] = hist
            self._diag_cache[(tid, 'bv')] = bv
            if windowed is not None:
                self._diag_cache[(tid, 'wh')] = (k, wp, windowed)
            if spectral is not None and view == 'psd':
                self._diag_cache[(tid, 'psd')] = ((nperseg, psd_window), spectral)
            elif spectral is not None and view == 'periodogram':
                self._diag_cache[(tid, 'pgram')] = ((fap,), spectral)
            then()

        def fail(_exc) -> None:
            if epoch != self._trace_epoch:
                return
            then()                       # let `then` surface the error inline (guarded)

        if runner().try_submit(compute, done, fail,
                               description='trace diagnostics'):
            self.trace_stats.setText('Computing trace diagnostics…')
        else:
            # The worker is busy (a fit or ADF holds it): retry once it frees,
            # unless the view changed by then (parity with the ADF pending notice).
            # Each busy call queues its own `then`, so this site dispatches on the
            # gate directly: a RunSlot keeps one pending target and would drop one.
            self.trace_stats.setText('Computing trace diagnostics… (queued)')
            run_when_idle(lambda: self._ensure_focus_diag(tid, then)
                          if epoch == self._trace_epoch else None)

    def _on_diag_view_changed(self) -> None:
        """Diag sub-plot type or one of its parameters changed: re-render it
        (reusing the cache, or backgrounding for a long trace). The trace/text
        line is unaffected."""
        self._sync_diag_params()
        self._ensure_focus_diag(self._current_trace_id(), self._update_diag)

    def _set_diag_text(self, tid: str, stats, prefix: str = '') -> None:
        """One-line summary: trace stats + Fano factor + correlation + ADF verdict.
        The fast parts render immediately; the ADF stationarity test (the one slow
        call) runs on the worker thread and fills in when it finishes, so a long
        trace never freezes the window."""
        ifac, iu = _plot_factor('intensity'), _plot_unit('intensity')
        parts = [
            f'{stats.n_points} pts/{stats.duration_s:.1f}s',
            f'mean {stats.mean_cps * ifac:.4g} {iu}', f'CV {stats.cv:.3f}',
            f'baseline {stats.baseline_cps * ifac:.4g} {iu}',
        ]
        try:
            parts.append(f'Fano {self._focus_histogram(tid).fano_factor:.2f}')
        except Exception:  # noqa: BLE001 — optional stat omitted
            pass
        try:
            bv = self._focus_block_variance(tid)
            parts.append('correlated' if bv.correlations_detected else 'uncorrelated')
        except Exception:  # noqa: BLE001 — optional stat omitted
            pass
        # The windowed counts must be built into `base`, not appended to the label
        # afterwards: _dispatch_adf below rewrites the whole line when its verdict
        # lands, and anything written after this point is overwritten.
        if self.ov_whampel.isChecked():
            try:
                wh = self._focus_windowed(tid)
                parts.append(f'{wh.n_events} impulse events / {wh.n_flagged} samples '
                             f'of {wh.n_judged} judged (window {wh.window_pts} pts)')
                if wh.fraction_abstained:
                    # Named, not just totalled: an edge gap is expected geometry,
                    # while an imploded one says the detector stalled or quantized.
                    parts.append(
                        f'{wh.fraction_abstained:.0%} not judged '
                        f'(ends {wh.fraction_abstained_edge:.0%}, '
                        f'no scale {wh.fraction_abstained_implosion:.0%})')
            except Exception:  # noqa: BLE001 — optional stat omitted
                pass
        base = prefix + '   ·   '.join(parts)
        self._dispatch_adf(tid, base)

    def _dispatch_adf(self, tid: str, base: str) -> None:
        """Run the ADF stationarity test in the background and append its verdict
        to the stats line `base`. Superseded (epoch) if the view changes first.
        A verdict is memoized in `_diag_cache` like the stats beside it, so a redraw
        of the same trace writes it at once and submits nothing; a failure is not
        stored, so the next redraw retries."""
        controller = self.controller
        # The trace id alone keys the verdict: this call passes no significance, so every
        # stored verdict is the stored trace's at the controller's default one. A
        # user-settable significance would have to join this key.
        key = (tid, 'adf')

        def verdict_text(adf) -> str:
            return (f"{'stationary' if adf.is_stationary else 'non-stationary'} "
                    f"(ADF p = {adf.p_value:.2g})")

        if key in self._diag_cache:
            self.trace_stats.setText(base + '   ·   ' + verdict_text(self._diag_cache[key]))
            return

        def done(adf, token) -> None:
            self._diag_cache[key] = adf      # valid even if the view has moved on
            if not self._adf_slot.current(token):
                return
            self.trace_stats.setText(base + '   ·   ' + verdict_text(adf))

        def fail(_exc, token) -> None:
            if not self._adf_slot.current(token):
                return
            self.trace_stats.setText(base + '   ·   ADF n/a')

        # A fit (or another ADF) holding the worker queues the press: the fast line
        # shows now and the ADF retries once the worker frees, unless the view changed
        # by then. The dispatch reads the view's epoch and never moves it.
        if self._adf_slot.submit(
                lambda: controller.trace_stationarity(tid), done, fail,
                target=(self._adf_slot.epoch, tid, base), queue=True,
                notice=lambda: self.trace_stats.setText(base + '   ·   ADF: pending…'),
                description='trace stationarity (ADF)', bump=False):
            self.trace_stats.setText(base + '   ·   ADF: computing…')

    @property
    def _trace_epoch(self) -> int:
        """The trace view's staleness epoch, held by the ADF test's slot."""
        return self._adf_slot.epoch

    def _update_diag(self) -> None:
        self.diag_ax.clear()
        tid = self._current_trace_id()
        if tid is None:
            self.diag_canvas.draw_idle()
            return
        # Dispatched on the view KEY through an explicit table. The previous
        # prefix-match on the combo's TEXT had an `else` catching everything that
        # was not the histogram, so a view added to the combo would silently have
        # drawn the block variance instead of failing.
        view = self._current_diag_view()
        try:
            plotter = _DIAG_PLOTTERS[view]
            plotter(self._focus_view(tid, view), ax=self.diag_ax)
        except Exception as exc:  # noqa: BLE001 — surfaced on the axes
            self.diag_ax.text(0.5, 0.5, str(exc), ha='center', va='center',
                              fontsize=8, wrap=True, transform=self.diag_ax.transAxes)
        self.diag_fig.tight_layout()
        self.diag_canvas.draw_idle()

    def load_traces_dialog(self) -> None:
        """Load one or more count-rate traces. Instrument-agnostic: each file is
        auto-detected (ALV, Brookhaven), falling back to the plain two-column generic
        parser with a units prompt. Public so the sidebar's
        Traces node can trigger it too."""
        if runner().is_busy:                 # add mutates the workspace store
            busy_notice(self)
            return
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, 'Load count-rate trace(s)', open_start_dir(self.controller),
            'Trace files (*.ASC *.asc *.csv *.txt);;All files (*.*)')
        if not paths:
            return
        remember_open_dir(self.controller, paths[0])
        report = load_trace_files(self.controller, paths, detect=self._autodetect_trace,
                                  generic_fallback=self._load_generic_trace)
        last, unreadable = report.last_trace_id, report.unreadable
        parse_notes, malformed = report.parse_notes, report.malformed
        # The two reports are separate on purpose: see _finish_load's docstring in
        # gui/main_window.py. This tab does not go through _finish_load (different tab,
        # different dialog), so it renders its own -- the wording is deliberately parallel.
        if malformed:
            QtWidgets.QMessageBox.warning(
                self, 'Some traces could not be read',
                f'{len(malformed)} file(s) were recognised as count-rate traces, but '
                f'contain data the program cannot use. Nothing from them was loaded.'
                f'\n\n  ' + '\n\n  '.join(malformed)
                + '\n\nFix the file at the position named above, re-export it, or re-load it with the settings the message suggests.')
        if unreadable:
            files = '\n  '.join(unreadable)
            QtWidgets.QMessageBox.warning(
                self, 'Some traces not loaded',
                f'Could not read as a count-rate trace:\n  {files}')
        # After the "not loaded" warning, as in _finish_load: actionable first, then
        # informational. This loader does not go through _finish_load (different tab,
        # different dialog), so it calls the shared renderer directly.
        show_parse_notes(self, parse_notes)
        if last is not None:
            self.refresh_traces(select_id=last)
            self.workspaceChanged.emit()         # sidebar shows the new trace node(s)

    def _autodetect_trace(self, path: str):
        """Try each instrument trace parser. Returns a `Detection`, or None when a parser
        crashed unexpectedly and the user has already been told (skip this path).

        The detection itself is `app/loading.detect_trace`, the shared loop in
        `parsers/autodetect.py` over the trace roster. It used to be a hand-copy of the
        main window's, which is how one copy learned a new rule and the other did not,
        leaving a recognised trace file with a bad column reported as "unrecognised
        format". This method survives only for the Qt arm below, which the loader must
        not have; the loader calls it back for each file.
        """
        try:
            return detect_trace(path)
        except Exception as exc:  # noqa: BLE001 — dialog catch, a real bug surfaced; not a ParseError -- a real bug
            QtWidgets.QMessageBox.critical(self, 'Could not load trace', str(exc))
            return None

    def _load_generic_trace(self, path: str):
        """Parse a plain two-column trace and prompt for its units. Returns the
        previews, [] if not a numeric table, or None if the user canceled."""
        try:
            previews = GenericTraceParser().parse(path)
        except (ParseError, FileNotFoundError):
            # Measured: GenericTraceParser.parse() builds its preview from _raw_ columns
            # only, so __post_init__'s require_finite_data never sees one and no
            # MalformedDataError is reachable HERE. It surfaces later, at build() --
            # which is why `app/loading.load_trace_files` guards the controller call. The
            # same one-line note its DLS sibling carries, so the next sweep does not have
            # to re-derive it (and does not conclude this route is simply unguarded).
            return []
        if not previews:
            return []
        units = self._ask_trace_units()
        if units is None:
            return None
        for p in previews:
            p.time_unit, p.count_rate_unit = units
        return previews

    def _ask_trace_units(self):
        """Prompt for the time + count-rate units of a generic (plain-text) trace.
        Returns (time_unit, count_rate_unit) or None if canceled."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('Trace units')
        form = QtWidgets.QFormLayout(dlg)
        t_combo = QtWidgets.QComboBox()
        t_combo.addItems(['s', 'ms', 'min', 'h'])
        c_combo = QtWidgets.QComboBox()
        c_combo.addItems(['cps', 'kcps', 'Mcps'])
        form.addRow('Time unit', t_combo)
        form.addRow('Count-rate unit', c_combo)
        box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(dlg.accept)
        box.rejected.connect(dlg.reject)
        form.addRow(box)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            return t_combo.currentText(), c_combo.currentText()
        return None


    def remove_traces(self, tids) -> None:
        """Remove the given traces (used by the sidebar's trace context menu)."""
        if runner().is_busy:
            busy_notice(self)
            return
        for tid in tids:
            self.controller.remove_trace(tid)
        self.refresh_traces()
        self.workspaceChanged.emit()

    def focus_trace(self, tid: str) -> None:
        """Select+focus a trace by id (used when the sidebar trace node is clicked)."""
        self.inner.setCurrentIndex(0)            # raise the Traces sub-tab
        self.refresh_traces(select_id=tid)

    # ----------------------------------------------- synthetic-data tab ---
    def _build_synth_tab(self) -> QtWidgets.QWidget:
        """Unified synthetic-data generator (not sample-scoped). One set of inputs;
        each artifact can be Previewed (raw generated data), Saved to a loadable
        instrument file, and/or added to the workspace as an analysable sample.

        DLS and SLS need different physical inputs: the size populations drive the
        correlograms (Rh), while the SLS sample block carries Mw/Rg/A2. They share
        the optical conditions and the angle/concentration grid.
        """
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)

        # ---- left: inputs (scrollable; the panel is tall), Generate pinned under
        # them (the §3 column template), so the inputs' length never hides it
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setMinimumWidth(380)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(inner)
        scroll.setWidget(inner)
        column = QtWidgets.QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(scroll, 1)
        h.addLayout(column, 0)

        # Conditions (shared by DLS + SLS)
        cond_box = QtWidgets.QGroupBox('Conditions (Shared)')
        add_help_to_groupbox(cond_box, 'Generate synthetic data from the inputs below.',
                             bullets=[
            'Produces DLS correlograms (from the size populations) and SLS Zimm sets '
            '(from Mw / Rg / A₂), from the program’s own forward model.',
            'These <b>Conditions</b> (wavelength, n, temperature, viscosity) are shared '
            'by both.',
            'Each artifact can be Previewed, Saved to a loadable file, or added to the '
            'workspace.',
        ])
        cform = QtWidgets.QFormLayout(cond_box)
        self.syn_cond = {}
        for key, label, default in _SYNTH_CONDITIONS:
            e = QtWidgets.QLineEdit(default)
            self.syn_cond[key] = e
            cform.addRow(f'{label}:', e)
        # Temperature + viscosity with selectable units (default °C, mPa·s).
        # Stored in human units; converted to canonical (K, Pa·s) on read.
        self.syn_temp = QtWidgets.QLineEdit('25')
        self.syn_temp_unit = QtWidgets.QComboBox()
        self.syn_temp_unit.addItems(U.unit_options('temperature'))   # °C default
        cform.addRow('Temperature:', _value_unit_row(self.syn_temp, self.syn_temp_unit))
        self.syn_visc = QtWidgets.QLineEdit('0.89')
        self.syn_visc_unit = QtWidgets.QComboBox()
        self.syn_visc_unit.addItems(U.unit_options('viscosity'))     # mPa·s default
        cform.addRow('Viscosity:', _value_unit_row(self.syn_visc, self.syn_visc_unit))
        left.addWidget(cond_box)

        # DLS sizes: populations + correlogram controls
        dls_box = QtWidgets.QGroupBox('DLS Sizes')
        add_help_to_groupbox(dls_box, 'Sets the correlogram: the sizes in the sample '
                             'and what the correlator does to them.', bullets=[
            'One population row per mode; <b>Weight</b> is read by the <b>Weight '
            'basis</b>, <b>Spread CV</b> is the lognormal width (0 = a single '
            'exponential).',
            '<b>Lag axis</b>: <i>Log</i> reads <b>Points</b>; <i>Multi-τ</i> reads '
            'Δt₀, <b>Channels/octave</b> and <b>Blocks</b>.',
            '<b>Noise level</b> sets the first channel’s SD; the shot-noise '
            '<b>Noise model</b> also reads <b>Mean count rate</b>.',
            '<b>Channel noise correlation</b> and <b>Baseline offset</b> each add one '
            'imperfection of a real correlator; 0 turns it off.',
            'Recover what you put in by analyzing the generated data in the DLS tab; '
            'the ground truth is printed under the preview.',
            'Every control in full: User-Manual §11.3.1.',
        ])
        dv = QtWidgets.QVBoxLayout(dls_box)
        dv.addWidget(QtWidgets.QLabel(
            'Size populations (R<sub>h</sub> drives the correlogram)'))
        self.pop_table = QtWidgets.QTableWidget(1, 3)
        self.pop_table.setHorizontalHeaderLabels(['Rₕ (nm)', 'Weight', 'Spread CV'])
        self.pop_table.horizontalHeader().setStretchLastSection(True)
        self.pop_table.setMaximumHeight(130)
        for col, val in enumerate(('30', '1.0', '0')):
            self.pop_table.setItem(0, col, QtWidgets.QTableWidgetItem(val))
        dv.addWidget(self.pop_table)
        # Weight basis: how the 'Weight' column is interpreted. 'Intensity' (default) is the
        # scattering amplitude the correlogram uses directly; Number/Volume/Surface treat
        # the weight as a COMPOSITION and forward-model the intensity (Rayleigh R^6, angle-
        # independent) so the generated set round-trips through the DLS weighting reweighter.
        brow = QtWidgets.QHBoxLayout(); brow.setContentsMargins(0, 0, 0, 0)
        brow.addWidget(QtWidgets.QLabel('Weight basis:'))
        self.syn_weight_basis = QtWidgets.QComboBox()
        for lbl, key in (('Intensity', 'intensity'), ('Number', 'number'),
                         ('Volume', 'volume'), ('Surface', 'surface')):
            self.syn_weight_basis.addItem(lbl, key)
        self.syn_weight_basis.setToolTip(
            'How the Weight column is read.\n'
            '• Intensity (default): the scattering amplitude, used directly.\n'
            '• Number / Volume / Surface: the weight is a composition; the intensity is '
            'forward-modeled with the chosen scattering model so the generated correlogram’s '
            'intensity distribution reweights back to this composition — exercises the '
            'DLS Distribution weighting selector from known ground truth.')
        self.syn_weight_basis.currentIndexChanged.connect(self._on_weight_basis_changed)
        brow.addWidget(self.syn_weight_basis)
        # Scattering model for a non-intensity basis (Rayleigh R⁶ / RDG / full Mie). RDG and
        # Mie carry the angular form factor and need the contrast m = solute/solvent RI.
        self.syn_comp_model = QtWidgets.QComboBox()
        for lbl, key in (('Rayleigh point', 'rayleigh'), ('RDG sphere', 'rdg'),
                         ('Mie sphere', 'mie')):
            self.syn_comp_model.addItem(lbl, key)
        self.syn_comp_model.setToolTip(
            'Scattering model used to forward-model intensity from a Number/Volume/Surface '
            'composition. Rayleigh (R⁶) for very small particles; RDG for low contrast; Mie '
            'for the general high-contrast/large case (e.g. latex in water). RDG and Mie use '
            'the solute refractive index in the Solute n field.')
        self.syn_comp_model.currentIndexChanged.connect(self._on_weight_basis_changed)
        self.syn_comp_model_label = QtWidgets.QLabel('Model:')
        brow.addWidget(self.syn_comp_model_label)
        brow.addWidget(self.syn_comp_model)
        self.syn_comp_solute_ri_label = QtWidgets.QLabel('Solute n:')
        self.syn_comp_solute_ri = QtWidgets.QLineEdit('1.585')   # polystyrene default text
        self.syn_comp_solute_ri.setMaximumWidth(70)
        self.syn_comp_solute_ri.setToolTip(
            'Solute (particle) refractive index, for the RDG/Mie composition forward model '
            '(contrast m = solute/solvent). Only used for a non-intensity basis with RDG/Mie.')
        brow.addWidget(self.syn_comp_solute_ri_label)
        brow.addWidget(self.syn_comp_solute_ri)
        brow.addStretch(1)
        dv.addLayout(brow)
        self._on_weight_basis_changed()
        prow = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton('Add')
        add.clicked.connect(self._synth_add_row)
        rem = QtWidgets.QPushButton('Remove')
        rem.clicked.connect(self._synth_remove_row)
        prow.addWidget(add)
        prow.addWidget(rem)
        prow.addStretch(1)
        dv.addLayout(prow)
        dform = QtWidgets.QFormLayout()
        self.syn_dls = {}
        for key, label, default in _SYNTH_DLS:
            e = QtWidgets.QLineEdit(default)
            self.syn_dls[key] = e
            dform.addRow(f'{label}:', e)
        self.syn_dls['baseline'].setToolTip(
            'A flat offset added to g₂−1 at every lag, the residual a correlator '
            'leaves when its measured baseline misses its calculated one; 0 is a '
            'perfect baseline, which no instrument gives. It reaches the Correlogram, '
            'Multi-angle DLS and Depolarized (VV/VH) artifacts; what the Nonlinear '
            'cumulant’s floating baseline does with it is '
            'Theory-and-Equations-Guide §3.2.5.')
        # Lag axis + noise model. Every real instrument emits a multi-tau staircase, so
        # these two selectors decide whether the generated data is as hard to invert as
        # a measurement or systematically easier.
        self.syn_lag_axis = QtWidgets.QComboBox()
        for lbl, key in (('Log (uniform)', 'log'), ('Multi-τ (real correlator)', 'multitau')):
            self.syn_lag_axis.addItem(lbl, key)
        self.syn_lag_axis.setToolTip(
            'Shape of the delay-time axis.\n'
            '• Log (uniform): evenly log-spaced, set by the Points field above.\n'
            '• Multi-τ: the staircase real correlators emit, the sample time doubling '
            'every L channels, set by the fields below.\n'
            'It reaches the Correlogram, Multi-angle, Two-mode and Mode-resolved '
            'artifacts; the other DLS artifacts keep a uniform grid on Points '
            '(Theory-and-Equations-Guide §7.1).')
        self.syn_lag_axis.currentIndexChanged.connect(self._on_lag_axis_changed)
        dform.addRow('Lag axis:', self.syn_lag_axis)
        # The per-channel models offered here are the exact-in-Γ Biganzoli–Ferri pair; the
        # generator's Schätzel closed forms stay API-only (owner-ruled 2026-09-11, this
        # comment its normative home).
        self.syn_noise_model = QtWidgets.QComboBox()
        for lbl, key in (('Constant', 'constant'),
                         ('Per-channel (Biganzoli–Ferri)', 'biganzoli_classical'),
                         ('Per-channel + shot noise (Biganzoli–Ferri, full)',
                          'biganzoli_full')):
            self.syn_noise_model.addItem(lbl, key)
        self.syn_noise_model.setToolTip(
            'How the noise SD varies across channels.\n'
            '• Constant: the Noise level on every channel, which no correlator gives.\n'
            '• Per-channel: the Biganzoli–Ferri variance’s classical limit; the Noise '
            'level sets the first channel’s SD.\n'
            '• Per-channel + shot noise: the same plus photon-counting terms at the '
            'Mean count rate; a real file’s header rate sets those terms.\n'
            'Theory-and-Equations-Guide §7.2.')
        self.syn_noise_model.currentIndexChanged.connect(self._on_lag_axis_changed)
        dform.addRow('Noise model:', self.syn_noise_model)
        self.syn_noise = {}
        for key, label, default in _SYNTH_NOISE:
            e = QtWidgets.QLineEdit(default)
            self.syn_noise[key] = e
            dform.addRow(f'{label}:', e)
        self.syn_noise['mean_count_rate_kcps'].setToolTip(
            'Mean photon count rate in kcps, the value a file header states; read only '
            'by Per-channel + shot noise, where a lower rate widens the short-lag noise. '
            'For Multi-angle and Two-mode DLS it is the rate at 90°: each angle’s rate '
            'follows 1/sin θ and its trace’s mean matches; a Mode-resolved set draws at '
            'its own static intensities (Theory-and-Equations-Guide §7.2).')
        self.syn_noise['noise_correlation'].setToolTip(
            'Lag-1 correlation between neighboring channels’ noise, −1 to 1, under '
            'every noise model. 0 draws each channel independently, which no correlator '
            'does; real correlograms sit near +0.3. It reproduces this one statistic, '
            'not the instrument’s covariance (Theory-and-Equations-Guide §7.3).')
        self.syn_multitau = {}
        for key, label, default in _SYNTH_MULTITAU:
            e = QtWidgets.QLineEdit(default)
            self.syn_multitau[key] = e
            dform.addRow(f'{label}:', e)
        self.syn_multitau['multitau_dt0_us'].setToolTip(
            'Sample time of the finest block, in microseconds (an ALV’s is 0.025 µs). '
            'Every later block doubles it.')
        self.syn_multitau['multitau_channels_per_octave'].setToolTip(
            'L — channels before the sample time doubles. Real hardware uses 8 or 16.')
        self.syn_multitau['multitau_n_blocks'].setToolTip(
            'How many octaves the axis spans. 22 blocks from 0.025 µs reaches ~0.8 s.')
        dv.addLayout(dform)
        self._on_lag_axis_changed()
        left.addWidget(dls_box)

        # SLS sample (Mw / Rg / A2 / dn-dc)
        sls_box = QtWidgets.QGroupBox('SLS Sample (Thermodynamic)')
        add_help_to_groupbox(sls_box,
                             'Ground-truth values for the synthetic SLS Zimm set.',
                             bullets=[
            '<b>Mw</b>, <b>Rg</b> and <b>A₂</b> are the thermodynamic values the '
            'Zimm / Berry analysis should recover from the generated set — compare '
            'the fit against them.',
            '<b>dn/dc</b> is used only to build the intensities (the optical '
            'contrast); the generator never proposes a dn/dc for real data, and '
            'the Data tab still needs one typed in.',
            'The angles and concentrations of the grid come from the box below.',
        ])
        sform = QtWidgets.QFormLayout(sls_box)
        self.syn_sls = {}
        for key, label, default in _SYNTH_SLS:
            e = QtWidgets.QLineEdit(default)
            self.syn_sls[key] = e
            sform.addRow(f'{label}:', e)
        self.syn_sls['mw'].setToolTip(
            'Weight-average molecular weight the Zimm / Berry fit should recover from '
            'the generated SLS set.')
        self.syn_sls['rg_nm'].setToolTip(
            'Radius of gyration the Zimm / Berry fit should recover from the '
            'generated SLS set.')
        self.syn_sls['a2'].setToolTip(
            'Second virial coefficient the Zimm / Berry fit should recover from '
            'the generated SLS set.')
        self.syn_sls['dn_dc'].setToolTip(
            'Sets the contrast the intensities are built from; not recovered by '
            'a fit.')
        left.addWidget(sls_box)

        # Count-rate trace (feeds the "Count-rate trace" artifact)
        trace_box = QtWidgets.QGroupBox('Count-Rate Trace')
        add_help_to_groupbox(trace_box, 'Inputs for the <b>Count-rate trace</b> '
                             'artifact, the ground truth for the Traces sub-tab’s '
                             'diagnostics.', bullets=[
            '<b>CV</b> is a fixed relative scatter, not shot noise: the excess the '
            'Fano factor measures.',
            '<b>Dust spikes</b> are single-sample multiplicative hits, the impulses '
            'both outlier overlays find.',
            'The <b>offset</b> trio plants a step that <i>persists</i>: much longer '
            'than the Hampel window it is absorbed into the rolling baseline with no '
            'flags; shorter than half the window it flags. Leave <b>Offset start</b> '
            'empty for none.',
            'The <b>sine</b> pair plants a periodic component of known frequency for '
            'the spectrum and the periodogram to find; above half the sampling rate '
            'it aliases, as on a real instrument.',
        ])
        tform = QtWidgets.QFormLayout(trace_box)
        self.syn_trace = {}
        for key, label, default in _SYNTH_TRACE:
            e = QtWidgets.QLineEdit(default)
            self.syn_trace[key] = e
            tform.addRow(f'{label}:', e)
        self.syn_trace['offset_amplitude'].setToolTip(
            'Height of the step as a fraction of the mean count rate, matching the '
            'Drift field above (0.5 = half again the mean).')
        self.syn_trace['spikes'].setToolTip(
            'Number of single-sample spikes, each multiplying its sample by a random '
            '2–3.5x. Planted at random indices, so two can land on one sample.')
        left.addWidget(trace_box)

        # Depolarization / shape (feeds the "Depolarized (VV/VH)" artifact)
        depol_box = QtWidgets.QGroupBox('Depolarization / Shape (DPLS)')
        add_help_to_groupbox(depol_box, 'Inputs for the <b>Depolarized (VV/VH)</b> artifact '
                             '(static Cabannes + dynamic DDLS).', bullets=[
            'ρ<sub>v</sub> sets the static VV/VH intensity pair (→ δ², Cabannes factor). '
            'Copy the shown I<sub>VV</sub>/I<sub>VH</sub> into the SLS '
            '<b>Anisotropy correction</b> check.',
            'The <b>shape</b> (rod, ellipsoid or sphere) + its dimensions set the dynamic VV/VH '
            'correlograms with a true D<sub>t</sub> and D<sub>r</sub>, over the shared angle grid.',
            'Rod uses the rigid-rod model (valid 2 &lt; L/d &lt; 30); ellipsoid uses the '
            'Perrin ellipsoid model (Theory-and-Equations-Guide §3.6 and §4.7).',
        ])
        pdform = QtWidgets.QFormLayout(depol_box)
        self.syn_depol = {}
        for key, label, default in _SYNTH_DEPOL:
            e = QtWidgets.QLineEdit(default)
            self.syn_depol[key] = e
            pdform.addRow(f'{label}:', e)
        self.syn_depol['rho_v'].setToolTip(
            'Depolarization ratio ρv = I_VH / I_VV (vertical incident). Physical range '
            '[0, 3/4); δ² diverges at 3/4 (Theory-and-Equations-Guide §4.7.2).')
        self.syn_depol['vh_noise'].setToolTip(
            'g₂−1 noise added to the depolarized (VH) channel only — VH is intrinsically '
            'weak, so it is the noisy side. The VV channel is generated clean.')
        self.syn_shape_combo = QtWidgets.QComboBox()
        # The builder's own vocabulary, not a copy of it: a shape added to
        # DDLS_SHAPES reaches this combo, and one removed from it disappears here.
        self.syn_shape_combo.addItems(list(DDLS_SHAPES))
        self.syn_shape_combo.setToolTip(
            'Rod → the rigid-rod D_t, D_r model (valid 2 < L/d < 30); ellipsoid → the '
            'Perrin ellipsoid model; sphere → the one-unknown Stokes–Einstein–Debye '
            'model, whose R from D_r and R from D_t must agree. Sets the dynamic VV/VH '
            'decay rates (Theory-and-Equations-Guide §3.6.4–3.6.6).')
        pdform.addRow('Shape:', self.syn_shape_combo)
        self.syn_dim1 = QtWidgets.QLineEdit('60')
        self.syn_dim2 = QtWidgets.QLineEdit('6')
        self._syn_dim1_lbl = QtWidgets.QLabel('Length L (nm):')
        self._syn_dim2_lbl = QtWidgets.QLabel('Diameter d (nm):')
        pdform.addRow(self._syn_dim1_lbl, self.syn_dim1)
        pdform.addRow(self._syn_dim2_lbl, self.syn_dim2)
        self.syn_shape_combo.currentTextChanged.connect(self._synth_shape_changed)
        left.addWidget(depol_box)

        # Angle / concentration grid + calibration
        grid_box = QtWidgets.QGroupBox('SLS / Multi-Angle Grid')
        add_help_to_groupbox(grid_box,
                             'The angle × concentration grid every SLS artifact '
                             'is generated on.',
                             bullets=[
            'Comma-separated <b>angles</b> (°) and <b>concentrations</b> (mg/mL); '
            'every artifact uses the same grid.',
            '<b>Calibration: Default</b> writes calibration metadata into the '
            'saved SLS file and, when the set is added to the workspace, '
            'prefills the session calibration; <b>Uncalibrated</b> leaves it '
            'out, so the injected sample runs on an arbitrary scale with Mw and '
            'A₂ flagged unreliable — the same path a real uncalibrated load '
            'takes.',
            'Untick <b>Write dn/dc into the SLS file</b> for the dn/dc-less '
            'variant (its own tooltip says what that changes).',
        ])
        gform = QtWidgets.QFormLayout(grid_box)
        self.syn_angles = QtWidgets.QLineEdit('35, 50, 65, 80, 95, 110, 125, 140')
        self.syn_concs = QtWidgets.QLineEdit('0.2, 0.4, 0.6, 1.0, 1.4')
        gform.addRow('Angles (°):', self.syn_angles)
        gform.addRow('Concentrations (mg/mL):', self.syn_concs)
        crow = QtWidgets.QHBoxLayout()
        self.syn_cal_default = QtWidgets.QRadioButton('Default')
        self.syn_cal_uncal = QtWidgets.QRadioButton('Uncalibrated')
        self.syn_cal_default.setToolTip(
            'Writes calibration metadata into the saved SLS file and, when the '
            'set is added to the workspace, prefills the session calibration.')
        self.syn_cal_uncal.setToolTip(
            'Leaves calibration metadata out, so the injected sample runs on an '
            'arbitrary scale with Mw and A₂ flagged unreliable.')
        self.syn_cal_default.setChecked(True)
        cg = QtWidgets.QButtonGroup(self)
        cg.addButton(self.syn_cal_default)
        cg.addButton(self.syn_cal_uncal)
        crow.addWidget(self.syn_cal_default)
        crow.addWidget(self.syn_cal_uncal)
        crow.addStretch(1)
        gform.addRow('Calibration:', crow)
        # The dn/dc twin of the Uncalibrated choice: the intensities are still built
        # from the dn/dc entered above (a contrast is physics), but the saved file and
        # an injected measurement carry none, so the K = 1 arbitrary-scale path is
        # reachable from synthetic ground truth (owner-ruled 2026-08-26).
        self.syn_dn_dc_in_files = QtWidgets.QCheckBox('Write dn/dc into the SLS file')
        self.syn_dn_dc_in_files.setChecked(True)
        self.syn_dn_dc_in_files.setToolTip(
            'Unticked, the saved file and an injected sample carry no dn/dc, so the SLS '
            'analysis runs on an arbitrary optical scale (K = 1): Mw and A₂ flagged '
            'unreliable while Rg and 2·A₂·Mw survive, as for a real file with no dn/dc. '
            'The intensities still use the dn/dc entered above '
            '(Theory-and-Equations-Guide §1.5.1).')
        gform.addRow('dn/dc:', self.syn_dn_dc_in_files)
        left.addWidget(grid_box)

        # What to generate (Preview / Save per artifact)
        gen_box = QtWidgets.QGroupBox('What to Generate')
        add_help_to_groupbox(gen_box, 'Pick which artifacts to build.', bullets=[
            'Tick <b>Preview</b> to plot an artifact on the right, <b>Save</b> to '
            'write it to the folder below; either one generates it.',
            'The multi-file sets (Full Mw-series, Architecture series, '
            'Mode-resolved linked set, Form-factor series) save to a subfolder.',
        ])
        gl = QtWidgets.QGridLayout(gen_box)
        gl.addWidget(QtWidgets.QLabel('Artifact'), 0, 0)
        gl.addWidget(QtWidgets.QLabel('Preview'), 0, 1)
        gl.addWidget(QtWidgets.QLabel('Save'), 0, 2)
        self.syn_preview = {}
        self.syn_save = {}
        center = QtCore.Qt.AlignmentFlag.AlignCenter
        for r, (key, label, can_prev, can_save) in enumerate(_SYNTH_ARTIFACTS, 1):
            gl.addWidget(QtWidgets.QLabel(label), r, 0)
            if can_prev:
                cb = QtWidgets.QCheckBox()
                self.syn_preview[key] = cb
                gl.addWidget(cb, r, 1, alignment=center)
            else:
                gl.addWidget(QtWidgets.QLabel('–'), r, 1, alignment=center)
            if can_save:
                cb = QtWidgets.QCheckBox()
                self.syn_save[key] = cb
                gl.addWidget(cb, r, 2, alignment=center)
        self.syn_preview['correlogram'].setChecked(True)   # a sensible default
        note = ThemedLabel(
            'Ticking Preview or Save generates that artifact; the workspace toggle '
            'below then adds the generated data too.', role='hint', size=10)
        note.setWordWrap(True)
        gl.addWidget(note, len(_SYNTH_ARTIFACTS) + 1, 0, 1, 3)
        left.addWidget(gen_box)

        # Output: save folder + workspace toggle + seed + Generate
        out_box = QtWidgets.QGroupBox('Output')
        add_help_to_groupbox(out_box, 'Where the generated data goes.', bullets=[
            '<b>Save to</b>: the folder for saved artifacts.',
            '<b>Add to workspace as sample</b> injects the generated '
            'measurements under the name given, so you can analyze them like '
            'loaded data.',
            'The <b>Seed</b> makes a run reproducible; blank draws a fresh one.',
        ])
        ov = QtWidgets.QVBoxLayout(out_box)
        frow = QtWidgets.QHBoxLayout()
        frow.addWidget(QtWidgets.QLabel('Save to:'))
        self.syn_folder = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton('Browse…')
        browse.clicked.connect(self._synth_browse_folder)
        frow.addWidget(self.syn_folder, 1)
        frow.addWidget(browse)
        ov.addLayout(frow)
        wrow = QtWidgets.QHBoxLayout()
        self.syn_ws_check = QtWidgets.QCheckBox('Add to workspace as sample:')
        self.syn_ws_name = QtWidgets.QLineEdit('PEG 1M / water')
        wrow.addWidget(self.syn_ws_check)
        wrow.addWidget(self.syn_ws_name, 1)
        ov.addLayout(wrow)
        srow = QtWidgets.QHBoxLayout()
        srow.addWidget(QtWidgets.QLabel('Seed (blank = random):'))
        self.syn_seed = QtWidgets.QLineEdit('1')
        self.syn_seed.setMaximumWidth(90)
        srow.addWidget(self.syn_seed)
        srow.addStretch(1)
        self.syn_gen_button = QtWidgets.QPushButton('Generate')
        self.syn_gen_button.setToolTip(
            'Builds (and optionally saves/adds) in the background — the window '
            'stays usable and the preview appears when it finishes. One task at a time.')
        self.syn_gen_button.clicked.connect(self._synth_generate)
        column.addWidget(self.syn_gen_button)
        ov.addLayout(srow)
        left.addWidget(out_box)
        left.addStretch(1)
        # β / noise / points keep their in-tab default text (the synthetic-generator
        # defaults are session fields here, not Settings).

        # ---- right: preview ----
        right = QtWidgets.QVBoxLayout()
        h.addLayout(right, 1)
        shrow = QtWidgets.QHBoxLayout()
        shrow.addWidget(QtWidgets.QLabel('Showing:'))
        self.syn_show_combo = QtWidgets.QComboBox()
        self.syn_show_combo.currentIndexChanged.connect(self._synth_show_changed)
        shrow.addWidget(self.syn_show_combo, 1)
        right.addLayout(shrow)
        self.syn_fig = Figure(figsize=(4.3, 3.2))
        self.syn_canvas = make_canvas_expanding(LazyCanvas(self.syn_fig))
        self.syn_ax = self.syn_fig.add_subplot(111)
        right.addWidget(themed_navtoolbar(self.syn_canvas, w, self.controller))
        set_plot_export_name(self.syn_canvas, 'synthetic_preview')
        right.addWidget(self.syn_canvas, 1)
        self.syn_truth = ThemedLabel('Choose artifacts and Generate.',
                                     role='muted', size=11)
        self.syn_truth.setWordWrap(True)
        right.addWidget(self.syn_truth)

        self._syn_built = {}        # key -> built artifact object
        return w

    def _on_weight_basis_changed(self) -> None:
        """Show the composition scattering-model row only for a non-intensity basis, and the
        solute-RI field only for RDG/Mie (which need the contrast m)."""
        non_intensity = self.syn_weight_basis.currentData() != 'intensity'
        for w in (self.syn_comp_model_label, self.syn_comp_model):
            w.setVisible(non_intensity)
        needs_ri = non_intensity and self.syn_comp_model.currentData() in ('rdg', 'mie')
        for w in (self.syn_comp_solute_ri_label, self.syn_comp_solute_ri):
            w.setVisible(needs_ri)

    def _synth_axis_kw(self) -> dict:
        """The lag-axis / noise-model keywords for the generator, read from the fields.

        Only what the current selection uses is sent, so an untouched panel produces
        exactly the call it always did (no `grid=` and no `noise_model=`), and the
        generator's own defaults stay the single source for them.
        """
        kw: dict = {}
        if self.syn_lag_axis.currentData() == 'multitau':
            kw['grid'] = 'multitau'
            kw['multitau_dt0_s'] = (
                float(self.syn_multitau['multitau_dt0_us'].text()) * 1.0e-6)
            kw['multitau_channels_per_octave'] = int(float(
                self.syn_multitau['multitau_channels_per_octave'].text()))
            kw['multitau_n_blocks'] = int(float(
                self.syn_multitau['multitau_n_blocks'].text()))
        model = self.syn_noise_model.currentData()
        if model != 'constant':
            kw['noise_model'] = model
        if model == 'biganzoli_full':
            # kcps in the field, cps in the engine -- the same unit seam as the trace
            # loaders, converted here so the generator never sees a display unit.
            kw['mean_count_rate_cps'] = (
                float(self.syn_noise['mean_count_rate_kcps'].text()) * 1.0e3)
        rho = float(self.syn_noise['noise_correlation'].text() or 0.0)
        if rho != 0.0:
            kw['noise_correlation'] = rho
        return kw

    def _on_lag_axis_changed(self) -> None:
        """Enable the multi-tau axis fields when the multi-tau axis is selected.

        Named for the lag axis alone, which is all it branches on: the noise model is
        orthogonal to the axis and enables nothing of its own. The noise-model combo
        stays connected so a future model needing its own inputs has the slot already
        wired, and so the two selectors cannot drift out of sync.

        Disabled rather than hidden: the fields document what a multi-tau axis is made
        of even when the uniform grid is selected, and the box stops changing height as
        the user switches.

        `Points` deliberately stays ENABLED on a multi-tau axis. It is ignored by the
        four artifacts that take the lag axis (Correlogram, Multi-angle DLS, Two-mode
        DLS, the Mode-resolved linked set) but is still read by every other DLS-emitting
        artifact — Depolarized (VV/VH), the form-factor series, the full set — which
        build their own uniform grid. Greying
        it out while those artifacts consume it was the state a reviewer caught: a
        disabled control that still drives output is worse than a live one whose scope
        the help explains.
        """
        multitau = self.syn_lag_axis.currentData() == 'multitau'
        for e in self.syn_multitau.values():
            e.setEnabled(multitau)
        # The count rate is read by the full model alone, so it follows that
        # selection (same idiom); the correlation applies to every model and stays live.
        self.syn_noise['mean_count_rate_kcps'].setEnabled(
            self.syn_noise_model.currentData() == 'biganzoli_full')

    def _synth_add_row(self) -> None:
        r = self.pop_table.rowCount()
        self.pop_table.insertRow(r)
        for col, val in enumerate(('200', '1.0', '0')):
            self.pop_table.setItem(r, col, QtWidgets.QTableWidgetItem(val))

    def _synth_remove_row(self) -> None:
        r = self.pop_table.currentRow()
        if r >= 0 and self.pop_table.rowCount() > 1:
            self.pop_table.removeRow(r)

    def _synth_shape_changed(self, shape: str) -> None:
        """Relabel the two dimension fields (and reset to that shape's canonical defaults)
        when the DPLS shape switches between rod, ellipsoid and sphere.

        A sphere has ONE dimension, so the second field is disabled rather than hidden --
        the same shown-but-disabled idiom the multi-tau fields in this panel already use
        for a control that does not apply to the current selection."""
        if shape == 'ellipsoid':
            self._syn_dim1_lbl.setText('Polar semi-axis a (nm):')
            self._syn_dim2_lbl.setText('Equatorial radius b (nm):')
            self.syn_dim1.setText('30')
            self.syn_dim2.setText('10')
        elif shape == 'sphere':
            self._syn_dim1_lbl.setText('Radius R (nm):')
            self._syn_dim2_lbl.setText('(not used for a sphere):')
            self.syn_dim1.setText('40')
            self.syn_dim2.setText('40')
        else:                                             # rod
            self._syn_dim1_lbl.setText('Length L (nm):')
            self._syn_dim2_lbl.setText('Diameter d (nm):')
            self.syn_dim1.setText('60')
            self.syn_dim2.setText('6')
        one_dimension = shape == 'sphere'
        self.syn_dim2.setEnabled(not one_dimension)
        self._syn_dim2_lbl.setEnabled(not one_dimension)

    def _read_pop_specs(self) -> List[dict]:
        specs = []
        for r in range(self.pop_table.rowCount()):
            def cell(c, r=r):
                it = self.pop_table.item(r, c)
                return it.text().strip() if it is not None else ''
            rh, weight, cv = cell(0), cell(1), cell(2)
            if not rh and not weight:
                continue
            specs.append({'rh_nm': float(rh), 'weight': float(weight),
                          'spread_cv': float(cv or 0.0)})
        return specs

    # ---- input helpers ----
    @staticmethod
    def _synth_floats(text: str) -> List[float]:
        vals = [float(tok) for tok in text.replace(',', ' ').split()]
        # float() accepts 'inf'/'nan', which sail past the callers' `> 0` filters and
        # overflow deep in the generator (round 333f f12, corrected decline 2026-09-01):
        # refuse them here, at the one parse site, so the error surface names the input.
        for v in vals:
            if not math.isfinite(v):
                raise ValueError(f'non-finite value in a numeric field: {v!r}')
        return vals

    def _synth_identity(self):
        """Parse the workspace sample-name field into (polymer, solvent)."""
        raw = self.syn_ws_name.text().strip() or 'sample'
        if '/' in raw:
            poly, solv = [p.strip() for p in raw.split('/', 1)]
        else:
            poly, solv = raw, 'solvent'
        return (poly or 'sample'), (solv or 'solvent')

    def _synth_browse_folder(self) -> None:
        # Seed-only: the field's own text wins; a blank field starts in the shared
        # export folder. Not written back — the chosen folder is a programmatic
        # output workspace, not a save-dialog destination.
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Choose output folder',
            self.syn_folder.text() or export_start_dir(self.controller))
        if d:
            self.syn_folder.setText(d)

    def _synth_temperature_K(self) -> float:
        """The generator's temperature, read in the chosen unit → canonical K."""
        return U.to_canonical('temperature', float(self.syn_temp.text()),
                              self.syn_temp_unit.currentText())

    def _synth_viscosity_Pa_s(self) -> float:
        """The generator's viscosity, read in the chosen unit → canonical Pa·s."""
        return U.to_canonical('viscosity', float(self.syn_visc.text()),
                              self.syn_visc_unit.currentText())

    def _synth_trace_kw(self) -> dict:
        """The count-rate trace fields, parsed for `build_count_rate_trace`.

        `spikes` is a count and `offset_start_s` is optional (blank = no offset);
        everything else is a plain float. Raised errors land in the same "Check
        your inputs" dialog as the rest of the generator's fields, so a typo here
        reads the same as a typo anywhere else in the panel.
        """
        text = {k: e.text().strip() for k, e in self.syn_trace.items()}
        kw = {'spikes': int(float(text['spikes'] or 0))}
        for key in ('duration_s', 'dt_s', 'mean_kHz', 'trace_cv', 'drift',
                    'offset_duration_s', 'offset_amplitude',
                    'sine_frequency_hz', 'sine_amplitude'):
            kw[key] = float(text[key] or 0.0)
        kw['offset_start_s'] = float(text['offset_start_s']) if text['offset_start_s'] else None
        return kw

    # ---- generate / build / save / inject ----
    def _synth_generate(self) -> None:
        try:
            cond = {k: float(e.text()) for k, e in self.syn_cond.items()}
            poly, solv = self._synth_identity()
            request = synthetic.SynthRequest(
                wl=cond['wavelength_nm'], n=cond['solvent_refractive_index'],
                T=self._synth_temperature_K(), eta=self._synth_viscosity_Pa_s(),
                angle=float(self.syn_dls['angle_deg'].text()),
                beta=float(self.syn_dls['beta'].text()),
                noise=float(self.syn_dls['noise_level'].text()),
                baseline=float(self.syn_dls['baseline'].text() or 0.0),
                npts=int(float(self.syn_dls['n_points'].text())),
                axis_kw=self._synth_axis_kw(),
                mw=float(self.syn_sls['mw'].text()),
                rg=float(self.syn_sls['rg_nm'].text()),
                a2=float(self.syn_sls['a2'].text()),
                dn_dc=float(self.syn_sls['dn_dc'].text()),
                angles=self._synth_floats(self.syn_angles.text()),
                concs=[c * 1e-3 for c in self._synth_floats(self.syn_concs.text()) if c > 0],
                with_calibration=self.syn_cal_default.isChecked(),
                dn_dc_in_files=self.syn_dn_dc_in_files.isChecked(),
                seed=(int(self.syn_seed.text().strip()) if self.syn_seed.text().strip() else None),
                rho_v=float(self.syn_depol['rho_v'].text()),
                vh_noise=float(self.syn_depol['vh_noise'].text()),
                shape=self.syn_shape_combo.currentText(),
                # A sphere is built from ONE dimension: the second field is disabled, so
                # its text is whatever the previous shape left there and must not be
                # read. The builder refuses two unequal entries, so mirroring here is
                # what keeps the disabled field from raising on the user's behalf.
                dims_nm=((float(self.syn_dim1.text()), float(self.syn_dim1.text()))
                         if self.syn_shape_combo.currentText() == 'sphere'
                         else (float(self.syn_dim1.text()), float(self.syn_dim2.text()))),
                weight_basis=self.syn_weight_basis.currentData(),
                comp_model=self.syn_comp_model.currentData(),
                comp_solute_ri=(float(self.syn_comp_solute_ri.text())
                                if self.syn_comp_solute_ri.text().strip() else None),
                trace_kw=self._synth_trace_kw(),
                specs=self._read_pop_specs(),
                polymer=poly, solvent=solv,
            )
        except Exception as exc:  # noqa: BLE001 — dialog catch
            QtWidgets.QMessageBox.warning(self, 'Check your inputs', str(exc))
            return

        active = [k for (k, _l, _p, _s) in _SYNTH_ARTIFACTS
                  if (k in self.syn_preview and self.syn_preview[k].isChecked())
                  or (k in self.syn_save and self.syn_save[k].isChecked())]
        if not active:
            QtWidgets.QMessageBox.information(
                self, 'Nothing selected',
                'Tick Preview or Save for at least one artifact.')
            return

        # Everything below runs on the worker (building a full multi-angle set or
        # a full test set is the slow part). Hoist every widget read here first;
        # the thunk touches only the controller + plain values.
        save_keys = [k for k in active
                     if k in self.syn_save and self.syn_save[k].isChecked()]
        folder = self.syn_folder.text().strip()
        eta_cp = self._synth_viscosity_Pa_s() * 1e3
        do_inject = self.syn_ws_check.isChecked()
        inj_conc0 = request.concs[0] if request.concs else 0.6 * 1e-3
        inj_T = self._synth_temperature_K()
        inj_eta = self._synth_viscosity_Pa_s()
        preview_keys = [k for k in active
                        if k in self.syn_preview and self.syn_preview[k].isChecked()]

        def work():
            # Build (the slow part) + save (file I/O) run on the worker. The
            # inject is deliberately NOT here: it mutates workspace.measurements +
            # regroups, structural changes the main thread iterates — so it runs
            # in done() on the main thread instead.
            built = {}
            for key in active:
                if key == 'full_set':
                    continue            # written directly at save time
                built[key] = synthetic.build(self.controller, key, request)
            out = {'built': built}
            try:
                out['saved'] = synthetic.save(self.controller, save_keys, built, folder, eta_cp)
            except Exception as exc:                         # noqa: BLE001 — carried to done() as save_exc
                out['save_exc'] = exc
            return out

        def done(out, _token) -> None:
            built = out['built']
            self._syn_built = built
            if 'save_exc' in out:
                QtWidgets.QMessageBox.warning(self, 'Save failed', str(out['save_exc']))
            saved = out.get('saved', [])
            injected = ''
            if do_inject:                          # workspace mutation: main thread
                parts = []
                n_before = len(self.controller.workspace.measurements)
                try:
                    single = synthetic.inject(
                        self.controller, built, request.with_calibration, poly, solv,
                        inj_conc0, inj_T, inj_eta)
                    if single:
                        parts.append(single)
                    # Architecture is a multi-sample batch — it injects a whole Linear +
                    # Branched series (each tagged Chain-type), NOT the single active
                    # sample, so it runs its own path rather than through synthetic.inject.
                    if 'architecture' in active:
                        inj = self.controller.inject_architecture_series(solvent_name=solv)
                        parts.append(f'architecture series ({len(inj)} samples, '
                                     f'Linear/Branched tagged)')
                    # Form-factor series is likewise a multi-sample SLS+DLS batch (the
                    # uncalibrated sample is files-only, so it is not injected).
                    if 'form_factor' in active:
                        inj = self.controller.inject_form_factor_series(solvent_name=solv)
                        parts.append(f'form-factor series ({len(inj)} samples, '
                                     f'SLS + paired DLS)')
                except Exception as exc:           # noqa: BLE001 — dialog catch; earlier injects survive
                    QtWidgets.QMessageBox.warning(
                        self, 'Add to workspace failed', str(exc))
                    # Do NOT discard `parts`: an earlier inject may already have mutated the
                    # workspace, and its evidence must survive so the sidebar still refreshes.
                injected = '; '.join(parts)
                # Refresh whenever anything was actually added — even if a LATER inject then
                # raised — so the sidebar/Data view never desyncs from the workspace. The
                # measurement-count check also catches a partial batch that added samples
                # before raising (its report string is absent, but the rebuild must still run).
                if injected or len(self.controller.workspace.measurements) != n_before:
                    self.workspaceChanged.emit()   # sidebar/Data rebuild
                    self.refresh_traces()          # pick up any injected trace

            labels = {k: l for (k, l, _p, _s) in _SYNTH_ARTIFACTS}
            self.syn_show_combo.blockSignals(True)
            self.syn_show_combo.clear()
            for key in preview_keys:
                if key in built:
                    self.syn_show_combo.addItem(labels[key], key)
            self.syn_show_combo.blockSignals(False)
            self._synth_show_changed()

            extra = []
            if saved:
                extra.append(f'saved {len(saved)} file(s)')
            if injected:
                extra.append(injected)
            if extra:
                self.syn_truth.setText(self.syn_truth.text() + '   —   ' + '; '.join(extra))

        def fail(exc: BaseException, _token) -> None:
            QtWidgets.QMessageBox.warning(self, 'Generate failed', str(exc))

        # A busy press is refused with the status-bar notice (`busy_notice`).
        self._generate_slot.submit(work, done, fail, target=None, queue=False,
                                   busy_widgets=(self.syn_gen_button,),
                                   notice=lambda: busy_notice(self),
                                   description='synthetic data generation', paint=False)

    # ---- preview ----
    def _synth_show_changed(self) -> None:
        self.syn_ax.clear()
        key = self.syn_show_combo.currentData()
        obj = self._syn_built.get(key) if key else None
        if obj is None:
            self.syn_ax.text(0.5, 0.5, 'Nothing ticked for preview', ha='center',
                             va='center', color='#999', transform=self.syn_ax.transAxes)
            self.syn_canvas.draw_idle()
            if self.syn_show_combo.count() == 0:
                self.syn_truth.setText('Generated. (Tick Preview to see an artifact here.)')
            return
        try:
            self._synth_plot(key, obj)
        except Exception as exc:  # noqa: BLE001 — surfaced on the axes
            self.syn_ax.text(0.5, 0.5, str(exc), ha='center', va='center', fontsize=8,
                             wrap=True, transform=self.syn_ax.transAxes)
        self.syn_fig.tight_layout()
        self.syn_canvas.draw_idle()
        self.syn_truth.setText(self._synth_truth(key, obj))

    def _synth_plot(self, key, obj) -> None:
        if key == 'correlogram':
            plot_synthetic_correlogram(obj, ax=self.syn_ax)
        elif key == 'trace':
            plot_intensity_trace(obj, mode='absolute', ax=self.syn_ax)
        elif key == 'multi_dls':
            plot_synthetic_multi_dls(obj, ax=self.syn_ax)
        elif key == 'two_mode':
            plot_synthetic_multi_dls(obj.data, ax=self.syn_ax)
        elif key == 'depolarized':
            self._synth_plot_depolarized(obj)
        elif key == 'architecture':
            self._synth_plot_architecture(obj)
        elif key == 'form_factor':
            self._synth_plot_form_factor(obj)
        else:       # sls_*
            plot_synthetic_sls_set(obj, ax=self.syn_ax)

    def _synth_plot_depolarized(self, art) -> None:
        """VV vs VH correlograms at the mid angle: VH decays faster (rotation)."""
        ax = self.syn_ax
        vv, vh = art.ddls.vv, art.ddls.vh
        angles = vv.angles_deg
        a0 = angles[len(angles) // 2]                    # a mid angle
        ax.semilogx(vv.delay_times_s * 1e6, vv.signals[a0], '-', color='#0072B2',
                    label='VV (polarized)')
        ax.semilogx(vh.delay_times_s * 1e6, vh.signals[a0], '-', color='#D55E00',
                    label='VH (depolarized)')
        ax.set_xlabel('lag time (µs)')
        ax.set_ylabel('g₂(τ) − 1')
        ax.set_title(f'Depolarized correlograms at {a0:g}° (VH decays faster: rotation)')
        ax.legend()

    def _synth_plot_architecture(self, series: dict) -> None:
        """Rg–Mw of the series: linear reference vs branched (contracted) samples."""
        ax = self.syn_ax
        lin = [(p['mw'], p['rg']) for p in series.values() if p['role'] == 'linear']
        bra = [(p['mw'], p['rg']) for p in series.values() if p['role'] == 'branched']
        if lin:
            ax.loglog([m for m, _ in lin], [r for _, r in lin], 'o-', color='#0072B2',
                      label='linear reference')
        if bra:
            ax.loglog([m for m, _ in bra], [r for _, r in bra], 's', color='#D55E00',
                      label='branched (contracted)')
        ax.set_xlabel('M_w (g/mol)')
        ax.set_ylabel('R_g (nm)')
        ax.set_title('Architecture series: branched R_g sits below the linear line')
        ax.legend()

    def _synth_plot_form_factor(self, series: dict) -> None:
        """Rg vs Rh of the panel with the ρ = 1 diagonal: each shape sits on its own ρ ray,
        so the shape-verdict spread (sphere ρ<1 below, coil/rod ρ>1 above) is visible."""
        ax = self.syn_ax
        rg = [p['rg'] for p in series.values()]
        rh = [p['rh'] for p in series.values()]
        ax.scatter(rh, rg, c='#0072B2', zorder=3)
        for name, p in series.items():
            ax.annotate(name.replace('FF ', ''), (p['rh'], p['rg']),
                        fontsize=7, xytext=(3, 3), textcoords='offset points')
        lo = min(min(rg), min(rh)) * 0.8
        hi = max(max(rg), max(rh)) * 1.2
        ax.plot([lo, hi], [lo, hi], '--', color='#999999', label='ρ = 1')
        ax.set_xlabel('R_h (nm)')
        ax.set_ylabel('R_g (nm)')
        ax.set_title('Form-factor panel: shape sets ρ = R_g/R_h (above the line = coil/rod)')
        ax.legend()

    def _synth_truth(self, key, obj) -> str:
        if key == 'correlogram':
            return (f'Ground truth: R<sub>h,eff</sub> = {obj.rh_eff_nm:.1f} nm, '
                    f'PDI = {obj.pdi:.3f}. An ideal cumulant fit should recover these.')
        if key == 'trace':
            cr = np.asarray(obj.count_rates_cps, dtype=float)
            return (f'Synthetic trace: {cr.size} points, mean {cr.mean():,.0f} cps. '
                    'Load it in the Traces tab to test the diagnostics.')
        if key == 'multi_dls':
            ang = obj.angles_deg
            return (f'Multi-angle DLS: {len(ang)} angles ({min(ang):g}–{max(ang):g}°). '
                    'Load the .ASC into the DLS tab to test Γ-vs-q².')
        if key == 'two_mode':
            ang = obj.data.angles_deg
            cross = (f'{obj.crossing_angle_deg:g}°'
                     if np.isfinite(obj.crossing_angle_deg)
                     else 'outside the angle range')
            return (f'Two-mode DLS: fast R<sub>h</sub> = {obj.rh_fast_nm:g} nm '
                    f'(diffusive) + a q-independent slow mode, amplitudes '
                    f'{obj.a_fast_fraction:g}/{obj.a_slow_fraction:g}, over '
                    f'{len(ang)} angles ({min(ang):g}–{max(ang):g}°). The modes '
                    f'CROSS at {cross} — load the .ASC and use the DLS tab\'s '
                    'Mode Map to separate them (rank-based pairing fails at the '
                    'crossing by design).')
        if key == 'depolarized':
            pair, ddls = obj.pair, obj.ddls
            dims = ('L = {:g}, d = {:g} nm'.format(*ddls.dims_nm) if ddls.shape == 'rod'
                    else 'a = {:g}, b = {:g} nm'.format(*ddls.dims_nm))
            return (f'Static pair — I<sub>VV</sub> = {pair.i_vv:,.0f}, '
                    f'I<sub>VH</sub> = {pair.i_vh:,.0f} → ρ<sub>v</sub> = {pair.rho_v:.3f}, '
                    f'δ² = {pair.optical_anisotropy_sq:.3f}, Cabannes f = '
                    f'{pair.cabannes_isotropic_factor:.3f} '
                    '(copy I<sub>VV</sub>/I<sub>VH</sub> into the SLS anisotropy panel).  '
                    f'Dynamic — {ddls.shape} ({dims}), D<sub>r</sub> = {ddls.d_r_rad2_s:.3g} '
                    f'rad²/s over {len(ddls.vv.angles_deg)} angles; load the VV/VH .ASC, tag '
                    'each VV/VH in the Data tab, and run DDLS.')
        if key == 'architecture':
            n_lin = sum(1 for p in obj.values() if p['role'] == 'linear')
            n_bra = sum(1 for p in obj.values() if p['role'] == 'branched')
            gvals = ', '.join(f'{name.replace("PS ", "")} g={p["g"]:.3f}'
                              for name, p in obj.items() if p['g'] is not None)
            return (f'Architecture series: {n_lin} linear reference + {n_bra} branched '
                    f'({gvals}). Save or add to workspace — linear samples are tagged '
                    '<b>Linear</b> and branched <b>Branched</b>, so the Cross-Sample '
                    'Architecture tab recovers g / f_eff / n.')
        if key == 'form_factor':
            n = len(obj)
            n_degen = sum(1 for p in obj.values() if p['degenerate'])
            return (f'Form-factor series: {n} samples (SLS P(q) + paired DLS), each with a '
                    f'known ρ = R<sub>g</sub>/R<sub>h</sub> shape verdict. Save or add to '
                    f'workspace, then fit the Form factor P(q) for R<sub>g</sub> and pick it on '
                    f'Cross-Sample: ρ recovers sphere / soft-sphere / coil / rod, {n_degen} fits '
                    'are degenerate (ellipsoid Rg-divergent + core–shell Rg-agree consensus), and '
                    'the sphere ladder scales as ν ≈ 1/3 (model-dependent). The uncalibrated '
                    'sample is saved to file only (calibration is session-global).')
        mw = self.syn_sls['mw'].text()
        rg = self.syn_sls['rg_nm'].text()
        a2 = self.syn_sls['a2'].text()
        cal = ('calibrated' if obj.calibrated
               else 'uncalibrated (arbitrary M<sub>w</sub> scale)')
        if not getattr(obj, 'dn_dc_in_files', True):
            cal += ' · d<i>n</i>/d<i>c</i> withheld from the file'
        return (f'Ground truth: M<sub>w</sub> = {mw} g/mol, R<sub>g</sub> = {rg} nm, A₂ = {a2}. '
                f'{len(obj.concentrations_g_per_mL)} concentrations · '
                f'{len(obj.angles_deg)} angle(s) · {cal}.')

    # ------------------------------------------------------------- updates ---
    def reseed_from_settings(self) -> None:
        """Re-render after a Settings change (e.g. plot palette). The trace + synthetic
        defaults are session-only in-tab fields now, so there
        is nothing to re-seed from SettingsState here.

        NB: this must NOT cascade into the nested Solvent Explorer — default_solvent
        is a build-time seed ("seed, never override"), so an unrelated Settings Apply
        must not yank the user's active solvent/condition. The Explorer seeds itself
        once in its __init__ and re-themes via its own changeEvent."""
        self._update_trace()

    def set_measurement(self, item_id: Optional[str]) -> None:
        """Follow the shell's active sample: always adopt the focused
        measurement's sample; an incompatible one shows a named empty-state, not the old
        keep-last."""
        sid = self.controller.sample_id_of(item_id) if item_id is not None else None
        self.show_active_sample(sid)

    def show_active_sample(self, sid: Optional[str]) -> None:
        """Point I·sin θ at the active sample `sid`: mirror the dropdown to it when it can
        plot, else show a neutral placeholder there; `_update_isin` renders the plot or the
        named empty-state."""
        self.isin_selector.refresh()
        self.sample_id = sid
        if sid is not None and self.isin_selector.has_sample(sid):
            self.isin_selector.set_current_sample_id(sid)
        elif sid is not None:
            self.isin_selector.show_incompatible('(active sample has no c = 0 curve)')
        self._update_isin()
        self.selectionChanged.emit()          # repaint the sidebar mirror

    @QtCore.Slot(str)
    def _on_isin_sample(self, sid: str) -> None:
        """The user picked a sample in the I·sinθ dropdown → ask the shell to make it the
        single active sample (it fans back to `show_active_sample` + the sidebar)."""
        self.sampleFocusRequested.emit(sid or '')

    def selected_item_ids(self) -> list:
        """The measurements of the I·sinθ sample (sidebar-mirror contract). Traces are a
        separate store and are not mirrored here. An INCOMPATIBLE active sample (showing
        the named empty-state, not eligible in the selector) mirrors nothing — the tab
        isn't plotting it, matching the SLS tab's behaviour (focused-review)."""
        if self.sample_id is None or not self.isin_selector.has_sample(self.sample_id):
            return []
        s = self.controller.workspace.samples.get(self.sample_id)
        if s is None:
            return []
        # The c = 0 curves the plot draws (`run_i_sin_theta`: the reference slot plus
        # every extra blank) -- never the c > 0 rows it ignores (rule 10, round 1 f7).
        ids = list(s.extra_solvent_reference_item_ids)
        if s.solvent_reference_item_id:
            ids.insert(0, s.solvent_reference_item_id)
        return ids

    def _update_isin(self) -> None:
        # run_i_sin_theta writes the shared controller.results dict, so it must
        # not run while a background fit is writing it too. Defer
        # the whole refresh until the worker frees rather than race it — and dedup
        # (a single pending flag) so a burst of abs/rel toggles doesn't queue N
        # identical redraws that all fire on drain.
        if runner().is_busy:
            if not self._isin_refresh_pending:
                self._isin_refresh_pending = True
                run_when_idle(self._flush_isin)
            return
        self._isin_refresh_pending = False
        self.isin_ax.clear()
        if self.sample_id is None:
            self.isin_note.setText(
                'Pick a sample in the Workspace list (or the dropdown above).')
            self.isin_canvas.draw_idle()
            return
        # An active sample that can't produce I·sin θ (no SLS / solvent reference): named
        # empty-state, not a caught run error. `has_sample` is False for it
        # (only the neutral placeholder is selected in the dropdown).
        if not self.isin_selector.has_sample(self.sample_id):
            s = self.controller.workspace.samples.get(self.sample_id)
            label = sample_display_name(s) if s is not None else self.sample_id
            self.isin_note.setText(
                f'No c = 0 (solvent-reference) curve for {label} — I·sin θ plots only '
                'c = 0 intensities. Load or assign one, or pick another sample.')
            self.isin_canvas.draw_idle()
            return
        mode = 'normalized' if self.isin_rel.isChecked() else 'absolute'
        try:
            res = self.controller.run_i_sin_theta(self.sample_id, mode=mode)
        except Exception as exc:  # noqa: BLE001 — surfaced in the note
            self.isin_note.setText(str(exc))
            self.isin_canvas.draw_idle()
            return
        set_plot_export_name(
            self.isin_canvas,
            sample_suggested_name(self.controller, self.sample_id, 'i_sin_theta'))
        plot_i_sin_theta(res, ax=self.isin_ax)
        self.isin_fig.tight_layout()
        self.isin_canvas.draw_idle()
        self.isin_note.setText(
            f'{len(res.curves)} curve(s). Flat across angle = clean isotropic '
            'scattering; curvature or asymmetry about 90° flags alignment, stray '
            'light, or dust.')

    def _flush_isin(self) -> None:
        """Deferred I·sinθ refresh once the worker frees (re-checks busy, so a
        job started mid-drain pushes it to the next completion)."""
        self._isin_refresh_pending = False
        self._update_isin()
