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
    measurements, with an Absolute / Relative (normalised) scale toggle. For an
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

import os
from typing import List, Optional

import numpy as np
from PySide6 import QtCore, QtWidgets

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from app import units as U
from gui.plot_controls import (
    make_split_panels, make_canvas_expanding, make_vertical_plot_stack,
)
from gui.solvent_explorer_module import SolventExplorerModule
from gui.theme import ThemedLabel
from gui.widgets import roomy_tabs, SampleSelector, value_unit_row as _value_unit_row


def _sample_label(sample) -> str:
    """The 'polymer / solvent @ T K' sample name the sidebar shows (not the raw
    sample_id key)."""
    poly, solv, temp = sample.polymer_name, sample.solvent_name, sample.temperature_K
    if poly and solv and temp is not None and temp == temp:      # temp==temp: not NaN
        return f'{poly} / {solv} @ {temp:g} K'
    return f'{poly or "?"} / {solv or "?"}'
from gui.worker import BUSY_NOTICE, busy_notice, run_when_idle, runner

from plotting.plots import (
    plot_i_sin_theta, plot_synthetic_correlogram, plot_intensity_trace,
    plot_count_rate_histogram, plot_block_variance,
    plot_synthetic_multi_dls, plot_synthetic_sls_set,
    display_factor as _plot_factor, display_unit as _plot_unit,
)
from parsers.base_parser import ParseError
from parsers.alv_asc import ALVTraceParser
from parsers.brookhaven_dls import BrookhavenTraceParser
from parsers.generic_trace import GenericTraceParser

_OUTLIER_COLOUR = '#D55E00'   # Okabe-Ito vermilion, for the GUI-owned flag overlay
_TRACE_CYCLE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00',
                '#56B4E9', '#F0E442', '#000000']   # Okabe-Ito, for overlays

# Instrument trace parsers tried (in order) by the auto-detection on load. Each
# sniffs its own format strictly (ALV .ASC structure; Brookhaven HH:MM:SS
# timestamps) and raises ParseError otherwise, so the first that succeeds is the
# right one. The lenient generic two-column parser is the explicit fallback (it
# matches almost anything, so it must not be in this list) and prompts for units.
_TRACE_PARSERS = [ALVTraceParser, BrookhavenTraceParser]

# Default seeds for the intensity-trace controls. These used to live in Settings;
# per feedback 2026-06-26 #6 they are now plain in-tab fields seeded each launch
# from these fixed code constants (not persisted).
_DEFAULT_OUTLIER_K = 3.0
_DEFAULT_RUNAVG_WINDOW = 0     # 0 → auto (≈ n/20 points)


# Synthetic-data generator field groups: (key, label, default).
# Conditions are shared by DLS + SLS; the size populations drive DLS Rh, while the
# SLS sample block carries Mw/Rg/A2 (a different set of physical parameters).
# Temperature + viscosity are entered with selectable units (feedback 2026-06-26
# #14, defaults °C and mPa·s) and handled apart; only the unitless conditions go
# through this simple list.
_SYNTH_CONDITIONS = [
    ('wavelength_nm', 'Wavelength (nm)', '532'),
    ('solvent_refractive_index', 'n (solvent)', '1.33'),
]
_SYNTH_DLS = [
    ('angle_deg', 'Angle, single (deg)', '90'),
    ('beta', 'β (coherence)', '0.8'),
    ('noise_level', 'Noise level', '0.0'),
    ('n_points', 'Points', '200'),
]
_SYNTH_SLS = [
    ('mw', 'Mw (g/mol)', '1e6'),
    ('rg_nm', 'Rg (nm)', '55'),
    ('a2', 'A₂ (mol·mL/g²)', '9.5e-5'),
    ('dn_dc', 'dn/dc (mL/g)', '0.135'),
]
# Artifacts the generator can produce: (key, label, can_preview, can_save).
# "Full Mw-series test set" is save-only (it is many files) and not injectable.
_SYNTH_ARTIFACTS = [
    ('correlogram', 'Correlogram (one angle)', True, True),
    ('trace', 'Count-rate trace', True, True),
    ('multi_dls', 'Multi-angle DLS (all angles)', True, True),
    ('sls_zimm', 'SLS — full Zimm set', True, True),
    ('sls_single_conc', 'SLS — single concentration (angular)', True, True),
    ('sls_single_angle', 'SLS — single angle (conc. series)', True, True),
    ('full_set', 'Full Mw-series test set', False, True),
]
# Default save filenames per artifact (written into the chosen folder).
_SYNTH_FILENAMES = {
    'correlogram': 'Synthetic correlogram.csv',
    'trace': 'Synthetic trace (kcps vs s).csv',
    'multi_dls': 'Synthetic DLS multi-angle.ASC',
    'sls_zimm': 'Synthetic SLS Zimm.csv',
    'sls_single_conc': 'Synthetic SLS single-concentration.csv',
    'sls_single_angle': 'Synthetic SLS single-angle.csv',
}


class UtilitiesModule(QtWidgets.QWidget):
    """Sample-scoped data-quality diagnostics and tools."""

    # Emitted after the synthetic generator injects data into the workspace, so the
    # shell can rebuild the sidebar/Data tab (the utilities module has no reference
    # to the main window).
    workspaceChanged = QtCore.Signal()
    selectionChanged = QtCore.Signal()   # emitted when the I·sinθ sample changes (mirror)

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.sample_id: Optional[str] = None
        # Staleness token for the backgrounded ADF stationarity test (the sole
        # slow call on the trace path); bumped whenever the trace view refreshes.
        self._adf_epoch = 0
        self._isin_refresh_pending = False   # dedup deferred I·sinθ refreshes
        self._build_ui()
        self.set_measurement(None)

    # ------------------------------------------------------------------ UI ---
    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        self.inner = roomy_tabs(QtWidgets.QTabWidget())   # roomier tabs so labels don't clip (#3)
        self.inner.setMovable(True)              # drag to reorder sub-tabs (A4)
        outer.addWidget(self.inner)
        self.inner.addTab(self._build_traces_tab(), 'Traces')
        self.inner.addTab(self._build_isin_tab(), 'I·sin θ')
        self.inner.addTab(self._build_synth_tab(), 'Synthetic generator')
        # The Solvent Explorer is a self-contained global calculator; it seeds its
        # own solvent in its __init__ and ignores the sample selection by design.
        self.solvent_explorer = SolventExplorerModule(self.controller)
        self.inner.addTab(self.solvent_explorer, 'Solvent Explorer')
        self.refresh_traces()

    def _build_isin_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        # The tab owns its sample: pick it here rather than inheriting the sidebar
        # focus (the sidebar only navigates). Only SLS-bearing samples can produce an
        # I·sin θ plot (it needs the c = 0 angular intensities).
        self.isin_selector = SampleSelector(
            self.controller,
            predicate=lambda s: s.has_sls or bool(s.solvent_reference_item_id),
            label_fn=_sample_label, title='Sample',
            help_text='Choose which sample to plot I·sin θ for.',
            help_bullets=['Only samples with angular intensity data (SLS / a solvent '
                          'reference) appear.',
                          'I·sin θ uses the c = 0 (solvent-reference) curves.'])
        self.isin_selector.sampleChanged.connect(self._on_isin_sample)
        v.addWidget(self.isin_selector)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Scale:'))
        self.isin_abs = QtWidgets.QRadioButton('Absolute')
        self.isin_rel = QtWidgets.QRadioButton('Relative (normalised)')
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
        self.isin_canvas = make_canvas_expanding(FigureCanvas(self.isin_fig))
        self.isin_ax = self.isin_fig.add_subplot(111)
        v.addWidget(NavigationToolbar(self.isin_canvas, w))
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

        left.addWidget(QtWidgets.QLabel('Traces to plot'))
        # A multi-select checklist (Select all/none), matching the DLS correlogram
        # tab's measurement selector (feedback 2026-06-26 #4): tick several traces to
        # overlay them. The focused trace drives the stats line + secondary diagnostic.
        self.trace_list = QtWidgets.QListWidget()
        self.trace_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.trace_list.itemSelectionChanged.connect(self._update_trace)
        self.trace_list.currentItemChanged.connect(lambda *_: self._update_trace())
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
        brow = QtWidgets.QHBoxLayout()
        load = QtWidgets.QPushButton('Load trace…')
        load.clicked.connect(self.load_traces_dialog)
        rem = QtWidgets.QPushButton('Remove')
        rem.clicked.connect(self._remove_trace)
        brow.addWidget(load)
        brow.addWidget(rem)
        left.addLayout(brow)

        # scale toggle
        trow = QtWidgets.QHBoxLayout()
        trow.addWidget(QtWidgets.QLabel('Scale:'))
        self.trace_abs = QtWidgets.QRadioButton('Absolute')
        self.trace_rel = QtWidgets.QRadioButton('Relative (baseline-normalised)')
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
        # 'k' tooltip (feedback 2026-06-26 #11): k is the shot-noise band multiplier.
        _k_help = ('k — the shot-noise band multiplier. A point is flagged when its '
                   'count rate lies outside mean ± k·√mean (√mean is the Poisson '
                   'standard deviation for raw photon counts).')
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
        self.ov_running = QtWidgets.QCheckBox('Running average')
        self.ov_running.toggled.connect(self._update_trace)
        orow.addWidget(self.ov_running)
        # User-adjustable running-average window (feedback 2026-06-26 #12). The
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
        right.addLayout(orow)

        # The main trace plot and the secondary diagnostic are independent plots
        # (count rate vs time; histogram / block variance) with different x-axes, so
        # they go in a draggable vertical splitter — each resizable on its own
        # (feedback 2026-06-29 #9). (A fit + its residual, which must stay aligned,
        # share one canvas instead — see the DLS tabs.)
        top = QtWidgets.QWidget()
        tlay = QtWidgets.QVBoxLayout(top); tlay.setContentsMargins(0, 0, 0, 0)
        self.trace_fig = Figure(figsize=(5.2, 3.4))
        self.trace_canvas = make_canvas_expanding(FigureCanvas(self.trace_fig))
        self.trace_ax = self.trace_fig.add_subplot(111)
        tlay.addWidget(NavigationToolbar(self.trace_canvas, w))
        tlay.addWidget(self.trace_canvas, 1)
        self.trace_stats = ThemedLabel(
            'Load a count-rate trace (ALV .ASC, or a two-column CSV).',
            role='muted', size=11)
        self.trace_stats.setWordWrap(True)
        tlay.addWidget(self.trace_stats)

        bottom = QtWidgets.QWidget()
        blay = QtWidgets.QVBoxLayout(bottom); blay.setContentsMargins(0, 0, 0, 0)
        drow = QtWidgets.QHBoxLayout()
        drow.addWidget(QtWidgets.QLabel('Diagnostic:'))
        self.diag_combo = QtWidgets.QComboBox()
        self.diag_combo.addItems(['Count-rate histogram', 'Block variance'])
        self.diag_combo.currentIndexChanged.connect(self._update_diag)
        drow.addWidget(self.diag_combo)
        drow.addStretch(1)
        blay.addLayout(drow)
        self.diag_fig = Figure(figsize=(5.2, 2.6))
        self.diag_canvas = make_canvas_expanding(FigureCanvas(self.diag_fig), 130)
        self.diag_ax = self.diag_fig.add_subplot(111)
        blay.addWidget(self.diag_canvas, 1)

        right.addWidget(make_vertical_plot_stack(
            [top, bottom], sizes=[360, 220], min_heights=[160, 120]), 1)
        return w

    def refresh_traces(self, select_id: Optional[str] = None) -> None:
        """Rebuild the trace list from the workspace (call after a load or a
        session load). Selects `select_id` if given, else keeps/clears selection."""
        self.trace_list.blockSignals(True)
        self.trace_list.clear()
        for t in self.controller.traces():
            label = t.sample_label or t.trace_id
            item = QtWidgets.QListWidgetItem(f'{label}  [{t.trace_id}]')
            item.setData(QtCore.Qt.ItemDataRole.UserRole, t.trace_id)
            self.trace_list.addItem(item)
        self.trace_list.blockSignals(False)
        if select_id is not None:
            for i in range(self.trace_list.count()):
                if self.trace_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole) == select_id:
                    self.trace_list.setCurrentRow(i)
                    break
        elif self.trace_list.count():
            self.trace_list.setCurrentRow(0)
        self._update_trace()

    def _current_trace_id(self) -> Optional[str]:
        it = self.trace_list.currentItem()
        return it.data(QtCore.Qt.ItemDataRole.UserRole) if it is not None else None

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
        """Sliding-window length in points: the user value, or auto (≈ n/20)."""
        w = int(self.runavg_window.value())
        return max(3, n_points // 20) if w == 0 else max(3, w)

    def _update_trace(self) -> None:
        # A refresh supersedes any in-flight ADF from a previous view.
        self._adf_epoch += 1
        self.trace_ax.clear()
        tids = self._selected_trace_ids()
        focus = self._current_trace_id()
        if not tids and focus is not None:
            tids = [focus]
        if focus not in tids:
            focus = tids[0] if tids else None
        if not tids:
            self.trace_stats.setText(
                'Load a count-rate trace (an instrument file or a two-column CSV) '
                'and tick it.')
            self.trace_canvas.draw_idle()
            self._update_diag()
            return
        mode = 'relative' if self.trace_rel.isChecked() else 'absolute'
        single = len(tids) == 1
        focus_stats = None
        for idx, tid in enumerate(tids):
            try:
                stats = self.controller.run_trace_statistics(tid)
                trace = self.controller.build_trace(tid)
            except Exception as exc:
                if single:
                    self.trace_stats.setText(str(exc))
                continue
            if tid == focus:
                focus_stats = stats
            if single:
                plot_intensity_trace(trace, mode=mode,
                                     baseline_cps=stats.baseline_cps, ax=self.trace_ax)
            else:
                # Overlay raw curves only (per-trace flag/running overlays are
                # single-trace detail — they'd be unreadable stacked).
                disp = self._cr_transform(mode, stats.baseline_cps)
                t = np.asarray(trace.times_s, dtype=float)
                cr = np.asarray(trace.count_rates_cps, dtype=float)
                self.trace_ax.plot(t, disp(cr), lw=0.8,
                                   color=_TRACE_CYCLE[idx % len(_TRACE_CYCLE)],
                                   label=self._trace_label(tid))
            # GUI-owned overlays only when a single trace is focused.
            if single:
                self._draw_trace_overlays(tid, trace, stats, mode)
        if not single:
            self.trace_ax.set_xlabel('Time (s)')
            self.trace_ax.set_ylabel(
                'Count rate / baseline' if mode == 'relative'
                else f'Count rate ({_plot_unit("intensity")})')
            self.trace_ax.legend(frameon=False, fontsize=8)
        self.trace_fig.tight_layout()
        self.trace_canvas.draw_idle()
        if focus is not None and focus_stats is not None:
            prefix = '' if single else f'{len(tids)} traces · focused: '
            self._set_diag_text(focus, focus_stats, prefix=prefix)
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
                self.trace_ax.plot(t[mask], disp(cr[mask]), 'o',
                                   color=_OUTLIER_COLOUR, ms=4,
                                   label=f'{fl.n_flagged} outliers (±{fl.k:g}√mean)')
                self.trace_ax.legend(frameon=False, fontsize=8)
            except Exception:
                pass
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
            except Exception:
                pass

    def _set_diag_text(self, tid: str, stats, prefix: str = '') -> None:
        """One-line summary: trace stats + Fano factor + correlation + ADF verdict.
        The fast parts render immediately; the ADF stationarity test (the one slow
        call) runs on the worker thread and fills in when it finishes, so a long
        trace never freezes the window."""
        parts = [
            f'{stats.n_points} pts/{stats.duration_s:.1f}s',
            f'mean {stats.mean_cps:,.0f} cps', f'CV {stats.cv:.3f}',
            f'baseline {stats.baseline_cps:,.0f} cps',
        ]
        try:
            parts.append(f'Fano {self.controller.trace_histogram(tid).fano_factor:.2f}')
        except Exception:
            pass
        try:
            bv = self.controller.trace_block_variance(tid)
            parts.append('correlated' if bv.correlations_detected else 'uncorrelated')
        except Exception:
            pass
        base = prefix + '   ·   '.join(parts)
        self._dispatch_adf(tid, base)

    def _dispatch_adf(self, tid: str, base: str) -> None:
        """Run the ADF stationarity test in the background and append its verdict
        to the stats line `base`. Superseded (epoch) if the view changes first."""
        epoch = self._adf_epoch
        controller = self.controller

        def verdict_text(adf) -> str:
            return (f"{'stationary' if adf.is_stationary else 'non-stationary'} "
                    f"(ADF p={adf.p_value:.3f})")

        def done(adf) -> None:
            if epoch != self._adf_epoch:
                return
            self.trace_stats.setText(base + '   ·   ' + verdict_text(adf))

        def fail(_exc) -> None:
            if epoch != self._adf_epoch:
                return
            self.trace_stats.setText(base + '   ·   ADF n/a')

        if runner().try_submit(lambda: controller.trace_stationarity(tid),
                               done, fail, description='trace stationarity (ADF)'):
            self.trace_stats.setText(base + '   ·   ADF: computing…')
        else:
            # A fit (or another ADF) holds the worker: show the fast line now and
            # retry the ADF once the worker frees, unless the view changed by then.
            self.trace_stats.setText(base + '   ·   ADF: pending…')
            run_when_idle(
                lambda: self._dispatch_adf(tid, base)
                if epoch == self._adf_epoch else None)

    def _update_diag(self) -> None:
        self.diag_ax.clear()
        tid = self._current_trace_id()
        if tid is None:
            self.diag_canvas.draw_idle()
            return
        try:
            if self.diag_combo.currentText().startswith('Count-rate'):
                plot_count_rate_histogram(self.controller.trace_histogram(tid),
                                          ax=self.diag_ax)
            else:
                plot_block_variance(self.controller.trace_block_variance(tid),
                                    ax=self.diag_ax)
        except Exception as exc:
            self.diag_ax.text(0.5, 0.5, str(exc), ha='center', va='center',
                              fontsize=8, wrap=True, transform=self.diag_ax.transAxes)
        self.diag_fig.tight_layout()
        self.diag_canvas.draw_idle()

    def load_traces_dialog(self) -> None:
        """Load one or more count-rate traces. Instrument-agnostic: each file is
        auto-detected (ALV, Brookhaven), falling back to the plain two-column generic
        parser with a units prompt (feedback 2026-06-26 #3). Public so the sidebar's
        Traces node can trigger it too (#4)."""
        if runner().is_busy:                 # add mutates the workspace store
            busy_notice(self)
            return
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, 'Load count-rate trace(s)', '',
            'Trace files (*.ASC *.asc *.csv *.txt);;All files (*.*)')
        if not paths:
            return
        last = None
        unreadable = []
        for path in paths:
            previews = self._autodetect_trace(path)
            if previews is None:                 # matched an instrument but errored
                continue
            if not previews:                     # no instrument matched → generic
                previews = self._load_generic_trace(path)
                if previews is None:             # user cancelled the units prompt
                    return
                if not previews:
                    unreadable.append(os.path.basename(path))
                    continue
            for p in previews:
                last = self.controller.add_trace_from_preview(p, source_path=path)
        if unreadable:
            files = '\n  '.join(unreadable)
            QtWidgets.QMessageBox.warning(
                self, 'Some traces not loaded',
                f'Could not read as a count-rate trace:\n  {files}')
        if last is not None:
            self.refresh_traces(select_id=last)
            self.workspaceChanged.emit()         # sidebar shows the new trace node(s)

    def _autodetect_trace(self, path: str):
        """Try each instrument trace parser; return its previews, [] if none matched,
        or None if a parser matched the format but failed (already reported)."""
        for parser_cls in _TRACE_PARSERS:
            try:
                previews = parser_cls().parse(path)
            except (ParseError, FileNotFoundError):
                continue                         # not this format; try the next
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, 'Could not load trace', str(exc))
                return None
            if previews:
                return previews
        return []

    def _load_generic_trace(self, path: str):
        """Parse a plain two-column trace and prompt for its units. Returns the
        previews, [] if not a numeric table, or None if the user cancelled."""
        try:
            previews = GenericTraceParser().parse(path)
        except (ParseError, FileNotFoundError):
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
        Returns (time_unit, count_rate_unit) or None if cancelled."""
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

    def _remove_trace(self) -> None:
        if runner().is_busy:
            busy_notice(self)
            return
        tids = self._selected_trace_ids() or (
            [self._current_trace_id()] if self._current_trace_id() else [])
        if not tids:
            return
        for tid in tids:
            self.controller.remove_trace(tid)
        self.refresh_traces()
        self.workspaceChanged.emit()             # sidebar drops the removed node(s)

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

        # ---- left: inputs (scrollable; the panel is tall) ----
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(380)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(inner)
        scroll.setWidget(inner)
        h.addWidget(scroll, 0)

        # Conditions (shared by DLS + SLS)
        cond_box = QtWidgets.QGroupBox('Conditions (shared)')
        cform = QtWidgets.QFormLayout(cond_box)
        self.syn_cond = {}
        for key, label, default in _SYNTH_CONDITIONS:
            e = QtWidgets.QLineEdit(default)
            self.syn_cond[key] = e
            cform.addRow(label, e)
        # Temperature + viscosity with selectable units (feedback #14: default °C,
        # mPa·s). Stored in human units; converted to canonical (K, Pa·s) on read.
        self.syn_temp = QtWidgets.QLineEdit('25')
        self.syn_temp_unit = QtWidgets.QComboBox()
        self.syn_temp_unit.addItems(U.unit_options('temperature'))   # °C default
        cform.addRow('Temperature', _value_unit_row(self.syn_temp, self.syn_temp_unit))
        self.syn_visc = QtWidgets.QLineEdit('0.89')
        self.syn_visc_unit = QtWidgets.QComboBox()
        self.syn_visc_unit.addItems(U.unit_options('viscosity'))     # mPa·s default
        cform.addRow('Viscosity', _value_unit_row(self.syn_visc, self.syn_visc_unit))
        left.addWidget(cond_box)

        # DLS sizes: populations + correlogram controls
        dls_box = QtWidgets.QGroupBox('DLS sizes')
        dv = QtWidgets.QVBoxLayout(dls_box)
        dv.addWidget(QtWidgets.QLabel('Size populations (Rh drives the correlogram)'))
        self.pop_table = QtWidgets.QTableWidget(1, 3)
        self.pop_table.setHorizontalHeaderLabels(['Rh (nm)', 'Weight', 'Spread CV'])
        self.pop_table.horizontalHeader().setStretchLastSection(True)
        self.pop_table.setMaximumHeight(130)
        for col, val in enumerate(('30', '1.0', '0')):
            self.pop_table.setItem(0, col, QtWidgets.QTableWidgetItem(val))
        dv.addWidget(self.pop_table)
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
            dform.addRow(label, e)
        dv.addLayout(dform)
        left.addWidget(dls_box)

        # SLS sample (Mw / Rg / A2 / dn-dc)
        sls_box = QtWidgets.QGroupBox('SLS sample (thermodynamic)')
        sform = QtWidgets.QFormLayout(sls_box)
        self.syn_sls = {}
        for key, label, default in _SYNTH_SLS:
            e = QtWidgets.QLineEdit(default)
            self.syn_sls[key] = e
            sform.addRow(label, e)
        left.addWidget(sls_box)

        # Angle / concentration grid + calibration
        grid_box = QtWidgets.QGroupBox('SLS / multi-angle grid')
        gform = QtWidgets.QFormLayout(grid_box)
        self.syn_angles = QtWidgets.QLineEdit('35, 50, 65, 80, 95, 110, 125, 140')
        self.syn_concs = QtWidgets.QLineEdit('0.2, 0.4, 0.6, 1.0, 1.4')
        gform.addRow('Angles (deg)', self.syn_angles)
        gform.addRow('Concentrations (mg/mL)', self.syn_concs)
        crow = QtWidgets.QHBoxLayout()
        self.syn_cal_default = QtWidgets.QRadioButton('Default')
        self.syn_cal_uncal = QtWidgets.QRadioButton('Uncalibrated')
        self.syn_cal_default.setChecked(True)
        cg = QtWidgets.QButtonGroup(self)
        cg.addButton(self.syn_cal_default)
        cg.addButton(self.syn_cal_uncal)
        crow.addWidget(self.syn_cal_default)
        crow.addWidget(self.syn_cal_uncal)
        crow.addStretch(1)
        gform.addRow('Calibration', crow)
        left.addWidget(grid_box)

        # What to generate (Preview / Save per artifact)
        gen_box = QtWidgets.QGroupBox('What to generate')
        gl = QtWidgets.QGridLayout(gen_box)
        gl.addWidget(QtWidgets.QLabel('Artifact'), 0, 0)
        gl.addWidget(QtWidgets.QLabel('Preview'), 0, 1)
        gl.addWidget(QtWidgets.QLabel('Save'), 0, 2)
        self.syn_preview = {}
        self.syn_save = {}
        centre = QtCore.Qt.AlignmentFlag.AlignCenter
        for r, (key, label, can_prev, can_save) in enumerate(_SYNTH_ARTIFACTS, 1):
            gl.addWidget(QtWidgets.QLabel(label), r, 0)
            if can_prev:
                cb = QtWidgets.QCheckBox()
                self.syn_preview[key] = cb
                gl.addWidget(cb, r, 1, alignment=centre)
            else:
                gl.addWidget(QtWidgets.QLabel('–'), r, 1, alignment=centre)
            if can_save:
                cb = QtWidgets.QCheckBox()
                self.syn_save[key] = cb
                gl.addWidget(cb, r, 2, alignment=centre)
        self.syn_preview['correlogram'].setChecked(True)   # a sensible default
        note = ThemedLabel(
            'Ticking Preview or Save generates that artifact; the workspace toggle '
            'below then adds the generated data too.', role='hint', size=10)
        note.setWordWrap(True)
        gl.addWidget(note, len(_SYNTH_ARTIFACTS) + 1, 0, 1, 3)
        left.addWidget(gen_box)

        # Output: save folder + workspace toggle + seed + Generate
        out_box = QtWidgets.QGroupBox('Output')
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
        srow.addWidget(self.syn_gen_button)
        ov.addLayout(srow)
        left.addWidget(out_box)
        left.addStretch(1)
        # β / noise / points keep their in-tab default text (feedback 2026-06-26 #6:
        # the synthetic-generator defaults are session fields here, not Settings).

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
        self.syn_canvas = make_canvas_expanding(FigureCanvas(self.syn_fig))
        self.syn_ax = self.syn_fig.add_subplot(111)
        right.addWidget(NavigationToolbar(self.syn_canvas, w))
        right.addWidget(self.syn_canvas, 1)
        self.syn_truth = ThemedLabel('Choose artifacts and Generate.',
                                     role='muted', size=11)
        self.syn_truth.setWordWrap(True)
        right.addWidget(self.syn_truth)

        self._syn_built = {}        # key -> built artifact object
        return w

    def _synth_add_row(self) -> None:
        r = self.pop_table.rowCount()
        self.pop_table.insertRow(r)
        for col, val in enumerate(('200', '1.0', '0')):
            self.pop_table.setItem(r, col, QtWidgets.QTableWidgetItem(val))

    def _synth_remove_row(self) -> None:
        r = self.pop_table.currentRow()
        if r >= 0 and self.pop_table.rowCount() > 1:
            self.pop_table.removeRow(r)

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
        return [float(tok) for tok in text.replace(',', ' ').split()]

    def _synth_identity(self):
        """Parse the workspace sample-name field into (polymer, solvent)."""
        raw = self.syn_ws_name.text().strip() or 'sample'
        if '/' in raw:
            poly, solv = [p.strip() for p in raw.split('/', 1)]
        else:
            poly, solv = raw, 'solvent'
        return (poly or 'sample'), (solv or 'solvent')

    def _synth_browse_folder(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Choose output folder', self.syn_folder.text() or '')
        if d:
            self.syn_folder.setText(d)

    def _synth_temperature_K(self) -> float:
        """The generator's temperature, read in the chosen unit → canonical K (#14)."""
        return U.to_canonical('temperature', float(self.syn_temp.text()),
                              self.syn_temp_unit.currentText())

    def _synth_viscosity_Pa_s(self) -> float:
        """The generator's viscosity, read in the chosen unit → canonical Pa·s (#14)."""
        return U.to_canonical('viscosity', float(self.syn_visc.text()),
                              self.syn_visc_unit.currentText())

    # ---- generate / build / save / inject ----
    def _synth_generate(self) -> None:
        try:
            cond = {k: float(e.text()) for k, e in self.syn_cond.items()}
            ctx = dict(
                wl=cond['wavelength_nm'], n=cond['solvent_refractive_index'],
                T=self._synth_temperature_K(), eta=self._synth_viscosity_Pa_s(),
                angle=float(self.syn_dls['angle_deg'].text()),
                beta=float(self.syn_dls['beta'].text()),
                noise=float(self.syn_dls['noise_level'].text()),
                npts=int(float(self.syn_dls['n_points'].text())),
                mw=float(self.syn_sls['mw'].text()),
                rg=float(self.syn_sls['rg_nm'].text()),
                a2=float(self.syn_sls['a2'].text()),
                dn_dc=float(self.syn_sls['dn_dc'].text()),
                angles=self._synth_floats(self.syn_angles.text()),
                concs=[c * 1e-3 for c in self._synth_floats(self.syn_concs.text()) if c > 0],
                calibrated=self.syn_cal_default.isChecked(),
                seed=(int(self.syn_seed.text().strip()) if self.syn_seed.text().strip() else None),
            )
            specs = self._read_pop_specs()
            poly, solv = self._synth_identity()
        except Exception as exc:
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
        inj_concs = self._synth_floats(self.syn_concs.text())
        inj_conc0 = (inj_concs[0] if inj_concs else 0.6) * 1e-3
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
                built[key] = self._synth_build_one(key, specs, ctx, poly, solv)
            out = {'built': built}
            try:
                out['saved'] = self._synth_save(save_keys, built, folder, eta_cp)
            except Exception as exc:                         # noqa: BLE001
                out['save_exc'] = exc
            return out

        def done(out) -> None:
            built = out['built']
            self._syn_built = built
            if 'save_exc' in out:
                QtWidgets.QMessageBox.warning(self, 'Save failed', str(out['save_exc']))
            saved = out.get('saved', [])
            injected = ''
            if do_inject:                          # workspace mutation: main thread
                try:
                    injected = self._synth_inject(
                        built, ctx['calibrated'], poly, solv,
                        inj_conc0, inj_T, inj_eta)
                except Exception as exc:           # noqa: BLE001
                    QtWidgets.QMessageBox.warning(
                        self, 'Add to workspace failed', str(exc))
                    injected = ''
                if injected:
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

        def fail(exc: BaseException) -> None:
            QtWidgets.QMessageBox.warning(self, 'Generate failed', str(exc))

        if not runner().try_submit(work, done, fail,
                                   description='synthetic data generation',
                                   busy_widgets=(self.syn_gen_button,)):
            QtWidgets.QMessageBox.information(self, 'Busy', BUSY_NOTICE)

    def _synth_build_one(self, key, specs, ctx, poly, solv):
        c = self.controller
        if key == 'correlogram':
            if not specs:
                raise ValueError('Add at least one size population for the correlogram.')
            return c.synth_correlogram(
                specs, angle_deg=ctx['angle'], wavelength_nm=ctx['wl'],
                solvent_refractive_index=ctx['n'], temperature_K=ctx['T'],
                viscosity_Pa_s=ctx['eta'], beta=ctx['beta'], noise_level=ctx['noise'],
                n_points=ctx['npts'], seed=ctx['seed'])
        if key == 'trace':
            return c.synth_trace(seed=ctx['seed'], label=f'{poly} / {solv}')
        if key == 'multi_dls':
            if not specs:
                raise ValueError('Add at least one size population for the DLS set.')
            if not ctx['angles']:
                raise ValueError('Enter at least one angle.')
            conc0 = ctx['concs'][0] if ctx['concs'] else 6e-4
            return c.synth_multi_angle_dls(
                specs, angles_deg=ctx['angles'], wavelength_nm=ctx['wl'],
                solvent_refractive_index=ctx['n'], temperature_K=ctx['T'],
                viscosity_Pa_s=ctx['eta'], beta=ctx['beta'], noise_level=ctx['noise'],
                n_points=ctx['npts'], label=f'{poly} in {solv}',
                concentration_g_per_mL=conc0, seed=ctx['seed'])
        if key.startswith('sls'):
            if not ctx['angles']:
                raise ValueError('Enter at least one angle.')
            if not ctx['concs']:
                raise ValueError('Enter at least one (non-zero) concentration.')
            if key == 'sls_zimm':
                a_list, c_list, kind = ctx['angles'], ctx['concs'], 'zimm'
            elif key == 'sls_single_conc':
                a_list, c_list, kind = ctx['angles'], [ctx['concs'][0]], 'single_concentration'
            else:   # sls_single_angle: the angle nearest 90°
                a_near = min(ctx['angles'], key=lambda x: abs(x - 90.0))
                a_list, c_list, kind = [a_near], ctx['concs'], 'single_angle'
            return c.synth_sls_set(
                mw=ctx['mw'], rg_nm=ctx['rg'], a2_mol_mL_per_g2=ctx['a2'], angles_deg=a_list,
                concentrations_g_per_mL=c_list, wavelength_nm=ctx['wl'], temperature_K=ctx['T'],
                n_solvent=ctx['n'], dn_dc=ctx['dn_dc'], calibrated=ctx['calibrated'],
                noise_level=ctx['noise'], seed=ctx['seed'], polymer_name=poly,
                solvent_name=solv, label=f'{poly} in {solv}', kind=kind)
        raise ValueError(f'Unknown artifact {key!r}.')

    def _synth_save(self, save_keys, built, folder: str, eta_cp: float) -> List[str]:
        """Write the save-ticked artifacts. Takes plain values (no widget reads) so
        it can run on the worker thread; the caller hoists the folder/viscosity."""
        if not save_keys:
            return []
        if not folder:
            raise ValueError('Choose a "Save to" folder first.')
        os.makedirs(folder, exist_ok=True)
        c = self.controller
        saved = []
        for key in save_keys:
            if key == 'full_set':
                for prof in ('Synthetic Clean', 'Synthetic Messy'):
                    c.generate_full_test_set(os.path.join(folder, prof), prof)
                    saved.append(prof)
                continue
            path = os.path.join(folder, _SYNTH_FILENAMES[key])
            obj = built[key]
            if key == 'correlogram':
                c.export_synthetic(obj, path)
            elif key == 'trace':
                c.save_synth_trace(obj, path)
            elif key == 'multi_dls':
                c.save_synth_multi_angle_dls(obj, path, viscosity_cp=eta_cp)
            else:    # sls_*
                c.save_synth_sls_set(obj, path)
            saved.append(os.path.basename(path))
        return saved

    def _synth_inject(self, built, calibrated, poly, solv, conc0, T, eta) -> str:
        """Inject a coherent single sample: the primary DLS artifact (multi-angle
        preferred), the primary SLS artifact (Zimm preferred), and any trace.
        Injecting every SLS slice would pile duplicate concentrations into one
        sample, so only the primary of each kind is added.

        Takes plain values (no widget reads) so it runs on the worker; the caller
        emits workspaceChanged + refreshes the trace list on the main thread once
        this returns (mutating the workspace here is safe under the single-flight
        guard)."""
        c = self.controller
        parts = []
        if 'multi_dls' in built:
            c.inject_multi_angle_dls(built['multi_dls'], polymer_name=poly, solvent_name=solv)
            parts.append('multi-angle DLS')
        elif 'correlogram' in built:
            c.inject_correlogram(built['correlogram'], polymer_name=poly, solvent_name=solv,
                                 concentration_g_per_mL=conc0, temperature_K=T,
                                 viscosity_Pa_s=eta)
            parts.append('correlogram')
        sls_added = False
        for key in ('sls_zimm', 'sls_single_conc', 'sls_single_angle'):
            if key in built:
                c.inject_sls_set(built[key], polymer_name=poly, solvent_name=solv,
                                 prefill_calibration=calibrated)
                parts.append(key.replace('sls_', 'SLS '))
                sls_added = True
                break
        if 'trace' in built:
            c.inject_trace(built['trace'], label=f'{poly} / {solv}')
            parts.append('trace')
        if not parts:
            return ''
        cal_note = ' (session calibration set)' if (calibrated and sls_added) else ''
        return f'added to workspace as “{poly} / {solv}”: ' + ', '.join(parts) + cal_note

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
        except Exception as exc:
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
        else:       # sls_*
            plot_synthetic_sls_set(obj, ax=self.syn_ax)

    def _synth_truth(self, key, obj) -> str:
        if key == 'correlogram':
            return (f'Ground truth: Rh_eff = {obj.rh_eff_nm:.1f} nm, PDI = {obj.pdi:.3f}. '
                    'An ideal cumulant fit should recover these.')
        if key == 'trace':
            cr = np.asarray(obj.count_rates_cps, dtype=float)
            return (f'Synthetic trace: {cr.size} points, mean {cr.mean():,.0f} cps. '
                    'Load it in the Traces tab to test the diagnostics.')
        if key == 'multi_dls':
            ang = obj.angles_deg
            return (f'Multi-angle DLS: {len(ang)} angles ({min(ang):g}–{max(ang):g}°). '
                    'Load the .ASC into the DLS tab to test Γ-vs-q².')
        mw = self.syn_sls['mw'].text()
        rg = self.syn_sls['rg_nm'].text()
        a2 = self.syn_sls['a2'].text()
        cal = 'calibrated' if obj.calibrated else 'uncalibrated (arbitrary Mw scale)'
        return (f'Ground truth: Mw = {mw} g/mol, Rg = {rg} nm, A₂ = {a2}. '
                f'{len(obj.concentrations_g_per_mL)} concentrations · '
                f'{len(obj.angles_deg)} angle(s) · {cal}.')

    # ------------------------------------------------------------- updates ---
    def reseed_from_settings(self) -> None:
        """Re-render after a Settings change (e.g. plot palette). The trace + synthetic
        defaults are session-only in-tab fields now (feedback 2026-06-26 #6), so there
        is nothing to re-seed from SettingsState here.

        NB: this must NOT cascade into the nested Solvent Explorer — default_solvent
        is a build-time seed ("seed, never override"), so an unrelated Settings Apply
        must not yank the user's active solvent/condition. The Explorer seeds itself
        once in its __init__ and re-themes via its own changeEvent."""
        self._update_trace()

    def set_measurement(self, item_id: Optional[str]) -> None:
        """Sidebar focus is a SOFT seed for the I·sinθ sample: adopt the focused
        sample only if it can produce an I·sinθ plot, else keep the tab's own pick (the
        selector is the source of truth — the sidebar merely navigates)."""
        sid = self.controller.sample_id_of(item_id) if item_id is not None else None
        self.isin_selector.refresh()
        if sid is not None and self.isin_selector.has_sample(sid):
            self.isin_selector.set_current_sample_id(sid)
        self.sample_id = self.isin_selector.current_sample_id()
        self._update_isin()
        self.selectionChanged.emit()          # repaint the sidebar mirror

    @QtCore.Slot(str)
    def _on_isin_sample(self, sid: str) -> None:
        """The user picked a sample in the I·sinθ selector."""
        self.sample_id = sid or None
        self._update_isin()
        self.selectionChanged.emit()

    def selected_item_ids(self) -> list:
        """The measurements of the I·sinθ sample (sidebar-mirror contract). Traces are a
        separate store and are not mirrored here."""
        if self.sample_id is None:
            return []
        s = self.controller.workspace.samples.get(self.sample_id)
        if s is None:
            return []
        ids = list(s.dls_item_ids) + list(s.sls_item_ids)
        if s.solvent_reference_item_id:
            ids.append(s.solvent_reference_item_id)
        return ids

    def _update_isin(self) -> None:
        # run_i_sin_theta writes the shared controller.results dict, so it must
        # not run while a background fit is writing it too (invariant 4). Defer
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
                'Select a measurement in the sidebar to choose a sample.')
            self.isin_canvas.draw_idle()
            return
        mode = 'normalized' if self.isin_rel.isChecked() else 'absolute'
        try:
            res = self.controller.run_i_sin_theta(self.sample_id, mode=mode)
        except Exception as exc:
            self.isin_note.setText(str(exc))
            self.isin_canvas.draw_idle()
            return
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
