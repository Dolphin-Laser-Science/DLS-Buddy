"""
gui/solvent_explorer_module.py
==============================

The Solvent Explorer (a **Utilities sub-tab**): a global (no-sample), display-only
calculator over the solvent-property library (`physics/solvents.py`). It is the
standalone front-end for the estimate-tier solvents (which are never auto-filled into
a measurement) and a visual companion to the Data-tab autofill.

Its purpose is twofold:
  1. give the user the refractive index n and viscosity η at a chosen solvent +
     temperature + wavelength (with the library's display-only uncertainty), and
  2. let them see **at a glance the range of conditions the library can serve with
     confidence** — the validity box and the confidence band are the point of the view.

One shared figure, two side-by-side plots (core, not deferrable):
  * **Left — vs temperature.** η on the host/left axis (the quantity that varies
     strongly with T; log-capable, log by default) and RI on the twinx right axis.
     Each curve is drawn only across its OWN temperature box (n and η boxes can
     differ), carries a confidence band, and gets a user-selection dot at the chosen
     temperature.
  * **Right — vs wavelength.** RI against wavelength at the selected temperature,
     across the RI λ box, with its band + a selection dot. (η has no λ dependence.)

The confidence band is the engine's **per-condition conservative bound**:
σ_n(λ,T) absolute for n, σ_η,rel(T)·η for η — tightest near each source's reference
conditions, growing toward the validity-box edges, honestly FLAT where a source
supplies no shape (bulk-grade viscosities). Every part of it derives from real
quantities (stated source accuracy, fit-residual envelopes, dn/dT-slope
propagation) — never a fabricated model. It is **shown, never
propagated** into any analysis SE. All physics lives in the controller/engine; this
widget only drives the controller and draws the bare arrays it returns.

Global by design even though it lives under the sample-scoped Utilities tab: it
ignores the sidebar selection entirely, and nothing is ever written into a
measurement. It seeds its own solvent once in ``__init__`` ("seed, never override")
and is deliberately left out of Utilities' ``reseed_from_settings`` path.
"""

from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from analysis.uncertainty import format_pm
from app import units as U
from physics import solvents as solvents_lib
from gui.help import section_header
from gui.plot_controls import AxisControlBar, make_canvas_expanding, themed_navtoolbar
from gui.theme import ThemedLabel, token as theme_token
from gui.widgets import value_unit_row as _value_unit_row


# Curve hues (plots render on a white background — plot-background theming is deferred
# app-wide — so fixed Okabe-Ito colors read on both themes). Deliberately NOT red:
# red is the reserved error token (§5), so a confidence band must never be red.
_RI_COLOR = '#0072B2'     # blue
_ETA_COLOR = '#009E73'    # bluish green
_BAND_ALPHA = 0.18


class SolventExplorerModule(QtWidgets.QWidget):
    """Global, display-only solvent-property calculator with confidence-band plots."""

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._build_ui()
        self.reseed_from_settings()      # seed the solvent from default_solvent + draw

    # ------------------------------------------------------------------ UI ---
    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.addWidget(section_header(
            'Solvent Explorer — look up n and η for a solvent + temperature + wavelength',
            'A display-only calculator over the solvent-property library:',
            bullets=[
                'All library solvents appear here (primary + estimate). '
                '“(estimate)” marks a lower-confidence solvent.',
                'Values never enter an analysis — this view only shows them '
                '(use the Data tab to auto-fill a measurement).',
                'Each curve is drawn only across the range the library covers; '
                'the shaded band is a conservative bound at each condition '
                '(narrowest where the sources are strongest).',
                'The band is <b>shown, never propagated</b> into any reported ± '
                '(see Theory-and-Equations-Guide §5).',
            ]))

        # ---- top: form (left) + numeric readout (right) ----
        top = QtWidgets.QHBoxLayout()
        form_box = QtWidgets.QGroupBox('Condition')
        form = QtWidgets.QFormLayout(form_box)
        self.solvent_combo = QtWidgets.QComboBox()
        self._populate_solvent_combo()
        self.solvent_combo.currentIndexChanged.connect(self._recompute)
        form.addRow('Solvent:', self.solvent_combo)

        self.temp_edit = QtWidgets.QLineEdit('25')
        self.temp_edit.editingFinished.connect(self._recompute)
        self.temp_unit = QtWidgets.QComboBox()
        self.temp_unit.addItems(U.unit_options('temperature'))     # °C default
        # Track the previous unit so a unit switch CONVERTS the shown value (keeping the
        # physical temperature), rather than reinterpreting the same number in the new
        # unit — matching the Data tab's _on_unit_changed behavior.
        self._temp_unit_prev = self.temp_unit.currentText()
        self.temp_unit.currentIndexChanged.connect(self._on_temp_unit_changed)
        form.addRow('Temperature:', _value_unit_row(self.temp_edit, self.temp_unit))

        self.wl_edit = QtWidgets.QLineEdit('532')
        self.wl_edit.editingFinished.connect(self._recompute)
        wl_unit = QtWidgets.QLabel('nm')     # wavelength has no alternative unit
        form.addRow('Wavelength:', _value_unit_row(self.wl_edit, wl_unit))
        top.addWidget(form_box, 0)

        read_box = QtWidgets.QGroupBox('Value at This Condition')
        rv = QtWidgets.QVBoxLayout(read_box)
        # Rich-text QLabels: the tier badge is an inline colored ● regenerated on each
        # recompute (and on a theme switch, since _recompute re-runs) so it re-themes.
        self.n_readout = QtWidgets.QLabel()
        self.n_readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.n_readout.setWordWrap(True)
        self.eta_readout = QtWidgets.QLabel()
        self.eta_readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.eta_readout.setWordWrap(True)
        self.eta_caveat = ThemedLabel('', role='muted', size=11)
        self.eta_caveat.setWordWrap(True)
        rv.addWidget(self.n_readout)
        rv.addWidget(self.eta_readout)
        rv.addWidget(self.eta_caveat)
        rv.addStretch(1)
        top.addWidget(read_box, 1)
        outer.addLayout(top)

        # ---- ONE shared figure: T-plot (left cell) + RI-vs-λ (right cell) ----
        # Short and wide (≈ half the old stacked height): the two plots resize
        # together on one canvas — the DLS correlogram grid pattern — and the freed
        # vertical space goes to the form/readout above (owner request 2026-07-06).
        plots = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(plots)
        pv.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(9.6, 3.4), constrained_layout=True)
        self.canvas = make_canvas_expanding(FigureCanvas(self.figure))
        pv.addWidget(themed_navtoolbar(self.canvas, plots))
        pv.addWidget(self.canvas, 1)
        # One axis-control bar per y-axis, all on the one canvas (the T-cell's twin
        # axes share x, so editing x on either is consistent, not conflicting). The η
        # bar defaults to log — η spans orders of magnitude across solvents (≈1 to
        # ~1000 mPa·s for glycerol) — and η is the T-plot's HOST/left axis.
        self.a_bar_eta = AxisControlBar(self.canvas)
        self.a_bar_ri = AxisControlBar(self.canvas)
        self.b_bar = AxisControlBar(self.canvas)
        # η starts on a log scale, but remember the user's choice so a later recompute
        # doesn't snap it back to log every redraw.
        self._eta_yscale = 'log'
        self.a_bar_eta.y_scale.currentTextChanged.connect(
            lambda s: setattr(self, '_eta_yscale', s))
        pv.addWidget(self._labeled_bar('η axis (left):', self.a_bar_eta))
        pv.addWidget(self._labeled_bar('RI axis (right):', self.a_bar_ri))
        pv.addWidget(self._labeled_bar('RI vs λ axis:', self.b_bar))
        outer.addWidget(plots, 1)

        self.band_note = ThemedLabel(
            'Shaded band = conservative bound at each condition '
            '(Theory-and-Equations-Guide §5) — shown, never propagated into an analysis SE. '
            'Curves stop where the library’s validity ends.',
            role='hint', size=11)
        self.band_note.setWordWrap(True)
        outer.addWidget(self.band_note)

    @staticmethod
    def _labeled_bar(text: str, bar: AxisControlBar) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QtWidgets.QLabel(text))
        row.addWidget(bar, 1)
        return w

    def _populate_solvent_combo(self) -> None:
        """Fill the combo from all library solvents; estimate-tier rows are labeled
        '(estimate)'. The canonical name rides itemData so the label is cosmetic."""
        self.solvent_combo.blockSignals(True)
        self.solvent_combo.clear()
        for name in solvents_lib.available_solvents(None):
            tier = solvents_lib.solvent_tier(name)
            label = name if tier == 'primary' else f'{name} (estimate)'
            self.solvent_combo.addItem(label, name)
        self.solvent_combo.blockSignals(False)

    def reseed_from_settings(self) -> None:
        """Preselect the solvent from the global ``default_solvent`` setting (falls back
        to the first row if the saved default is absent). Called ONCE, on build — never
        on Settings Apply ("seed, never override"): neither the main window nor
        Utilities' reseed path cascades here."""
        default = getattr(self.controller.settings, 'default_solvent', None)
        if default:
            i = self.solvent_combo.findData(
                solvents_lib.normalize_solvent_name(default))
            if i >= 0:
                self.solvent_combo.blockSignals(True)
                self.solvent_combo.setCurrentIndex(i)
                self.solvent_combo.blockSignals(False)
        self._recompute()

    # ------------------------------------------------------------ compute ---
    def changeEvent(self, ev) -> None:
        # A theme switch delivers PaletteChange: re-run _recompute so the inline tier-
        # badge colors (plain-QLabel HTML spans) re-theme. The current solvent /
        # condition are preserved — this only recolors + redraws.
        if ev.type() == QtCore.QEvent.Type.PaletteChange and hasattr(self, 'n_readout'):
            self._recompute()
        super().changeEvent(ev)

    def _current_solvent(self) -> Optional[str]:
        data = self.solvent_combo.currentData()
        return data or None

    def _on_temp_unit_changed(self, *args) -> None:
        """Convert the shown temperature into the newly chosen unit (preserving the
        physical value), then recompute — a unit switch must not reinterpret the number."""
        new_unit = self.temp_unit.currentText()
        try:
            k = U.to_canonical('temperature', float(self.temp_edit.text()),
                               self._temp_unit_prev)
            self.temp_edit.setText(
                f"{U.from_canonical('temperature', k, new_unit):g}")
        except (ValueError, KeyError):
            pass       # leave an unparseable entry as-is; _recompute will note it
        self._temp_unit_prev = new_unit
        self._recompute()

    def _temperature_C(self) -> Optional[float]:
        """The temperature in °C from the value+unit inputs, or None if unparseable."""
        try:
            k = U.to_canonical('temperature', float(self.temp_edit.text()),
                               self.temp_unit.currentText())
        except (ValueError, KeyError):
            return None
        return k - 273.15

    def _wavelength_nm(self) -> Optional[float]:
        try:
            return float(self.wl_edit.text())
        except ValueError:
            return None

    def _tier_badge(self, tier: str) -> str:
        """An inline colored ● + tier word (teal primary / violet estimate), themed."""
        role = 'lib_primary' if tier == 'primary' else 'lib_estimate'
        color = theme_token(self, role)
        return (f'<span style="color:{color}">●</span> '
                f'<span style="color:{color}"><b>{tier}</b></span>')

    def _recompute(self, *args) -> None:
        """Recompute the numeric readout and both plots for the current inputs."""
        name = self._current_solvent()
        if name is None:
            self.n_readout.setText('')
            self.eta_readout.setText('')
            self.eta_caveat.setText('')
            self._clear_plots()
            return
        tier = solvents_lib.solvent_tier(name) or 'estimate'
        info = solvents_lib.solvent_property_info(name)
        t_c = self._temperature_C()
        lam = self._wavelength_nm()
        badge = self._tier_badge(tier)

        # --- n readout ---
        n_val = None
        if t_c is None or lam is None:
            self.n_readout.setText('n — enter a numeric temperature and wavelength.')
        else:
            try:
                n_val, n_unc = self.controller.solvent_value_n(name, lam, t_c)
                # format_pm: the same PDG place-chooser the Data-tab autofill rounds
                # with, so this readout and an autofilled cell can never disagree in
                # their last digit.
                self.n_readout.setText(
                    f'n = <b>{format_pm(n_val, n_unc)}</b> &nbsp; {badge}')
            except ValueError as exc:
                self.n_readout.setText(f'n — {exc}')

        # --- η readout + source-grade caveat ---
        eta_val = None
        if not info.get('has_viscosity'):
            self.eta_readout.setText(f'η — not available for {name}. &nbsp; {badge}')
            self.eta_caveat.setText('')
        elif t_c is None:
            self.eta_readout.setText('η — enter a numeric temperature.')
            self.eta_caveat.setText('')
        else:
            try:
                eta_val, eta_unc = self.controller.solvent_value_eta(name, t_c)
                self.eta_readout.setText(
                    f'η = <b>{format_pm(eta_val * 1e3, eta_unc * 1e3, "mPa·s")}</b> '
                    f'&nbsp; {badge}')
                self.eta_caveat.setText(self._eta_caveat(info.get('eta_source_grade')))
            except ValueError as exc:
                self.eta_readout.setText(f'η — {exc}')
                self.eta_caveat.setText('')

        self._draw_plots(name, t_c, lam, n_val, eta_val)

    @staticmethod
    def _eta_caveat(grade: Optional[str]) -> str:
        """One short line driven by the three-tier eta_source_grade ladder; no caveat
        for a critically-evaluated reference correlation."""
        if grade == 'measured':
            return 'Viscosity: single-lab measurement.'
        if grade == 'bulk':
            return 'Viscosity: handbook/bulk-compilation estimate.'
        return ''       # 'reference' (or unknown) → no caveat

    # -------------------------------------------------------------- plots ---
    def _clear_plots(self) -> None:
        self.figure.clear()
        self.a_bar_eta.attach(None)
        self.a_bar_ri.attach(None)
        self.b_bar.attach(None)
        self.canvas.draw_idle()

    def _draw_plots(self, name, t_c, lam, n_sel, eta_sel) -> None:
        """Both plot cells on the ONE shared figure. Left cell: η (host/left axis) +
        RI (twinx right) vs temperature, each over its own box, each with its
        confidence band + a selection dot at the chosen temperature. Right cell: RI
        vs wavelength at the selected temperature, over the RI λ box, with band +
        a selection dot at the chosen wavelength.

        The gridspec stays FLAT (a nested subgridspec trips constrained_layout's
        tick-bbox pass on degenerate log axes — same as the DLS correlogram grid);
        twinx() inside a gridspec cell is fine."""
        self.figure.clear()
        gs = self.figure.add_gridspec(1, 2)
        ax_eta = self.figure.add_subplot(gs[0, 0])   # host: η, the strong T-variation
        ax_ri = ax_eta.twinx()                       # RI rides the right axis
        ax_lam = self.figure.add_subplot(gs[0, 1])
        ax_eta.set_xlabel('Temperature (°C)')
        ax_eta.set_ylabel('Viscosity η (mPa·s)', color=_ETA_COLOR)
        ax_ri.set_ylabel('Refractive index n', color=_RI_COLOR)
        ax_eta.tick_params(axis='y', colors=_ETA_COLOR)
        ax_ri.tick_params(axis='y', colors=_RI_COLOR)

        # η curve (T only; may be absent).
        eta_drawn = False
        try:
            te, e, be = self.controller.solvent_curve_eta_vs_T(name)
            e_mpa, be_mpa = e * 1e3, be * 1e3
            ax_eta.plot(te, e_mpa, color=_ETA_COLOR, lw=1.6, label='η')
            ax_eta.fill_between(te, e_mpa - be_mpa, e_mpa + be_mpa, color=_ETA_COLOR,
                                alpha=_BAND_ALPHA, linewidth=0)
            eta_drawn = True
        except ValueError:
            pass
        # RI curve (needs the selected wavelength in range).
        ri_drawn = False
        if lam is not None:
            try:
                t, n, band = self.controller.solvent_curve_n_vs_T(name, lam)
                ax_ri.plot(t, n, color=_RI_COLOR, lw=1.6, label='n')
                ax_ri.fill_between(t, n - band, n + band, color=_RI_COLOR,
                                   alpha=_BAND_ALPHA, linewidth=0)
                ri_drawn = True
            except ValueError:
                pass

        # User-selection dots (hidden for a property that is out of range at t_c).
        if eta_drawn and eta_sel is not None and t_c is not None:
            ax_eta.plot([t_c], [eta_sel * 1e3], 'o', color=_ETA_COLOR, ms=7,
                        markeredgecolor='white', zorder=5)
        if ri_drawn and n_sel is not None and t_c is not None:
            ax_ri.plot([t_c], [n_sel], 'o', color=_RI_COLOR, ms=7,
                       markeredgecolor='white', zorder=5)

        # Degenerate branches: a missing curve blanks ITS OWN axis's ticks (η-absent
        # solvent → bare host; RI out of λ-range → bare right axis).
        if eta_drawn:
            ax_eta.set_yscale(self._eta_yscale)   # remembered (log by default)
        else:
            ax_eta.set_yticks([])
        if not ri_drawn:
            ax_ri.set_yticks([])

        # Right cell: RI vs wavelength.
        ax_lam.set_xlabel('Wavelength (nm)')
        ax_lam.set_ylabel('Refractive index n')
        lam_drawn = False
        if t_c is not None:
            try:
                w, n, band = self.controller.solvent_curve_n_vs_lambda(name, t_c)
                ax_lam.plot(w, n, color=_RI_COLOR, lw=1.6)
                ax_lam.fill_between(w, n - band, n + band, color=_RI_COLOR,
                                    alpha=_BAND_ALPHA, linewidth=0)
                lam_drawn = True
            except ValueError:
                pass
        if lam_drawn and n_sel is not None and lam is not None:
            ax_lam.plot([lam], [n_sel], 'o', color=_RI_COLOR, ms=7,
                        markeredgecolor='white', zorder=5)

        self.canvas.draw_idle()
        self.a_bar_eta.attach(ax_eta if eta_drawn else None)
        self.a_bar_ri.attach(ax_ri if ri_drawn else None)
        self.b_bar.attach(ax_lam if lam_drawn else None)
