"""
gui/settings_module.py
======================

The Settings tab: **global** user preferences (the shell disables the sidebar
here). Edits the controller's `SettingsState` and persists it to settings.json.

The locked rule is **seed, never override**: every value here is the *starting
default* a module's per-run control is seeded with. Changing a setting updates the
defaults future runs/controls start from — it never alters an existing result or
acts as a hidden global multiplier. Appearance (theme) is the one purely-global
exception.

Apply commits + persists and tells the shell to re-apply the theme and re-seed the
module controls; Restore defaults resets to the factory `SettingsState` and applies.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from app import units as U
from app.settings import SettingsState
from gui.help import HelpBadge
from gui.theme import ThemedLabel
from gui.worker import busy_notice, runner

# Plot-axis display-unit choices surfaced in Settings (feedback 2026-06-26 #8):
# (quantity, label). The combo for each offers U.unit_options(quantity).
_PLOT_UNIT_ROWS = [
    ('time', 'Delay time τ'),
    ('intensity', 'Count rate'),
    ('scattering_q2', 'Scattering q²'),
    ('diffusion', 'Diffusion coeff. D'),
    ('decay_rate', 'Decay rate Γ'),
    ('concentration', 'Concentration'),
    ('radius', 'Radius (Rh / Rg)'),
    ('molar_mass', 'Molar mass Mw'),
]


class SettingsModule(QtWidgets.QWidget):
    """Editor for the global SettingsState (seed defaults + appearance)."""

    applied = QtCore.Signal()   # emitted after Apply/Restore: shell re-seeds + re-themes

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._build_ui()
        self._load_from(self.controller.settings)

    # ------------------------------------------------------------------ UI ---
    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        note = ThemedLabel(
            'These are starting defaults that seed each module’s per-run controls. '
            'You can still change a value per run, and that per-run value (not the '
            'setting) is what is used and recorded. Saved to settings.json at the '
            'program root.', role='muted')
        note.setWordWrap(True)
        outer.addWidget(note)

        host = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(host)

        # NB: the DLS cumulant-order and Rh-grid / CONTIN-α defaults used to live
        # here. They now have per-run controls ON the DLS Correlogram and
        # Distribution sub-tabs (where the fit happens), seeded from these same
        # SettingsState fields. The fields are retained (and round-tripped through
        # _collect) so Apply/Restore never silently zeroes them.

        form.addRow(self._header('DLS analysis defaults'))
        self.cumulant_method = QtWidgets.QComboBox()
        self.cumulant_method.addItem('Nonlinear', 'nonlinear')
        self.cumulant_method.addItem('Linear', 'linear')
        self.cumulant_method.setToolTip(
            'Cumulant fitting method, applied to every cumulant-based DLS analysis '
            '(per-measurement cumulant, Γ vs q², replicate averaging).\n'
            '• Nonlinear (default): fits g₂−1 directly with a floating '
            'baseline — more robust to drift and noisy/low-count data.\n'
            '• Linear: weighted polynomial fit of ln(g₂−1).\n'
            'Switching this clears existing cumulant-based results (you will be asked '
            'to confirm).')
        form.addRow('Cumulant method', self.cumulant_method)
        self.skip_channels = QtWidgets.QSpinBox()
        self.skip_channels.setRange(0, 50)
        self.skip_channels.setToolTip(
            'Drop this many leading correlator channels (the shortest lags) from '
            'every DLS fit, to remove detector afterpulsing / dead-time artefacts. '
            '0 = keep all channels. Applies to cumulant and distribution methods '
            'alike, and composes with a delay-window minimum (the later of the two '
            'wins). Keep it small for distribution fits — over-skipping can erase a '
            'genuine small-particle population.')
        form.addRow('Skip initial lag channels', self.skip_channels)

        form.addRow(self._header('SLS defaults'))
        self.geometry = QtWidgets.QComboBox()
        self.geometry.addItems(['VU', 'VV', 'VH'])
        form.addRow('Standard geometry', self.geometry)
        self.qrg_max = QtWidgets.QDoubleSpinBox()
        self.qrg_max.setRange(0.5, 3.0)
        self.qrg_max.setSingleStep(0.1)
        form.addRow('Guinier qRg validity limit', self.qrg_max)

        form.addRow(self._header('Uncertainty'))
        self.se_estimator = QtWidgets.QComboBox()
        self.se_estimator.addItem('HC3 (robust)', 'hc3')
        self.se_estimator.addItem('Classical OLS', 'ols')
        self.se_estimator.setToolTip(
            'Covariance estimator behind every reported regression ± (SLS Zimm/Berry, '
            'Debye, Guinier, calibration-free A₂; DLS Γ vs q² and kD; the Rg/A₂–Mw '
            'scaling exponent). Switching clears existing ±-bearing results (you will '
            'be asked to confirm). See Advanced Guide §15.1.')
        est_help = HelpBadge(
            'Which standard error the app reports for its straight-line and '
            'multilinear fits.',
            bullets=[
                '<b>HC3 (robust)</b> — the default. A heteroscedasticity-consistent '
                'SE that never under-reports when point precision varies across '
                'angles/concentrations. Use this unless you have a specific reason not to.',
                '<b>Classical OLS</b> — the textbook s²(XᵀX)⁻¹ SE. Choose this only to '
                'reproduce a result from classical software, a published table, or a '
                'spreadsheet like-for-like. It can under-report (~10% low on a short, '
                'high-leverage concentration ladder), which is why it is not the default.',
                'The choice is global, persists across restart, and is recorded on '
                'each result and written into the export (an OLS ± is labelled '
                '“SE: classical OLS”). Point estimates (Mw, Rg, D, Rh) are unaffected.',
                'Detail and formulas: Advanced Guide §15.1.',
            ])
        est_row = QtWidgets.QWidget()
        est_lay = QtWidgets.QHBoxLayout(est_row)
        est_lay.setContentsMargins(0, 0, 0, 0)
        est_lay.addWidget(self.se_estimator)
        est_lay.addWidget(est_help)
        est_lay.addStretch(1)
        form.addRow('Regression SE estimator', est_row)

        # NB: the synthetic-generator (β/noise/points) and intensity-trace (outlier
        # k, running-average window) defaults used to live here. Per feedback
        # 2026-06-26 #6 they were moved into their own tabs (Utilities → Synthetic
        # generator / Traces) as plain session fields, so they no longer clutter
        # this global tab.

        form.addRow(self._header('Display units'))
        plot_note = ThemedLabel(
            'Display units for plot axes AND the Cross-Sample result tables (Rg/Rh, Mw). '
            'Everything is stored in canonical units and only converted for display; '
            'changing a unit redraws the plots and re-labels the tables.',
            role='hint', size=11)
        plot_note.setWordWrap(True)
        form.addRow(plot_note)
        self.plot_unit_combos = {}
        for q, label in _PLOT_UNIT_ROWS:
            combo = QtWidgets.QComboBox()
            combo.addItems(U.unit_options(q))
            self.plot_unit_combos[q] = combo
            form.addRow(label, combo)

        form.addRow(self._header('Appearance'))
        self.theme = QtWidgets.QComboBox()
        self.theme.addItems(['system', 'light', 'dark'])
        self.theme.setToolTip(
            "'system' follows your OS; 'light'/'dark' override it.")
        form.addRow('Theme', self.theme)
        self.palette = QtWidgets.QComboBox()
        self.palette.addItems(['okabe_ito', 'tab10', 'grayscale'])
        form.addRow('Plot palette', self.palette)
        self.show_tooltips = QtWidgets.QCheckBox('Show tooltips on hover')
        self.show_tooltips.setToolTip(
            'Passive hover tooltips throughout the app. The "?" help buttons still '
            'work on click when this is off.')
        form.addRow('Tooltips', self.show_tooltips)

        outer.addWidget(host)

        btns = QtWidgets.QHBoxLayout()
        apply_btn = QtWidgets.QPushButton('Apply')
        apply_btn.clicked.connect(self._apply)
        reset_btn = QtWidgets.QPushButton('Restore defaults')
        reset_btn.clicked.connect(self._restore)
        btns.addWidget(apply_btn)
        btns.addWidget(reset_btn)
        btns.addStretch(1)
        outer.addLayout(btns)
        outer.addStretch(1)

    @staticmethod
    def _header(text: str) -> QtWidgets.QLabel:
        # ThemedLabel so the header follows the theme (a plain stylesheet without a
        # `color` froze to the build-time palette → unreadable light-grey on the light
        # theme). Kept sentence-case + bold; no all-caps (feedback 2026-06-30 #2).
        return ThemedLabel(text, role='header', bold=True, extra='margin-top:8px;')

    # ----------------------------------------------------------- transfer ---
    def _load_from(self, s: SettingsState) -> None:
        # Relocated DLS seeds have no widget here anymore; retain their values so
        # _collect can round-trip them. Loading SettingsState() on Restore therefore
        # resets them to the factory defaults too.
        self._retained_dls = dict(
            cumulant_order=s.cumulant_order,
            rh_grid_min_nm=s.rh_grid_min_nm, rh_grid_max_nm=s.rh_grid_max_nm,
            rh_grid_points=s.rh_grid_points,
            lcurve_alpha_min=s.lcurve_alpha_min, lcurve_alpha_max=s.lcurve_alpha_max)
        i = self.cumulant_method.findData(s.cumulant_method)
        self.cumulant_method.setCurrentIndex(i if i >= 0 else 0)
        self.skip_channels.setValue(s.skip_initial_channels)
        self.geometry.setCurrentText(s.standard_geometry)
        self.qrg_max.setValue(s.guinier_qrg_max)
        j = self.se_estimator.findData(s.se_estimator)
        self.se_estimator.setCurrentIndex(j if j >= 0 else 0)
        for q, combo in self.plot_unit_combos.items():
            combo.setCurrentText(s.plot_units.get(q, U.default_unit(q)))
        self.theme.setCurrentText(s.theme)
        self.palette.setCurrentText(s.plot_palette)
        self.show_tooltips.setChecked(s.show_tooltips)

    def _collect(self) -> SettingsState:
        return SettingsState(
            # Relocated DLS seeds: carried through unchanged (edited via the DLS
            # sub-tabs' per-run controls, not here).
            **self._retained_dls,
            cumulant_method=self.cumulant_method.currentData(),
            skip_initial_channels=self.skip_channels.value(),
            standard_geometry=self.geometry.currentText(),
            guinier_qrg_max=self.qrg_max.value(),
            se_estimator=self.se_estimator.currentData(),
            plot_units={q: combo.currentText()
                        for q, combo in self.plot_unit_combos.items()},
            theme=self.theme.currentText(),
            plot_palette=self.palette.currentText(),
            show_tooltips=self.show_tooltips.isChecked(),
        )

    # ----------------------------------------------------------- actions ---
    @QtCore.Slot()
    def _apply(self) -> None:
        # Settings values seed the analysis defaults a background run reads
        # mid-flight — applying under it could change its numbers (invariant 4).
        if runner().is_busy:
            busy_notice(self)
            return
        new = self._collect()
        # Two independent switches (cumulant method, SE estimator) each make some
        # existing results stale and offer a clear-and-warn. Collect BOTH confirmations
        # up front and only mutate after every prompt is confirmed — otherwise cancelling
        # the second prompt would leave the first's clear already done (results gone) but
        # the Apply aborted, a dropdown/setting mismatch. Any Cancel reverts every dropdown
        # and applies nothing.
        old_method = self.controller.settings.cumulant_method
        old_estimator = self.controller.settings.se_estimator
        clear_cumulant = False
        clear_se = False

        if new.cumulant_method != old_method:
            n = self.controller.cumulant_dependent_result_count()
            if n > 0:
                resp = QtWidgets.QMessageBox.question(
                    self, 'Switch cumulant method?',
                    f"Switching the cumulant method will clear {n} existing "
                    f"cumulant-based result(s) — cumulant fits, Γ vs q², and "
                    f"replicate averages. Distributions, SLS, and any hand-entered "
                    f"Rh are kept.\n\nContinue?",
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.Cancel,
                    QtWidgets.QMessageBox.StandardButton.Cancel)
                if resp != QtWidgets.QMessageBox.StandardButton.Yes:
                    self._revert_dropdowns(old_method, old_estimator)
                    return
                clear_cumulant = True

        if new.se_estimator != old_estimator:
            n = self.controller.se_dependent_result_count()
            if n > 0:
                label = 'classical OLS' if new.se_estimator == 'ols' else 'HC3 (robust)'
                resp = QtWidgets.QMessageBox.question(
                    self, 'Switch uncertainty estimator?',
                    f"Switching to {label} will clear {n} existing result(s) that "
                    f"report a regression ± — Zimm/Berry, Debye, Guinier, "
                    f"calibration-free A₂, Γ vs q², and D vs c — so they recompute "
                    f"under the new estimator. Point estimates (Mw, Rg, D, Rh) are "
                    f"unaffected; distributions, cumulants, and any hand-entered "
                    f"values are kept.\n\nContinue?",
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.Cancel,
                    QtWidgets.QMessageBox.StandardButton.Cancel)
                if resp != QtWidgets.QMessageBox.StandardButton.Yes:
                    self._revert_dropdowns(old_method, old_estimator)
                    return
                clear_se = True

        # All prompts confirmed — now it is safe to clear and apply.
        if clear_cumulant:
            self.controller.clear_cumulant_dependent_results()
        if clear_se:
            self.controller.clear_se_dependent_results()
        self.controller.apply_settings(new)               # persists to settings.json
        self.applied.emit()

    def _revert_dropdowns(self, cumulant_method: str, se_estimator: str) -> None:
        """Restore both stale-guarded dropdowns to the still-current settings after a
        cancelled Apply, so no uncommitted choice lingers in the UI."""
        j = self.cumulant_method.findData(cumulant_method)
        self.cumulant_method.setCurrentIndex(j if j >= 0 else 0)
        k = self.se_estimator.findData(se_estimator)
        self.se_estimator.setCurrentIndex(k if k >= 0 else 0)

    @QtCore.Slot()
    def _restore(self) -> None:
        if runner().is_busy:      # before _load_from, so the widgets aren't
            busy_notice(self)     # reset to values that then fail to apply
            return
        self._load_from(SettingsState())                  # factory defaults
        self._apply()
