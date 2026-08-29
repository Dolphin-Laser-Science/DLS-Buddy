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
from gui.solvent_picker import index_of_solvent, populate_solvent_combo
from gui.widgets import first_selectable_index
from gui.theme import ThemedLabel
from gui.worker import busy_notice, runner

# Calculation-nuance tooltips for the unit choices whose reach is easy to
# misjudge (spec 04 stage 2 extended both to the SLS axes).
_PLOT_UNIT_TIPS = {
    'scattering_q2':
        'Applies to every q² axis: the DLS Γ-vs-q² and DDLS plots AND the SLS '
        'plots (Rayleigh ratio, Debye, Guinier, form factors, Zimm — whose '
        'printed spacing constant k follows this unit). Fitted results are '
        'unaffected; only axes rescale.',
    'concentration':
        'Applies to the concentration axes: the DLS D-vs-c plot and the SLS '
        'calibration-free A₂ plot. Legend concentration labels always read '
        'mg/mL, whatever the axis unit.',
}

# Plot-axis display-unit choices surfaced in Settings:
# (quantity, label). The combo for each offers U.unit_options(quantity).
_PLOT_UNIT_ROWS = [
    ('time', 'Delay time τ'),
    ('intensity', 'Count rate'),
    ('scattering_q2', 'Scattering q²'),
    ('diffusion', 'Diffusion coeff. D'),
    ('decay_rate', 'Decay rate Γ'),
    ('concentration', 'Concentration'),
    ('radius', 'Radius (R<sub>h</sub> / R<sub>g</sub>)'),
    ('molar_mass', 'Molar mass M<sub>w</sub>'),
]


def _seed_combo(combo: QtWidgets.QComboBox, value, *, use_data: bool,
                find=None) -> None:
    """Select ``value`` in ``combo`` (by data or text). If it isn't a current item —
    e.g. a hand-edited settings.json, or a solvent later dropped from the library —
    insert it as an extra entry rather than silently falling back to a different one,
    so an untouched Apply writes the same value back instead of replacing it.

    ``find`` overrides how a row is located, for a field whose own matching rule is not
    exact equality. The solvent field passes ``index_of_solvent`` so a hand-edited
    ``"Water"`` snaps to the ``water`` row instead of adding a near-duplicate beside it —
    one kind of value must not have two matching rules in one program.

    The append below is only sound because the appended row is findable BY THE SAME RULE
    on the next call. If it were not, every reload would append again: this runs on init,
    on Restore defaults, and on a cancelled Apply's revert, so the list would grow without
    bound. ``index_of_solvent`` casefolds both sides for exactly this reason.
    """
    if not value:
        # Nothing to seed. Appending here would add a literal 'None'/'' row -- and, since
        # no lookup would ever match it, one more on every reload.
        combo.setCurrentIndex(max(0, first_selectable_index(combo)))
        return
    if find is not None:
        idx = find(combo, value)
    else:
        idx = combo.findData(value) if use_data else combo.findText(value)
    if idx < 0:
        combo.addItem(str(value), value if use_data else None)
        idx = combo.count() - 1
    combo.setCurrentIndex(idx)


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
            'setting) is what is used and recorded — except where a setting says '
            'otherwise in its own tooltip: the file-loading, appearance, display-unit '
            'and result-formatting choices are global, and the Brookhaven .dat '
            'baseline is read once, when the file loads. Saved to settings.json at '
            'the program root.', role='muted')
        note.setWordWrap(True)
        outer.addWidget(note)

        host = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(host)

        # NB: the DLS cumulant-order and Rh-grid / CONTIN-α defaults used to live
        # here. They now have per-run controls ON the DLS Correlogram and
        # Distribution sub-tabs (where the fit happens), seeded from these same
        # SettingsState fields. The fields are retained (and round-tripped through
        # _collect) so Apply/Restore never silently zeroes them.

        form.addRow(self._header('DLS Analysis Defaults'))
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
        form.addRow('Cumulant method:', self.cumulant_method)
        self.skip_channels = QtWidgets.QSpinBox()
        self.skip_channels.setRange(0, 50)
        self.skip_channels.setToolTip(
            'Drop this many leading correlator channels (shortest lags) from every DLS '
            'fit to remove afterpulsing / dead-time artifacts; keep it small so a real '
            'small-particle mode is not erased. See the Theory-and-Equations-Guide.')
        form.addRow('Skip initial lag channels:', self.skip_channels)

        form.addRow(self._header('File Loading'))
        self.bh_dat_baseline = QtWidgets.QComboBox()
        self.bh_dat_baseline.addItem('Calculated', 'calculated')
        self.bh_dat_baseline.addItem('Measured', 'measured')
        self.bh_dat_baseline.setToolTip(
            'Baseline used to normalize a Brookhaven raw ASCII .dat correlogram, '
            'whose channels hold raw counts: g₂−1 = counts/B − 1. Applies only when '
            'the file itself does not name a baseline. See the '
            'Theory-and-Equations-Guide.')
        dat_help = HelpBadge(
            'Brookhaven .dat files store raw correlator counts plus two baselines, '
            'so one of them has to be chosen before the correlogram exists.',
            bullets=[
                '<b>Calculated</b> — the default. Formed from the run totals '
                '(counts A × counts B, over the number of samples), so it uses '
                'the whole measurement.',
                '<b>Measured</b> — the average of the baseline channels the acquiring '
                'software set aside. Matches what Particle Explorer reports, but '
                'depends on those channels lying past the decay.',
                'A file that names its own baseline overrides this setting.',
                'The two typically differ by a few tenths of a percent. That makes '
                'no difference to the default fits, which float or estimate a '
                'baseline of their own and absorb it; it shifts the size by a few '
                'percent for the linear cumulant method, which assumes none.',
                'A large difference between the two baselines is itself a dust or '
                'slow-drift warning, whichever you pick.',
            ])
        dat_row = QtWidgets.QWidget()
        dat_lay = QtWidgets.QHBoxLayout(dat_row)
        dat_lay.setContentsMargins(0, 0, 0, 0)
        dat_lay.addWidget(self.bh_dat_baseline, 1)   # stretch like a badge-less row
        dat_lay.addWidget(dat_help)
        form.addRow('Brookhaven .dat baseline:', dat_row)

        form.addRow(self._header('SLS Defaults'))
        self.geometry = QtWidgets.QComboBox()
        self.geometry.addItems(['VU', 'VV', 'VH'])
        self.geometry.setToolTip(
            'Scattering geometry of the calibration standard: VV = polarized, '
            'VH = depolarized, VU = no analyzer (a vertical laser with no analyzer). '
            'See the Theory-and-Equations-Guide.')
        form.addRow('Standard geometry:', self.geometry)
        self.qrg_max = QtWidgets.QDoubleSpinBox()
        self.qrg_max.setRange(0.5, 3.0)
        self.qrg_max.setSingleStep(0.1)
        self.qrg_max.setToolTip(
            'Maximum qRg for a valid Guinier fit; above it the Guinier expansion '
            'breaks down. See the Theory-and-Equations-Guide (Guinier).')
        form.addRow('Guinier qRg validity limit:', self.qrg_max)

        form.addRow(self._header('Solvent Library'))
        self.default_solvent = QtWidgets.QComboBox()
        populate_solvent_combo(self.default_solvent)
        self.default_solvent.setToolTip(
            'The solvent the Solvent Explorer (a Utilities sub-tab) starts on. A '
            'convenience default only — it never sets a value in any analysis.')
        form.addRow('Default solvent:', self.default_solvent)

        form.addRow(self._header('Uncertainty'))
        self.se_estimator = QtWidgets.QComboBox()
        self.se_estimator.addItem('HC3 (robust)', 'hc3')
        self.se_estimator.addItem('Classical OLS', 'ols')
        self.se_estimator.setToolTip(
            'Covariance estimator behind every reported regression ± (SLS Zimm/Berry, '
            'Debye, Guinier, calibration-free A₂; DLS Γ vs q² and kD; the Rg/A₂–Mw '
            'scaling exponent). Switching clears existing ±-bearing results (you will '
            'be asked to confirm). See Theory-and-Equations-Guide §6.1.')
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
                'each result and written into the export (an OLS ± is labeled '
                '“SE: classical OLS”). Point estimates (Mw, Rg, D, Rh) are unaffected.',
                'Detail and formulas: Theory-and-Equations-Guide §6.1.',
            ])
        est_row = QtWidgets.QWidget()
        est_lay = QtWidgets.QHBoxLayout(est_row)
        est_lay.setContentsMargins(0, 0, 0, 0)
        est_lay.addWidget(self.se_estimator, 1)      # stretch like a badge-less row
        est_lay.addWidget(est_help)
        form.addRow('Regression SE estimator:', est_row)

        # NB: the synthetic-generator (β/noise/points) and intensity-trace (outlier
        # k, running-average window) defaults used to live here. They were
        # moved into their own tabs (Utilities → Synthetic
        # generator / Traces) as plain session fields, so they no longer clutter
        # this global tab.

        form.addRow(self._header('Display Units'))
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
            if q in _PLOT_UNIT_TIPS:
                combo.setToolTip(_PLOT_UNIT_TIPS[q])
            self.plot_unit_combos[q] = combo
            form.addRow(f'{label}:', combo)

        form.addRow(self._header('Result Formatting'))
        self.no_unc_sig_figs = QtWidgets.QSpinBox()
        self.no_unc_sig_figs.setRange(1, 4)
        self.no_unc_sig_figs.setToolTip(
            'Significant figures for EVERY result that has no honest uncertainty — '
            'a single correlogram, an NNLS/CONTIN distribution, a single-angle Mw, and '
            'derived numbers like Γ, PDI, R², and qRg. A value that HAS a ± is always '
            'rounded to the place its ± supports — this never changes that.')
        sig_help = HelpBadge(
            'How many digits to show for a result that carries no ±.',
            bullets=[
                'A value that <b>has</b> an uncertainty is shown to the precision that ± '
                'supports (a ±1 nm SE is not reported to the tenths place — that would '
                'imply confidence the data does not support). This setting does '
                '<b>not</b> touch those.',
                'A value with <b>no</b> uncertainty — a single-correlogram fit, an '
                'NNLS/CONTIN peak, a single-angle Mw, and every derived number that '
                'carries no ± (Γ, PDI, R², qRg, …) — has no uncertainty to set its '
                'precision, so a fixed number of significant figures is used instead. '
                'This one knob controls all of them, uniformly.',
                'Default 3. Lower it (e.g. to 2) for a more conservative look — note that '
                'coarsens everything without a ±, including R²/qRg, so their trailing '
                'digits are lost. The app never fabricates a ± to show more digits.',
            ])
        sig_row = QtWidgets.QWidget()
        sig_lay = QtWidgets.QHBoxLayout(sig_row)
        sig_lay.setContentsMargins(0, 0, 0, 0)
        sig_lay.addWidget(self.no_unc_sig_figs)
        sig_lay.addWidget(sig_help)
        sig_lay.addStretch(1)
        form.addRow('No-uncertainty precision (sig. figs):', sig_row)

        form.addRow(self._header('Appearance'))
        self.theme = QtWidgets.QComboBox()
        self.theme.addItems(['system', 'light', 'dark'])
        self.theme.setToolTip(
            "'system' follows your OS; 'light'/'dark' override it.")
        form.addRow('Theme:', self.theme)
        self.palette = QtWidgets.QComboBox()
        self.palette.addItems(['okabe_ito', 'tab10', 'grayscale'])
        self.palette.setToolTip(
            'Series colors for every plot. "okabe_ito" (default) is a colorblind-safe '
            'palette; series also vary by marker/linestyle so they stay distinct in '
            'grayscale and for colorblind readers.')
        form.addRow('Plot palette:', self.palette)
        self.plot_match_theme = QtWidgets.QCheckBox('Match plot to app theme (dark)')
        self.plot_match_theme.setToolTip(
            'When on and the app theme is dark, the on-screen plots use a dark '
            'background too. Saved/exported images always stay white for clean, '
            'print-parity figures — this only affects what you see on screen.')
        form.addRow('Plot theme:', self.plot_match_theme)
        self.ui_density = QtWidgets.QComboBox()
        self.ui_density.addItem('Compact', 'compact')
        self.ui_density.addItem('Comfortable', 'comfortable')
        self.ui_density.addItem('Large', 'large')
        self.ui_density.setToolTip(
            'Application-wide text size / density. "Large" aids readability; "Compact" '
            'fits more on screen. Applied app-wide and persists across restarts.')
        form.addRow('UI density:', self.ui_density)
        self.show_tooltips = QtWidgets.QCheckBox('Show tooltips on hover')
        self.show_tooltips.setToolTip(
            'Passive hover tooltips throughout the app. The "?" help buttons still '
            'work on click when this is off.')
        form.addRow('Tooltips:', self.show_tooltips)
        self.reopen_last_session = QtWidgets.QCheckBox('Reopen last session on startup')
        self.reopen_last_session.setToolTip(
            'When on, the current workspace is auto-saved when you close the program '
            'and reopened the next time you launch it. If the saved session is missing '
            'or damaged, the program starts empty. If it was saved by a different '
            'version it cannot be opened, and you are asked at startup whether to keep '
            'that file (nothing is auto-saved over it) or let this version replace it. '
            'Off by default.')
        form.addRow('Session:', self.reopen_last_session)
        self.confirm_run_replaces_pick = QtWidgets.QCheckBox(
            'Ask before a Run replaces a picked value')
        self.confirm_run_replaces_pick.setToolTip(
            'An explicit Run replaces a value you picked in Cross-Sample, and the '
            'run always reports the replacement afterwards. Turn this on to be '
            'asked first instead; Cancel keeps the pick and runs nothing.')
        confirm_help = HelpBadge(
            'Whether pressing Run asks before replacing a value you explicitly '
            'picked in Cross-Sample.',
            bullets=[
                'A <b>picked</b> value is replaced by the two runs that write '
                'sample values — a Zimm/Berry run (picked Rg/Mw/A₂) and replicate '
                'averaging (picked Rh); pressing Run is a deliberate act. Other '
                'runs, and every passive refresh, leave picks alone.',
                '<b>Off</b> (default) — the run proceeds, and its summary / status '
                'line reports what it replaced. That report always appears; it is '
                'not part of this setting.',
                '<b>On</b> — a confirm names the picked value(s) first; Cancel '
                'keeps the pick and runs nothing.',
                'Hand-entered values are never replaced by any run, with or '
                'without this setting.',
            ])
        confirm_row = QtWidgets.QWidget()
        confirm_lay = QtWidgets.QHBoxLayout(confirm_row)
        confirm_lay.setContentsMargins(0, 0, 0, 0)
        confirm_lay.addWidget(self.confirm_run_replaces_pick)
        confirm_lay.addWidget(confirm_help)
        confirm_lay.addStretch(1)
        form.addRow('Runs:', confirm_row)

        outer.addWidget(host)

        btns = QtWidgets.QHBoxLayout()
        apply_btn = QtWidgets.QPushButton('Update')
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
        # `color` froze to the build-time palette → unreadable light-gray on the light
        # theme). Title Case + bold; no all-caps (casing per style guide §9).
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
            lcurve_alpha_min=s.lcurve_alpha_min, lcurve_alpha_max=s.lcurve_alpha_max,
            contin_alpha_method=s.contin_alpha_method,
            contin_ftest_prob_reject=s.contin_ftest_prob_reject)
        _seed_combo(self.cumulant_method, s.cumulant_method, use_data=True)
        self.skip_channels.setValue(s.skip_initial_channels)
        _seed_combo(self.bh_dat_baseline, s.brookhaven_dat_baseline, use_data=True)
        _seed_combo(self.geometry, s.standard_geometry, use_data=False)
        self.qrg_max.setValue(s.guinier_qrg_max)
        # use_data: the picker puts the canonical name on itemData and treats the label
        # as cosmetic, and its section headers carry no data — so matching on data can
        # never land on one. find: the same caseless rule the Data tab and the Explorer
        # use, so this field does not become a second way to match a solvent name.
        _seed_combo(self.default_solvent, s.default_solvent, use_data=True,
                    find=index_of_solvent)
        _seed_combo(self.se_estimator, s.se_estimator, use_data=True)
        # _seed_combo, not setCurrentText: a stored value the list no longer offers
        # (a hand-edited settings.json, a unit later dropped -- app/units.py records
        # viscosity's 'cP') must round-trip through an untouched Apply, not be
        # silently replaced by the combo's first row. The plot layer already falls
        # back to its default for a unit it does not know (`plots.set_plot_units`).
        for q, combo in self.plot_unit_combos.items():
            _seed_combo(combo, s.plot_units.get(q, U.default_unit(q)), use_data=False)
        self.no_unc_sig_figs.setValue(s.no_uncertainty_sig_figs)
        # last_session_path / last_export_dir / last_open_dir are managed by the shell
        # (autosave on exit; the file dialogs), not edited here. _collect() reads them
        # LIVE from controller.settings rather than retaining a snapshot here: the file
        # dialogs mutate the settings object mid-session, and an Apply collecting a
        # construction-time snapshot would silently revert them (review finding).
        _seed_combo(self.theme, s.theme, use_data=False)
        _seed_combo(self.palette, s.plot_palette, use_data=False)
        self.plot_match_theme.setChecked(s.plot_match_theme)
        _seed_combo(self.ui_density, s.ui_density, use_data=True)
        self.show_tooltips.setChecked(s.show_tooltips)
        self.reopen_last_session.setChecked(s.reopen_last_session)
        self.confirm_run_replaces_pick.setChecked(s.confirm_run_replaces_pick)

    def _collect(self) -> SettingsState:
        return SettingsState(
            # Relocated DLS seeds: carried through unchanged (edited via the DLS
            # sub-tabs' per-run controls, not here).
            **self._retained_dls,
            cumulant_method=self.cumulant_method.currentData(),
            skip_initial_channels=self.skip_channels.value(),
            brookhaven_dat_baseline=self.bh_dat_baseline.currentData(),
            standard_geometry=self.geometry.currentText(),
            guinier_qrg_max=self.qrg_max.value(),
            # `or ''` because the field is typed `str`: currentData() is None on a blank
            # or header row, and writing None into a str field puts a value in
            # settings.json that no reader is prepared for.
            default_solvent=self.default_solvent.currentData() or '',
            se_estimator=self.se_estimator.currentData(),
            plot_units={q: combo.currentText()
                        for q, combo in self.plot_unit_combos.items()},
            no_uncertainty_sig_figs=self.no_unc_sig_figs.value(),
            theme=self.theme.currentText(),
            ui_density=self.ui_density.currentData(),
            plot_palette=self.palette.currentText(),
            plot_match_theme=self.plot_match_theme.isChecked(),
            show_tooltips=self.show_tooltips.isChecked(),
            reopen_last_session=self.reopen_last_session.isChecked(),
            confirm_run_replaces_pick=self.confirm_run_replaces_pick.isChecked(),
            last_session_path=self.controller.settings.last_session_path,
            last_export_dir=self.controller.settings.last_export_dir,
            last_open_dir=self.controller.settings.last_open_dir,
        )

    # ----------------------------------------------------------- actions ---
    @QtCore.Slot()
    def _apply(self, *, force: bool = False) -> bool:
        """Apply the edited settings. Returns True if applied, False if it was
        aborted (busy, or a stale-guard prompt canceled) — `_restore` uses this to
        fully revert the widgets on cancel.

        A no-change Apply is a no-op (no persist, no `applied` cascade) unless
        `force` is set: `_restore` forces it, because the shell's reseed of the DLS
        per-run controls rides the `applied` signal and those controls are not part of
        the compared `SettingsState` — equality of the object is not equality of the
        effect (owner-ruled 2026-08-28; the constraint's home is CLAUDE.md's locked
        decision "Settings seed, never override").

        `force` is KEYWORD-ONLY on purpose: this method is a slot on the Update
        button's `clicked(bool checked)`, and a positional first parameter takes that
        payload — measured on PySide6 6.11.1, `@QtCore.Slot()` does not suppress it, so
        a checkable button would deliver `force=True` on every click. Keyword-only
        removes the gap instead of policing it: Qt cannot fill it."""
        # Settings values seed the analysis defaults a background run reads
        # mid-flight — applying under it could change its numbers.
        if runner().is_busy:
            busy_notice(self)
            return False
        new = self._collect()
        if not force and new == self.controller.settings:
            # Nothing changed: no persist, no `applied` cascade (the shell's re-theme,
            # re-seed and every-tab redraw). Same shape as Controller.is_dirty(): the
            # collected state against the applied one. A fresh state's first Apply
            # still goes through: its `plot_units` is `{}` until _collect fills the
            # eight axis units, a real change to persist once.
            return True
        # Two independent switches (cumulant method, SE estimator) each make some
        # existing results stale and offer a clear-and-warn. Collect BOTH confirmations
        # up front and only mutate after every prompt is confirmed — otherwise canceling
        # the second prompt would leave the first's clear already done (results gone) but
        # the Apply aborted, a dropdown/setting mismatch. Any Cancel reverts every dropdown
        # and applies nothing.
        old_method = self.controller.settings.cumulant_method
        old_estimator = self.controller.settings.se_estimator
        clear_cumulant = False
        clear_se = False

        if new.cumulant_method != old_method:
            n = self.controller.cumulant_dependent_result_count()
            # The picked count gates the prompt too: a dangling pick can exist while
            # the row/cache counts are zero (its row removed by a declined re-run, or
            # the cache emptied by a reload), and a clear that destroys a pick must
            # never run unconfirmed (SampleResult docstring: proceed or back out).
            p = self.controller.cumulant_dependent_picked_count()
            if n > 0 or p > 0:
                picked_clause = (f", plus {p} explicitly picked Rh value(s) that "
                                 f"came from them" if p else "")
                resp = QtWidgets.QMessageBox.question(
                    self, 'Switch cumulant method?',
                    f"Switching the cumulant method will clear {n} existing "
                    f"cumulant-based result(s) (cumulant fits, Γ vs q², D vs c, and "
                    f"replicate averages){picked_clause}. A sample Rh taken from a "
                    f"cleared result goes with it, so ρ = Rg/Rh is blank until you "
                    f"re-run. Distributions, SLS, and any hand-entered Rh are "
                    f"kept.\n\nContinue?",
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.Cancel,
                    QtWidgets.QMessageBox.StandardButton.Cancel)
                if resp != QtWidgets.QMessageBox.StandardButton.Yes:
                    self._revert_dropdowns(old_method, old_estimator)
                    return False
                clear_cumulant = True

        if new.se_estimator != old_estimator:
            n = self.controller.se_dependent_result_count()
            p = self.controller.se_dependent_picked_count()   # gates like the cumulant arm
            if n > 0 or p > 0:
                label = 'classical OLS' if new.se_estimator == 'ols' else 'HC3 (robust)'
                picked_clause = (f", plus {p} explicitly picked value(s) whose ± "
                                 f"is estimator-routed (a pick cannot refresh "
                                 f"its ±)" if p else "")
                resp = QtWidgets.QMessageBox.question(
                    self, 'Switch uncertainty estimator?',
                    f"Switching to {label} will clear {n} existing result(s) that "
                    f"report a regression ± (Zimm/Berry, Debye, Guinier, "
                    f"calibration-free A₂, Γ vs q², and D vs c){picked_clause}, so "
                    f"they recompute under the new estimator. Computed point "
                    f"estimates (Mw, Rg, D, Rh) are unaffected; distributions, "
                    f"cumulants, and any hand-entered values are kept.\n\nContinue?",
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.Cancel,
                    QtWidgets.QMessageBox.StandardButton.Cancel)
                if resp != QtWidgets.QMessageBox.StandardButton.Yes:
                    self._revert_dropdowns(old_method, old_estimator)
                    return False
                clear_se = True

        # All prompts confirmed — now it is safe to clear and apply.
        if clear_cumulant:
            self.controller.clear_cumulant_dependent_results()
        if clear_se:
            self.controller.clear_se_dependent_results()
        self.controller.apply_settings(new)               # persists to settings.json
        self.applied.emit()
        return True

    def _revert_dropdowns(self, cumulant_method: str, se_estimator: str) -> None:
        """Restore both stale-guarded dropdowns to the still-current settings after a
        canceled Apply, so no uncommitted choice lingers in the UI."""
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
        # B7: if a stale-guard prompt cancels the apply, the widgets are still showing
        # the factory values just loaded — fully reload them from the (unchanged)
        # applied settings, not just the two guarded dropdowns, so the UI can't be
        # left displaying defaults that were never applied.
        if not self._apply(force=True):      # force: the reseed rides `applied`
            self._load_from(self.controller.settings)
