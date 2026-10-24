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

from app.settings import SettingsState
from core import units as U
from gui.help import HelpBadge
from gui.solvent_picker import index_of_solvent, populate_solvent_combo
from gui.theme import ThemedLabel
from gui.worker import busy_notice, runner
from gui.distribution_fit_fields import (  # noqa: F401 -- re-exported for importers
    RESIDUAL_WEIGHTING_CHOICES, RESIDUAL_WEIGHTING_TOOLTIP, DISTRIBUTION_SEED_FIELDS,
    ALPHA_SELECTION_CHOICES, DistributionFitFields, _seed_combo)

# Calculation-nuance tooltips for the unit choices whose reach is easy to
# misjudge, both reaching the SLS axes as well as the DLS ones.
_PLOT_UNIT_TIPS = {
    'scattering_q2':
        'Applies to every q² axis: the DLS Γ vs q² and DDLS plots AND the SLS '
        'plots (Rayleigh ratio, Debye, Guinier, form factors, Zimm — whose '
        'printed spacing constant k follows this unit). Fitted results are '
        'unaffected; only axes rescale.',
    'concentration':
        'Applies to the concentration axes: the DLS D vs c plot and the SLS '
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
    ('molar_mass', 'Molecular weight M<sub>w</sub>'),
]

# The clear-and-warn prompt's words for the cumulant residual-weighting policy tokens; an
# unrecognised (hand-edited) token is named as itself.
_WEIGHTING_PROMPT_LABELS = {'sourced': 'sourced', 'uniform': 'unweighted'}


def _weighting_label(token) -> str:
    return _WEIGHTING_PROMPT_LABELS.get(token, str(token))


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

        # The cumulant order and the distribution seeds (Rh grid, CONTIN α) also have
        # per-run controls ON the DLS Correlogram and Distribution sub-tabs and the Mode
        # Map, seeded from these same SettingsState fields. The widgets here edit the
        # seeds, which are also what the paths with no per-run control run -- replicate
        # averaging above all.

        form.addRow(self._header('DLS Analysis Defaults'))
        self.cumulant_method = QtWidgets.QComboBox()
        self.cumulant_method.addItem('Nonlinear', 'nonlinear')
        self.cumulant_method.addItem('Linear', 'linear')
        self.cumulant_method.setToolTip(
            'Cumulant fitting method of every cumulant-based DLS analysis.\n'
            '• Nonlinear (default): fits g₂−1 directly with a floating baseline; more '
            'robust to a baseline slightly off unity, to the number of points fitted and '
            'to poor starting values.\n'
            '• Linear: polynomial fit of ln(g₂−1).\n'
            'Switching clears cumulant-based results after you confirm. See '
            'Theory-and-Equations-Guide §3.2.3 and §3.2.5.')
        form.addRow('Cumulant method:', self.cumulant_method)
        self.cumulant_order = QtWidgets.QSpinBox()
        self.cumulant_order.setRange(1, 4)
        self.cumulant_order.setToolTip(
            'Cumulant expansion order of every fit with no order box of its own (Γ vs q², '
            'D vs c, DDLS, replicate averaging); it also seeds the Correlogram sub-tab’s '
            'order box. Order 1 fits a single decay rate and gives no PDI.\n'
            'Changing it clears those results after you confirm; the Correlogram '
            'sub-tab’s fits keep their own order.')
        form.addRow('Cumulant order:', self.cumulant_order)
        self.cumulant_weighting = QtWidgets.QComboBox()
        self.cumulant_weighting.addItem('Sourced variance (default)', 'sourced')
        self.cumulant_weighting.addItem('Unweighted', 'uniform')
        self.cumulant_weighting.setToolTip(
            'Residual weighting of every cumulant fit (per-measurement cumulant, Γ vs q², '
            'D vs c, replicate averaging, DDLS and the Cross-Sample size candidates).\n'
            '• Sourced variance (default): each method’s own sourced weight.\n'
            '• Unweighted: plain least squares, for comparison with a fit made without a '
            'noise model.\n'
            'Switching clears cumulant-based results after you confirm. See '
            'Theory-and-Equations-Guide §3.2.8 and §3.2.11.')
        form.addRow('Cumulant residual weighting:', self.cumulant_weighting)
        # The distribution seeds: the same field group the Distribution sub-tab and the
        # Mode Map hold per run. Its residual weighting sits here, beside the cumulant
        # family's, so the two labels are read against each other; the rest of the group
        # has its own section below.
        self.distribution_fields = DistributionFitFields(self, seeds=True)
        self.distribution_fields.add_rows(
            form, keys=('weighting',),
            labels={'weighting': 'Distribution residual weighting:'})
        self.skip_channels = QtWidgets.QSpinBox()
        self.skip_channels.setRange(0, 50)
        self.skip_channels.setToolTip(
            'Leading correlator channels (the shortest lags) dropped from every DLS fit, '
            'for afterpulsing and dead-time artifacts; a large skip can erase a real '
            'small-particle mode. Changing it clears every DLS result after you confirm, '
            'but keeps recorded mode sets, their results and the mode families you picked. '
            'See Theory-and-Equations-Guide §3.1.1.')
        form.addRow('Skip initial lag channels:', self.skip_channels)
        self.cumulant_cutoff = QtWidgets.QDoubleSpinBox()
        # Widened to the ENGINE's contract, not a comfortable subset of it: `_FIELD_RANGES`
        # and `fit_cumulants` accept any finite value in the OPEN interval (0, 1). A
        # narrower box does not merely restrict typing -- `_load_from` calls setValue() on
        # whatever settings.json holds and `_collect` reads it back, so a legal 0.0005 was
        # clamped to 0.001 and PERSISTED, changing an analysis parameter because the user
        # opened the tab. Anything finer than 1e-6 is preserved by `_cumulant_cutoff_value`.
        self.cumulant_cutoff.setDecimals(6)
        self.cumulant_cutoff.setRange(1e-6, 0.999999)
        self.cumulant_cutoff.setSingleStep(0.01)
        self.cumulant_cutoff.setToolTip(
            'Amplitude at which the Linear fit’s range stops, as a fraction of the '
            'correlogram peak. The DLS standard prescribes 0.01; this program defaults '
            'to 0.1, a benchmarked deviation — see the Theory-and-Equations-Guide §3.2.3 '
            'for the measurement and for which of the standard’s companion clauses are '
            'implemented.')
        cutoff_help = HelpBadge(
            'The Linear fit reads a contiguous run of channels around the correlogram’s '
            'peak, stopping at the first whose amplitude falls to this fraction of the '
            'peak.',
            bullets=[
                '<b>Lower</b> (towards the DLS standard’s 0.01, '
                'Theory-and-Equations-Guide §3.2.3) fits further down the decay; the PDI '
                'can fall and clear the polydispersity warning (35 of 190 measurements of '
                'the program’s test corpus, never the reverse).',
                '<b>Higher</b> keeps only the strongest part of the decay.',
                'Under <b>Nonlinear</b>, which fits every live channel, it still sets the '
                'range of the reported RMS error and the fit’s starting guess.',
                'Changing it clears cumulant-based results after you confirm and marks a '
                'cumulant fit on screen as out of date.',
            ])
        cutoff_row = QtWidgets.QWidget()
        cutoff_lay = QtWidgets.QHBoxLayout(cutoff_row)
        cutoff_lay.setContentsMargins(0, 0, 0, 0)
        cutoff_lay.addWidget(self.cumulant_cutoff, 1)   # stretch like a badge-less row
        cutoff_lay.addWidget(cutoff_help)
        form.addRow('Cumulant fit cutoff:', cutoff_row)

        dist_head = QtWidgets.QWidget()
        dist_head_lay = QtWidgets.QHBoxLayout(dist_head)
        dist_head_lay.setContentsMargins(0, 0, 0, 0)
        dist_head_lay.addWidget(self._header('DLS Distribution Defaults'))
        dist_head_lay.addWidget(HelpBadge(
            'The starting values of the distribution fits (CONTIN, NNLS, lognormal).',
            bullets=[
                'They seed the <b>Distribution</b> sub-tab’s and the <b>Mode Map</b>’s '
                'own fields, which you can still change per run there.',
                'Replicate averaging of a distribution method (the sidebar’s '
                '<b>Average derived results…</b>) has no fields of its own: it runs on '
                'these values unless its prompt copies the Distribution sub-tab’s.',
                'The Rh grid applies to every method; the α rows are CONTIN’s alone.',
                'Changing a value here does not touch a result already computed.',
            ]))
        dist_head_lay.addStretch(1)
        form.addRow(dist_head)
        self.distribution_fields.add_rows(
            form, keys=('rh_grid', 'alpha_method', 'alpha_range', 'gcv_gamma', 'ftest'))

        form.addRow(self._header('File Loading'))
        self.bh_dat_baseline = QtWidgets.QComboBox()
        self.bh_dat_baseline.addItem('Calculated', 'calculated')
        self.bh_dat_baseline.addItem('Measured', 'measured')
        self.bh_dat_baseline.setToolTip(
            'Baseline used to normalize a Brookhaven raw ASCII .dat correlogram, '
            'whose channels hold raw counts: g₂−1 = counts/B − 1. See '
            'Theory-and-Equations-Guide §3.1.2.')
        dat_help = HelpBadge(
            'Which of a Brookhaven .dat file’s two baselines turns its raw counts into a '
            'correlogram.',
            bullets=[
                '<b>Calculated</b> (default) — formed from the run totals (counts A × '
                'counts B, over the number of samples), so it uses the whole measurement.',
                '<b>Measured</b> — the mean of the baseline channels the acquiring '
                'software set aside, as Particle Explorer reports; it needs those '
                'channels past the decay.',
                'A file that names its own baseline overrides this setting.',
                'A few tenths of a percent between the two moves the Linear fit’s size by '
                'a few percent and leaves the default fits unchanged (User-Manual §12.3).',
                'A large difference between them signals dust or slow drift, whichever '
                'you pick.',
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
            'VH = depolarized, VU = a vertical laser with no analyzer. '
            'See Theory-and-Equations-Guide §1.6.')
        form.addRow('Standard geometry:', self.geometry)
        self.qrg_max = QtWidgets.QDoubleSpinBox()
        self.qrg_max.setRange(0.5, 3.0)
        self.qrg_max.setSingleStep(0.1)
        self.qrg_max.setToolTip(
            'Maximum qRg for every small-qRg fit: the Zimm, Debye, Guinier and Amplitude '
            'Guinier Rg are flagged above it, and the Zimm, Debye and Guinier Mw with them; '
            'a Berry fit is not, and its Rg and Mw labels say they are past the Zimm limit. '
            'See Theory-and-Equations-Guide §4.4 and §4.6.')
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
            'Which standard error the program reports for its straight-line and '
            'multilinear fits.',
            bullets=[
                '<b>HC3 (robust)</b> — the default; it never under-reports when point '
                'precision varies across angles or concentrations.',
                '<b>Classical OLS</b> — the textbook SE, only to match classical '
                'software, a published table or a spreadsheet like-for-like; it can '
                'under-report.',
                'The choice is global, recorded on each result and written into the '
                'export, where an OLS ± reads “± from classical OLS, not HC3”. Point '
                'estimates (Mw, Rg, D, Rh) are unaffected.',
                'Detail and formulas: Theory-and-Equations-Guide §6.1.',
            ])
        est_row = QtWidgets.QWidget()
        est_lay = QtWidgets.QHBoxLayout(est_row)
        est_lay.setContentsMargins(0, 0, 0, 0)
        est_lay.addWidget(self.se_estimator, 1)      # stretch like a badge-less row
        est_lay.addWidget(est_help)
        form.addRow('Regression SE estimator:', est_row)

        # The synthetic-generator (β/noise/points) and intensity-trace (outlier k,
        # running-average window) defaults are plain session fields on their own
        # Utilities tabs (Synthetic generator / Traces), not global settings.

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
            'Significant figures for every result that has no honest uncertainty. A '
            'value that HAS a ± is rounded to the place its ± supports, since more '
            'digits would claim a precision the data lack; only a ± so large it would '
            'round the value to zero leaves the value at this precision, shown beside '
            'its ±. See Theory-and-Equations-Guide §6.')
        sig_help = HelpBadge(
            'How many significant figures to show for a result that carries no ±.',
            bullets=[
                'It sets every such value alike: a single-correlogram fit, an NNLS/CONTIN '
                'peak, and every derived number without a ± (Γ, PDI, R², qRg, …).',
                'A value that <b>has</b> a ± is shown to the place its ± supports, unless '
                'the ± would round it to zero: then it takes this precision, beside its ±.',
                'Default 3. Lowering it gives a more conservative look and drops the '
                'trailing digits of everything without a ±, R² and qRg included.',
                'The program never fabricates a ± to show more digits.',
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
            'When on, the workspace is kept when you close the program and reopened at '
            'the next launch; a missing or damaged file starts the program empty. A file '
            'from a different version cannot be reopened: you are asked at startup '
            'whether to keep it, with nothing written over it, or let this version '
            'replace it. Off by default.')
        form.addRow('Session:', self.reopen_last_session)
        self.confirm_run_replaces_pick = QtWidgets.QCheckBox(
            'Ask before a Run replaces a picked value')
        self.confirm_run_replaces_pick.setToolTip(
            'An explicit Run replaces a value you picked in Cross-Sample, and the '
            'run always reports the replacement afterwards.')
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

        # The form scrolls and Update / Restore defaults are pinned under it, so the
        # form's length can never carry them out of view (the pinned primary action of
        # the GUI style guide's §3 column template).
        form_scroll = QtWidgets.QScrollArea()
        form_scroll.setWidgetResizable(True)
        form_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        form_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        form_scroll.setWidget(host)
        outer.addWidget(form_scroll, 1)

        btns = QtWidgets.QHBoxLayout()
        apply_btn = QtWidgets.QPushButton('Update')
        apply_btn.clicked.connect(self._apply)
        reset_btn = QtWidgets.QPushButton('Restore defaults')
        reset_btn.clicked.connect(self._restore)
        btns.addWidget(apply_btn)
        btns.addWidget(reset_btn)
        btns.addStretch(1)
        outer.addLayout(btns)

    @staticmethod
    def _header(text: str) -> QtWidgets.QLabel:
        # ThemedLabel so the header follows the theme (a plain stylesheet without a
        # `color` froze to the build-time palette → unreadable light-gray on the light
        # theme). Title Case + bold; no all-caps (casing per style guide §9).
        return ThemedLabel(text, role='header', bold=True, extra='margin-top:8px;')

    # ----------------------------------------------------------- transfer ---
    def _load_from(self, s: SettingsState) -> None:
        _seed_combo(self.cumulant_method, s.cumulant_method, use_data=True)
        self.cumulant_order.setValue(s.cumulant_order)
        _seed_combo(self.cumulant_weighting, s.cumulant_weighting, use_data=True)
        # The group keeps each exact seed, so a value its widgets cannot show round-trips
        # through an untouched Update unchanged.
        self.distribution_fields.seed_from_settings(s)
        self.skip_channels.setValue(s.skip_initial_channels)
        # Remember what was LOADED so `_collect` can tell "the user chose this" from "the
        # spin box could not represent it". Without that, visiting the tab rewrites the
        # setting to whatever the widget rounded it to.
        self._loaded_cumulant_cutoff = float(s.cumulant_fit_cutoff)
        self.cumulant_cutoff.setValue(s.cumulant_fit_cutoff)
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
        # (a hand-edited settings.json, a unit later dropped -- core/units.py records
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

    def _cumulant_cutoff_value(self) -> float:
        """The cutoff to persist: the box's value, unless it never represented the load.

        A `QDoubleSpinBox` clamps and rounds on `setValue`, so a legal settings.json value
        outside the widget's grid (finer than its decimals, or outside its range) comes
        back changed. Writing that back would edit an analysis parameter because the user
        opened the tab. So: if the box still reads exactly what the load rounded it TO, the
        user did not touch it and the ORIGINAL value is persisted unchanged; once they move
        it, their choice wins. The engine's contract is the open interval (0, 1) and the
        widget's grid is a subset of it -- this closes the gap rather than papering it.
        """
        loaded = getattr(self, '_loaded_cumulant_cutoff', None)
        if loaded is None:
            return float(self.cumulant_cutoff.value())
        # What `setValue(loaded)` produced, recomputed the same way the widget did.
        as_shown = float(self.cumulant_cutoff.valueFromText(
            self.cumulant_cutoff.textFromValue(loaded)))
        if float(self.cumulant_cutoff.value()) == as_shown:
            return float(loaded)
        return float(self.cumulant_cutoff.value())

    def _collect(self) -> SettingsState:
        return SettingsState(
            **self.distribution_fields.values(),
            cumulant_method=self.cumulant_method.currentData(),
            cumulant_order=self.cumulant_order.value(),
            cumulant_weighting=self.cumulant_weighting.currentData(),
            skip_initial_channels=self.skip_channels.value(),
            cumulant_fit_cutoff=self._cumulant_cutoff_value(),
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
        # A CONTIN α the engine would refuse is never saved: it would seed the per-run
        # fields and the replicate average, which would then fail far from here.
        problem = self.distribution_fields.alpha_problem()
        if problem:
            QtWidgets.QMessageBox.warning(
                self, 'Settings not applied',
                f'{problem}\n\nNothing was applied; correct the field and press Update '
                f'again.')
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
        old = self.controller.settings
        old_method = old.cumulant_method
        old_estimator = old.se_estimator
        old_cutoff = old.cumulant_fit_cutoff
        old_weighting = old.cumulant_weighting
        clear_cumulant = False
        clear_skip = False
        clear_se = False

        # The CUTOFF joins the method's arm rather than getting one of its own: both
        # change which channels the fit regresses, both invalidate exactly the same result
        # set, and a user who changes both at once should be asked once. It had no arm at
        # all -- so the setting shipped able to move a promoted Rh while leaving it on the
        # sample, still feeding rho = Rg/Rh in Cross-Sample.
        cumulant_input_changed = (new.cumulant_method != old_method
                                  or new.cumulant_fit_cutoff != old_cutoff
                                  or new.cumulant_weighting != old_weighting)
        # The ORDER is asked about in the same prompt but clears less: only the results
        # that fit at the Settings order, never the Correlogram tab's own fits, which ran
        # at its order box. One controller enumeration per scope, so the count asked about
        # is the set cleared (owner-ruled 2026-09-23; the scopes' home is
        # `Controller.clear_cumulant_dependent_results`).
        order_changed = new.cumulant_order != old.cumulant_order
        order_only = order_changed and not cumulant_input_changed
        # The SKIPPED CHANNELS feed every DLS fit, so their scope is every DLS-derived
        # result (owner-ruled 2026-09-23; its home is
        # `Controller.clear_skip_dependent_results`). It contains the method and order
        # scopes, so a skip change beside another cumulant input still asks once.
        skip_changed = new.skip_initial_channels != old.skip_initial_channels

        if skip_changed or cumulant_input_changed or order_changed:
            # The picked count gates the prompt too: a dangling pick can exist while
            # the row/cache counts are zero (its row removed by a declined re-run, or
            # the cache emptied by a reload), and a clear that destroys a pick must
            # never run unconfirmed (SampleResult docstring: proceed or back out).
            if skip_changed:
                n = self.controller.skip_dependent_result_count()
                p = self.controller.skip_dependent_picked_count()
            else:
                n = self.controller.cumulant_dependent_result_count(order_only=order_only)
                p = self.controller.cumulant_dependent_picked_count(order_only=order_only)
            if n > 0 or p > 0:
                picked_clause = (f", plus {p} explicitly picked Rh value(s) that "
                                 f"came from them" if p else "")
                # Name whichever input actually moved, so the prompt is about the change
                # the user made rather than a generic one.
                changed = []
                if new.cumulant_method != old_method:
                    changed.append('cumulant method')
                if new.cumulant_fit_cutoff != old_cutoff:
                    changed.append(f'fit cutoff ({old_cutoff:g} → '
                                   f'{new.cumulant_fit_cutoff:g})')
                if new.cumulant_weighting != old_weighting:
                    changed.append(f'residual weighting ({_weighting_label(old_weighting)} '
                                   f'→ {_weighting_label(new.cumulant_weighting)})')
                if new.skip_initial_channels != old.skip_initial_channels:
                    changed.append(f'skipped initial channels '
                                   f'({old.skip_initial_channels} → '
                                   f'{new.skip_initial_channels})')
                if order_changed:
                    changed.append(f'cumulant order ({old.cumulant_order} → '
                                   f'{new.cumulant_order})')
                kept = 'Distributions, SLS, and any hand-entered Rh are kept.'
                if skip_changed:
                    scope = (f"will clear {n} existing DLS result(s) (every "
                             f"per-measurement fit, Γ vs q², D vs c, DDLS, the Mode Map "
                             f"with its mode-family fits, and replicate averages)"
                             f"{picked_clause}")
                    kept = ('Recorded mode sets and their results, the mode families you '
                            'picked on the Mode Map, SLS, and any hand-entered Rh are '
                            'kept.')
                elif order_only:
                    scope = (f"will clear {n} existing result(s) fitted at the Settings "
                             f"order (Γ vs q², D vs c, DDLS and replicate averages, "
                             f"including an average run with the Correlogram tab’s order "
                             f"copied in){picked_clause}. The Correlogram tab’s own "
                             f"cumulant fits ran at its own order and are kept")
                else:
                    scope = (f"will clear {n} existing cumulant-based result(s) "
                             f"(cumulant fits, Γ vs q², D vs c, DDLS and replicate "
                             f"averages){picked_clause}")
                resp = QtWidgets.QMessageBox.question(
                    self, 'Change the cumulant fit inputs?',
                    f"Changing the {' and '.join(changed)} {scope}. A sample Rh taken "
                    f"from a cleared result goes with it, so ρ = Rg/Rh is blank until "
                    f"you re-run. {kept}\n\nContinue?",
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.Cancel,
                    QtWidgets.QMessageBox.StandardButton.Cancel)
                if resp != QtWidgets.QMessageBox.StandardButton.Yes:
                    self._revert_dropdowns(old)
                    return False
                if skip_changed:
                    clear_skip = True
                else:
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
                    f"calibration-free A₂, Γ vs q², D vs c and DDLS){picked_clause}, so "
                    f"they recompute under the new estimator. Computed point "
                    f"estimates (Mw, Rg, D, Rh) are unaffected; distributions, "
                    f"cumulants, and any hand-entered values are kept.\n\nContinue?",
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.Cancel,
                    QtWidgets.QMessageBox.StandardButton.Cancel)
                if resp != QtWidgets.QMessageBox.StandardButton.Yes:
                    self._revert_dropdowns(old)
                    return False
                clear_se = True

        # All prompts confirmed — now it is safe to clear and apply.
        if clear_skip:
            self.controller.clear_skip_dependent_results()
        if clear_cumulant:
            self.controller.clear_cumulant_dependent_results(order_only=order_only)
        if clear_se:
            self.controller.clear_se_dependent_results()
        self.controller.apply_settings(new)               # persists to settings.json
        self.applied.emit()
        return True

    def _revert_dropdowns(self, old: SettingsState) -> None:
        """Restore every stale-guarded control to the still-current settings `old` after
        a canceled Apply, so no uncommitted choice lingers in the UI.

        The cutoff, order and skipped channels are spin boxes rather than dropdowns, but
        they guard the same result set and revert with them: leaving one showing a value
        the Apply refused is exactly the control/setting mismatch this helper exists to
        prevent."""
        j = self.cumulant_method.findData(old.cumulant_method)
        self.cumulant_method.setCurrentIndex(j if j >= 0 else 0)
        # `_seed_combo`, not findData-or-0: a hand-edited value the list does not offer
        # must revert to itself, not to 'sourced'.
        _seed_combo(self.cumulant_weighting, old.cumulant_weighting, use_data=True)
        k = self.se_estimator.findData(old.se_estimator)
        self.se_estimator.setCurrentIndex(k if k >= 0 else 0)
        self._loaded_cumulant_cutoff = float(old.cumulant_fit_cutoff)
        self.cumulant_cutoff.setValue(old.cumulant_fit_cutoff)
        self.cumulant_order.setValue(old.cumulant_order)
        self.skip_channels.setValue(old.skip_initial_channels)

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
