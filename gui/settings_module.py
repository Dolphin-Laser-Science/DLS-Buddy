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

import math

from PySide6 import QtCore, QtWidgets

from app import units as U
from app.controller import distribution_weighting_request
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

# The clear-and-warn prompt's words for the cumulant residual-weighting policy tokens; an
# unrecognised (hand-edited) token is named as itself.
_WEIGHTING_PROMPT_LABELS = {'sourced': 'sourced', 'uniform': 'unweighted'}


def _weighting_label(token) -> str:
    return _WEIGHTING_PROMPT_LABELS.get(token, str(token))


# The residual-weighting forms the CONTIN / NNLS / lognormal distribution fits offer, as
# (label, policy token) pairs. ONE roster: the seed combo below and the DLS Distribution
# tab's per-run combo are both populated from it, so the same choice cannot be worded two
# ways (style guide §9). The tokens are the keys of the controller's
# DISTRIBUTION_WEIGHTING_POLICIES, which maps each to an engine request; a further form is
# one row here plus one policy there, and nothing reads this roster by position or length.
RESIDUAL_WEIGHTING_CHOICES = (
    ('Sourced variance (default)', 'sourced'),
    ('g₁² (older form)', 'g1-squared'),
)

# The calculation-nuance tooltip both those combos carry, in one place for the same reason
# the roster is: two surfaces for one choice must say the same thing about it.
RESIDUAL_WEIGHTING_TOOLTIP = (
    'Which residual weight the distribution fits (CONTIN, NNLS, lognormal) measure their '
    'fit error in.\n'
    '• Sourced variance (default): weights each channel by the variance a correlator '
    'actually has at that lag, and reads the weight’s signal factor from a first-pass '
    'fit — so each fit runs twice and takes about twice as long.\n'
    '• g₁² (older form): takes the noise as equal on every channel. Choose it to '
    'reproduce a distribution reported before this version.\n'
    'Without a committed mean count rate the sourced weight uses the '
    'infinite-count-rate limit of its variance model; committing one adds the '
    'photon-counting terms to the weight. See the Theory-and-Equations-Guide '
    '(distribution methods).'
)


#: The distribution fits' seed fields, in `SettingsState`'s own names. `DistributionFitFields`
#: carries values under these names on every surface, so a Settings collect, a reseed from
#: Settings and a copy between two tabs are one operation with no renaming step between them.
DISTRIBUTION_SEED_FIELDS = (
    'rh_grid_min_nm', 'rh_grid_max_nm', 'rh_grid_points', 'distribution_weighting',
    'lcurve_alpha_min', 'lcurve_alpha_max', 'contin_alpha_method',
    'contin_ftest_prob_reject', 'contin_gcv_gamma')

#: The CONTIN α selectors, as (label, token) pairs -- one roster for every surface that
#: offers the choice.
ALPHA_SELECTION_CHOICES = (
    ('GCV (generalized cross-validation)', 'gcv'),
    ('L-curve corner', 'lcurve'),
    ('F-test (probability to reject)', 'ftest'),
)


class DistributionFitFields(QtCore.QObject):
    """The distribution fits' input fields -- the Rh grid, the residual weighting, and
    CONTIN's α selector, α range, robust-GCV γ and F-test level -- built in ONE place and
    placed by each surface that holds them: the Settings tab (as the seeds), and the DLS
    Distribution sub-tab and Mode Map (as per-run fields seeded from those seeds). One
    set of widgets, ranges and tooltips, so the surfaces cannot drift apart.

    Values travel under the `SettingsState` field names (`DISTRIBUTION_SEED_FIELDS`):
    `values` reads them, `seed` writes them, and `fit_kwargs` maps them to the engine's
    keywords for one method. Each numeric field keeps the exact value it was seeded with
    until the user moves it -- a spin box clamps and rounds on `setValue`, and a seed
    finer than its grid must run, and round-trip through Settings, unchanged (the Settings
    tab's `_cumulant_cutoff_value` is the same rule).

    ``seeds=True`` builds the Settings tab's copy: a value the roster does not offer (a
    hand-edited settings.json) is kept as an extra entry so an untouched Update writes it
    back, and every row stays shown, since each is a seed in its own right. A per-run copy
    seeds such a token's first row instead -- the controller refuses the token by name on
    any path this widget does not fill -- and shows only the selected α selector's own
    control (owner: show/hide, not gray-out).
    """

    #: Any field moved -- by the user or by `seed`, which emits once per call.
    changed = QtCore.Signal()

    # key -> the default form-row label. `add_rows` places rows in this order.
    _ROW_LABELS = {
        'rh_grid': 'R<sub>h</sub> grid (min / max / pts):',
        'weighting': 'Residual weighting:',
        'alpha_method': 'CONTIN α selection:',
        'alpha_range': 'CONTIN α (min / max):',
        'gcv_gamma': 'GCV robustness γ:',
        'ftest': 'F-test probability to reject:',
    }
    _CONTIN_ROWS = ('alpha_method', 'alpha_range', 'gcv_gamma', 'ftest')

    def __init__(self, parent: QtWidgets.QWidget, *, seeds: bool = False) -> None:
        super().__init__(parent)
        self._seeds = seeds
        self._seeding = False
        self._contin_shown = True
        self._loaded: dict = {}      # field -> the exact value last seeded
        self._shown: dict = {}       # field -> what its widget showed right after seeding
        self._rows: dict = {}        # key -> (label widget or None, field widget)

        self.rh_min = QtWidgets.QDoubleSpinBox()
        self.rh_min.setRange(0.01, 1.0e5); self.rh_min.setDecimals(2)
        self.rh_max = QtWidgets.QDoubleSpinBox()
        self.rh_max.setRange(0.01, 1.0e5); self.rh_max.setDecimals(1)
        self.rh_points = QtWidgets.QSpinBox(); self.rh_points.setRange(10, 1000)
        # A cost hint, not a cap: the range stays 10-1000 (owner-ruled 2026-08-26).
        self.rh_points.setToolTip(
            'Grid resolution of the inversion. CONTIN’s automatic α selection solves a '
            'dense system of this size about 40 times, so the cost climbs steeply with '
            'the point count — roughly half a second at 100 points, ~5 s at 300 and '
            '~30 s at 1000 with the default GCV selector on a typical laptop (the '
            'L-curve corner is cheaper but climbs the same way). The fit runs in the '
            'background, so the window stays usable.')
        self._grid_holder = QtWidgets.QWidget()
        grow = QtWidgets.QHBoxLayout(self._grid_holder); grow.setContentsMargins(0, 0, 0, 0)
        for w in (self.rh_min, self.rh_max, self.rh_points):
            grow.addWidget(w)
        # The residual weighting of the FIT: which noise model its error is measured in.
        # All three methods take it, so it is never hidden with the CONTIN-only rows.
        self.residual_weighting = QtWidgets.QComboBox()
        for label, token in RESIDUAL_WEIGHTING_CHOICES:
            self.residual_weighting.addItem(label, token)
        self.residual_weighting.setToolTip(RESIDUAL_WEIGHTING_TOOLTIP)
        self.alpha_method = QtWidgets.QComboBox()
        for label, token in ALPHA_SELECTION_CHOICES:
            self.alpha_method.addItem(label, token)
        self.alpha_method.setToolTip(
            'How CONTIN chooses its regularization α automatically.\n'
            '• GCV (default): generalized cross-validation, robust by default (γ = 0.1, '
            'built for correlated channel noise); γ = 1 is classical GCV.\n'
            '• L-curve corner: the corner of the fit-vs-size trade-off curve, located by '
            'adaptive pruning of the sampled sweep; insensitive to the α range as long '
            'as the range spans the corner — the sweep-edge notes say when it does not.\n'
            '• F-test: the smoothest solution whose fit is not significantly worse than '
            'the best — for comparison against legacy CONTIN output.\n'
            'See the Theory-and-Equations-Guide (CONTIN).')
        self.alpha_min = QtWidgets.QLineEdit()
        self.alpha_max = QtWidgets.QLineEdit()
        self._alpha_range_holder = QtWidgets.QWidget()
        arow = QtWidgets.QHBoxLayout(self._alpha_range_holder)
        arow.setContentsMargins(0, 0, 0, 0)
        for w in (self.alpha_min, self.alpha_max):
            arow.addWidget(w)
        # The range sits inside the engine's (0, 1] guard, so this control cannot ask for
        # a refused γ.
        self.gcv_gamma = QtWidgets.QDoubleSpinBox()
        self.gcv_gamma.setRange(0.01, 1.00)
        self.gcv_gamma.setSingleStep(0.01)
        self.gcv_gamma.setDecimals(2)
        self.gcv_gamma.setToolTip(
            'Robustness parameter γ of robust GCV. 0.1, the default, is the value its '
            'sources use; 1 is classical GCV. Below 1 the selector weighs the '
            'solution’s variance and smooths more, the repair for correlated channel '
            'noise. See the Theory-and-Equations-Guide (CONTIN, robust GCV).')
        self.ftest_prob = QtWidgets.QDoubleSpinBox()
        self.ftest_prob.setRange(0.01, 0.99)
        self.ftest_prob.setSingleStep(0.05)
        self.ftest_prob.setDecimals(2)
        self.ftest_prob.setToolTip(
            'F-test significance level (default 0.50). A HIGHER value '
            'accepts more fit degradation and so selects a SMOOTHER, more parsimonious '
            'distribution; a LOWER value keeps a rougher, more-detailed one.')
        self._fields = {
            'rh_grid': self._grid_holder, 'weighting': self.residual_weighting,
            'alpha_method': self.alpha_method, 'alpha_range': self._alpha_range_holder,
            'gcv_gamma': self.gcv_gamma, 'ftest': self.ftest_prob}

        for box in (self.rh_min, self.rh_max, self.rh_points, self.gcv_gamma,
                    self.ftest_prob):
            box.valueChanged.connect(self._emit_changed)
        for combo in (self.residual_weighting, self.alpha_method):
            combo.currentIndexChanged.connect(self._emit_changed)
        for edit in (self.alpha_min, self.alpha_max):
            edit.textChanged.connect(self._emit_changed)
            # Refused where it is typed: an α the engine would refuse reverts to the last
            # accepted text when the edit finishes (the Correlogram region fields' house
            # pattern), with the reason shown at the field.
            edit.editingFinished.connect(lambda e=edit: self._check_alpha_edit(e))
        self._accepted_alpha: dict = {}   # edit -> the last text it held that was valid
        self.alpha_method.currentIndexChanged.connect(self._update_row_visibility)

    # ---- placement ----
    def add_rows(self, form: QtWidgets.QFormLayout, keys=None, labels=None) -> None:
        """Add this group's rows for `keys` (default: all, in the group's order) to
        `form`, each labelled from `labels` or the default. A surface may place its rows
        in two batches, e.g. to set the weighting beside another row it reads against."""
        labels = labels or {}
        for key in (keys or self._ROW_LABELS):
            form.addRow(labels.get(key, self._ROW_LABELS[key]), self._fields[key])
            self._rows[key] = (form.labelForField(self._fields[key]), self._fields[key])
        self._update_row_visibility()

    def set_contin_rows_visible(self, visible: bool) -> None:
        """Show or hide CONTIN's own rows -- for a surface running one method at a time,
        where α means nothing to NNLS."""
        self._contin_shown = bool(visible)
        self._update_row_visibility()

    def _set_row_visible(self, key: str, visible: bool) -> None:
        label, field = self._rows.get(key, (None, None))
        if field is not None:
            field.setVisible(visible)
        if label is not None:
            label.setVisible(visible)

    def _update_row_visibility(self) -> None:
        """The α range stays visible under every selector: all three sweep it, and the
        F-test's own reliability notes tell the user to move one of its ends. γ follows
        GCV and the level follows the F-test, on a per-run copy; the Settings copy shows
        both, since each seeds a per-run field whichever selector is the default."""
        method = self.alpha_method.currentData()
        for key in self._CONTIN_ROWS:
            shown = self._contin_shown
            if shown and not self._seeds:
                if key == 'gcv_gamma':
                    shown = method == 'gcv'
                elif key == 'ftest':
                    shown = method == 'ftest'
            self._set_row_visible(key, shown)

    def _emit_changed(self, *_args) -> None:
        if not self._seeding:
            self.changed.emit()

    # ---- values ----
    def _seed_token(self, combo: QtWidgets.QComboBox, token) -> None:
        if self._seeds:
            _seed_combo(combo, token, use_data=True)
        else:
            j = combo.findData(token)
            combo.setCurrentIndex(j if j >= 0 else 0)

    def seed(self, values: dict) -> None:
        """Set every field from `values` (`DISTRIBUTION_SEED_FIELDS` names), remembering
        each exact value so an untouched field reads back unchanged. Emits `changed` once."""
        self._seeding = True
        try:
            for name, box in (('rh_grid_min_nm', self.rh_min),
                              ('rh_grid_max_nm', self.rh_max),
                              ('rh_grid_points', self.rh_points),
                              ('contin_ftest_prob_reject', self.ftest_prob),
                              ('contin_gcv_gamma', self.gcv_gamma)):
                self._loaded[name] = values[name]
                box.setValue(values[name])
                self._shown[name] = box.value()
            for name, edit in (('lcurve_alpha_min', self.alpha_min),
                               ('lcurve_alpha_max', self.alpha_max)):
                self._loaded[name] = float(values[name])
                edit.setText(f'{float(values[name]):g}')
                self._shown[name] = edit.text()
                self._accepted_alpha[edit] = edit.text()
            self._seed_token(self.residual_weighting, values['distribution_weighting'])
            self._seed_token(self.alpha_method, values['contin_alpha_method'])
        finally:
            self._seeding = False
        self._update_row_visibility()
        self.changed.emit()

    def seed_from_settings(self, settings) -> None:
        self.seed({name: getattr(settings, name) for name in DISTRIBUTION_SEED_FIELDS})

    def _number(self, name: str, box):
        """The box's value, unless it still shows what the seed made it show -- then
        the exact seed."""
        if name in self._loaded and box.value() == self._shown.get(name):
            return self._loaded[name]
        return box.value()

    def _alpha(self, name: str, edit: QtWidgets.QLineEdit) -> float:
        """The typed α, or the exact seed while the text is what seeding wrote -- and
        also when the text does not parse, so a half-typed entry never reaches a fit."""
        loaded = self._loaded.get(name)
        text = edit.text().strip()
        if loaded is not None and text == self._shown.get(name):
            return loaded
        try:
            return float(text)
        except ValueError:
            return loaded if loaded is not None else float('nan')

    @staticmethod
    def _alpha_text_problem(text: str) -> str:
        """Why one typed α end cannot run, or '' -- `fit_contin`'s own refusal (finite and
        above zero), stated in the field's words."""
        try:
            value = float(text.strip())
        except ValueError:
            return f'“{text.strip()}” is not a number'
        if not (math.isfinite(value) and value > 0):
            return 'α must be a finite number above zero'
        return ''

    def alpha_problem(self) -> str:
        """Why the CONTIN α range as typed cannot run, or '' when it can: each end finite
        and above zero, and the minimum not above the maximum -- the two refusals
        `fit_contin` makes. ONE predicate for every surface holding this group, checked
        before the Settings tab's Update and before a CONTIN Run, so a value the engine
        refuses is never saved and never run. Read off the TEXT, not `values()`, which
        falls back to the seed for an unparseable entry and would hide it."""
        for label, edit in (('α min', self.alpha_min), ('α max', self.alpha_max)):
            problem = self._alpha_text_problem(edit.text())
            if problem:
                return f'CONTIN {label}: {problem}.'
        if float(self.alpha_min.text()) > float(self.alpha_max.text()):
            return 'CONTIN α min is above α max; lower the minimum or raise the maximum.'
        return ''

    def _check_alpha_edit(self, edit: QtWidgets.QLineEdit) -> None:
        """An α edit finished: keep it if the engine can run it, else put back the last
        accepted text and say why at the field. The order of the two ends is left to the
        Update / Run check, so a range can be widened one end at a time."""
        problem = self._alpha_text_problem(edit.text())
        if not problem:
            self._accepted_alpha[edit] = edit.text()
            return
        previous = self._accepted_alpha.get(edit)
        if previous is not None:
            edit.setText(previous)
        QtWidgets.QToolTip.showText(
            edit.mapToGlobal(QtCore.QPoint(0, edit.height())),
            f'{problem}; kept {previous or "the previous value"}.', edit)

    def values(self) -> dict:
        """Every field's current value under its `SettingsState` name."""
        return {
            'rh_grid_min_nm': float(self._number('rh_grid_min_nm', self.rh_min)),
            'rh_grid_max_nm': float(self._number('rh_grid_max_nm', self.rh_max)),
            'rh_grid_points': int(self._number('rh_grid_points', self.rh_points)),
            'distribution_weighting': self.residual_weighting.currentData(),
            'lcurve_alpha_min': self._alpha('lcurve_alpha_min', self.alpha_min),
            'lcurve_alpha_max': self._alpha('lcurve_alpha_max', self.alpha_max),
            'contin_alpha_method': self.alpha_method.currentData(),
            'contin_ftest_prob_reject': float(
                self._number('contin_ftest_prob_reject', self.ftest_prob)),
            'contin_gcv_gamma': float(self._number('contin_gcv_gamma', self.gcv_gamma)),
        }

    def fit_kwargs(self, method: str) -> dict:
        """The engine kwargs for one `method`: the Rh grid and the residual weighting for
        every method, and for CONTIN its α range, selector, F-test level and γ.

        The weighting is set for EVERY method, outside the CONTIN branch: all three arms
        take it, and a weight that reached only CONTIN would give one measurement two
        analyses depending on which method ran. It is mapped through the controller's own
        policy-to-request function, so a widget and the Settings seed cannot disagree
        about what a token means."""
        v = self.values()
        kw = dict(rh_min_nm=v['rh_grid_min_nm'], rh_max_nm=v['rh_grid_max_nm'],
                  n_grid=v['rh_grid_points'],
                  weighting=distribution_weighting_request(v['distribution_weighting']))
        if method == 'contin':
            kw.update(alpha_min=v['lcurve_alpha_min'], alpha_max=v['lcurve_alpha_max'],
                      alpha_method=v['contin_alpha_method'],
                      ftest_prob_reject=v['contin_ftest_prob_reject'],
                      gcv_gamma=v['contin_gcv_gamma'])
        return kw


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
            'Cumulant fitting method, applied to every cumulant-based DLS analysis '
            '(per-measurement cumulant, Γ vs q², replicate averaging).\n'
            '• Nonlinear (default): fits g₂−1 directly with a floating '
            'baseline — more robust to drift and noisy/low-count data.\n'
            '• Linear: weighted polynomial fit of ln(g₂−1).\n'
            'Switching this clears existing cumulant-based results (you will be asked '
            'to confirm).')
        form.addRow('Cumulant method:', self.cumulant_method)
        self.cumulant_order = QtWidgets.QSpinBox()
        self.cumulant_order.setRange(1, 4)
        self.cumulant_order.setToolTip(
            'Cumulant expansion order that replicate averaging fits at, and the starting '
            'value of the Correlogram sub-tab’s own order box (which you can still '
            'change per run). Order 1 fits a single decay rate and gives no PDI.\n'
            'Changing it clears the results fitted at the old order — Γ vs q², D vs c, '
            'DDLS and replicate averages (you will be asked to confirm); the Correlogram '
            'sub-tab’s own fits ran at its own order and are kept.')
        form.addRow('Cumulant order:', self.cumulant_order)
        self.cumulant_weighting = QtWidgets.QComboBox()
        self.cumulant_weighting.addItem('Sourced variance (default)', 'sourced')
        self.cumulant_weighting.addItem('Unweighted', 'uniform')
        self.cumulant_weighting.setToolTip(
            'Residual weighting of every cumulant fit (per-measurement cumulant, Γ vs q², '
            'D vs c, replicate averaging, DDLS and the Cross-Sample size candidates).\n'
            '• Sourced variance (default): each method’s own sourced weight.\n'
            '• Unweighted: plain least squares, for comparison with a fit made without a '
            'noise model — it moves Rh and PDI wherever the noise is not uniform.\n'
            'The choice is named in the correlogram-fit, Γ vs q² and D vs c exports (not in '
            'the DDLS, replicate-average or summary results). Switching '
            'clears existing cumulant-based results (you will be asked to confirm). See '
            'the Theory-and-Equations-Guide §3.2.8.')
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
            'Drop this many leading correlator channels (shortest lags) from every DLS '
            'fit to remove afterpulsing / dead-time artifacts; keep it small so a real '
            'small-particle mode is not erased. Changing it asks first, then clears every '
            'DLS result — the per-measurement fits, Γ vs q², D vs c, DDLS, the Mode Map '
            'with its mode-family fits, and replicate averages; recorded mode sets and '
            'their results, and the mode families you picked, are kept. See the '
            'Theory-and-Equations-Guide.')
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
            'Amplitude at which the linear cumulant’s fitted range stops, as a '
            'fraction of the correlogram peak. ISO 22412 prescribes 0.01; this program '
            'defaults to 0.1, a benchmarked deviation — see the '
            'Theory-and-Equations-Guide for the measurement and for which of the '
            'standard’s companion clauses are implemented.')
        cutoff_help = HelpBadge(
            'The linear cumulant fits a contiguous run of channels around the '
            'correlogram’s peak. This sets where that run stops — at the '
            'first channel whose amplitude falls to this fraction of the peak.',
            bullets=[
                '<b>Lower</b> (towards ISO’s 0.01) fits further down the decay, '
                'using more channels.',
                '<b>Higher</b> keeps only the strongest part of the decay.',
                'It applies to the <b>Linear</b> cumulant method. Under '
                '<b>Nonlinear</b>, which fits every live channel in the window, it still sets the range '
                'the reported RMS error is measured over and the starting guess the fit '
                'departs from.',
                'Lowering it can make a run look <i>less</i> polydisperse than it is: '
                'the extra channels are the noisiest, the fitted curve flattens, and the '
                'reported PDI falls. Since PDI is what the polydispersity warning reads, '
                'a run that was flagged can come back unflagged at a lower cutoff. On '
                'this program’s own test corpus that happened to 35 of 190 runs, '
                'and never the other way round.',
                'Changing it marks any cumulant fit already on screen as out of date.',
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
                '<b>Average derived results…</b>) has no fields of its own and runs on '
                'these values.',
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
            'a single correlogram, an NNLS/CONTIN distribution, and '
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
                'NNLS/CONTIN peak, and every derived number that '
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
