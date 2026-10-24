"""
The distribution-fit controls the Settings tab and the DLS Distribution and Mode Map sub-tabs
share (`DistributionFitFields`), their choices and seed fields, and the combo seeding helper.
"""
from __future__ import annotations

import math

from PySide6 import QtCore, QtWidgets

from app.controller import distribution_weighting_request
from gui.widgets import first_selectable_index


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
    'The residual weight of the distribution fits (CONTIN, NNLS, lognormal).\n'
    "• Sourced variance (default): a correlator's variance at each lag, from the run's "
    'committed count rate (its infinite-count-rate limit without one); each fit takes '
    'about twice as long.\n'
    '• g₁² (older form): equal noise on every channel, to reproduce a result reported '
    'under it.\n'
    'See Theory-and-Equations-Guide §3.4.2.'
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
            'dense system of this size about 40 times, so the cost climbs steeply: on a '
            'laptop under GCV, about 0.5 s at 100 points, 5 s at 300 and 30 s at 1000. '
            'The fit runs in the background, so the window stays usable.')
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
            'How CONTIN chooses α automatically.\n'
            '• GCV (default): robust generalized cross-validation, for correlated channel '
            'noise (γ = 0.1; γ = 1 is classical GCV).\n'
            '• L-curve corner: the corner of the fit-vs-size trade-off curve, insensitive '
            'to any α range spanning it; a sweep-edge note says when one does not.\n'
            '• F-test: the smoothest solution not significantly worse than the best fit, '
            'for comparison with legacy CONTIN output.\n'
            'See Theory-and-Equations-Guide §3.4.5.')
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
            'noise. See Theory-and-Equations-Guide §3.4.7.')
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
