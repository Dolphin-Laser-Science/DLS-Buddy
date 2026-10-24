"""
The replicate-averaging dialog the shell opens, its request and the summary it shows after
the average.
"""
from __future__ import annotations

import dataclasses
import math
from collections import namedtuple
from typing import List, Optional, Tuple

from PySide6 import QtWidgets

from core.note_registry import REGISTRY
from core.notes import WHERE, Note, grouped, render
from gui.theme import ThemedLabel
from core.formatting import format_pm

from gui.dls.region import _AnalysisRegion, RegionCheckBox


# ===========================================================================
# Replicate-averaging dialogs (driven from the sidebar right-click)
# ===========================================================================

# Display name -> controller method key, in menu order. The distribution methods
# (NNLS/CONTIN/lognormal) yield peaks, averaged positionally (Rh-ascending).
_AVERAGE_METHODS = [
    ('Cumulant', 'cumulant'),
    ('Single exponential', 'single'),
    ('Double exponential', 'double'),
    ('Stretched Exponential (KWW)', 'kww'),
    ('NNLS distribution', 'nnls'),
    ('CONTIN distribution', 'contin'),
    ('Lognormal distribution', 'lognormal'),
]


#: What the averaging prompt can read or copy from the DLS tab, handed in by the module
#: that owns the tabs: the shared region, and zero-argument / per-method accessors for the
#: Correlogram tab's order box and the Distribution tab's fields, read at confirmation.
AverageSources = namedtuple('AverageSources',
                            'region cumulant_order distribution_kwargs')


class AverageRequest(namedtuple('AverageRequest',
                                'method region cumulant_order distribution_kwargs',
                                defaults=(None, None, None))):
    """What the averaging prompt asked for: the method, the region when its box was
    ticked, and the one-time copy of the Correlogram tab's order or the Distribution
    tab's fields when the copy box was ticked (None: the Settings values)."""

    def run_kwargs(self) -> dict:
        """The keyword arguments `Controller.average_dls_results` takes for them."""
        return dict(region=self.region, cumulant_order=self.cumulant_order,
                    distribution_kwargs=self.distribution_kwargs)


class AverageDialog(QtWidgets.QDialog):
    """The "Average derived results…" prompt: which fit each replicate gets, whether it
    reads the Correlogram tab's region (the shared `RegionCheckBox`), and whether it
    copies the matching DLS sub-tab's per-run settings -- the Correlogram tab's cumulant
    order for a cumulant average, the Distribution tab's fields for CONTIN, NNLS or
    lognormal. The Settings values an unticked copy runs on are shown either way, from
    the same controller computation that writes the summary's note, so what the prompt
    showed and what the average recorded cannot disagree. Averaging runs once, so a copy
    taken here is one-time by construction -- the Mode Map's copy button, in the one
    place averaging is started."""

    def __init__(self, parent, controller, sources: Optional[AverageSources] = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._sources = sources
        self.setWindowTitle('Average derived results')
        lay = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            'Fit each replicate with which method? The mean ± SD/√N of its parameters is '
            'reported; where a ± would measure nothing, the summary withholds it and says '
            'why. Distribution methods report per-peak averages in Rh order, and a peak on '
            'one grid point (not the grid’s end) in every fit also carries the grid’s '
            'resolution in its ±. CONTIN over many replicates is slow.')
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.method_combo = QtWidgets.QComboBox()
        for label, key in _AVERAGE_METHODS:
            self.method_combo.addItem(label, key)
        lay.addWidget(self.method_combo)
        self.region_box = RegionCheckBox(
            sources.region if sources is not None else _AnalysisRegion())
        lay.addWidget(self.region_box)
        self.copy_box = QtWidgets.QCheckBox()
        self.copy_box.setToolTip(
            'A one-time copy of that sub-tab\'s current per-run value(s), taken when you '
            'press OK; unticked, the average runs on the Settings values shown below. '
            'The average\'s summary records which, and a written Summary row names the '
            'cumulant order it ran at.')
        lay.addWidget(self.copy_box)
        self.settings_label = ThemedLabel('', role='hint', size=10)
        self.settings_label.setWordWrap(True)
        lay.addWidget(self.settings_label)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.method_combo.currentIndexChanged.connect(self._sync_method)
        self._sync_method()

    def method(self) -> str:
        return self.method_combo.currentData()

    def _sync_method(self, *_args) -> None:
        """Show the copy choice the method has -- the order for a cumulant average, the
        Distribution tab's fields for an inversion, none for the other fits -- and the
        Settings values it would otherwise run on."""
        method = self.method()
        s = self._sources
        if method == 'cumulant':
            now = f' (now {s.cumulant_order()})' if s is not None else ''
            self.copy_box.setText(f"Use the Correlogram tab's cumulant order{now}")
        else:
            self.copy_box.setText("Use the Distribution tab's settings")
        copyable = method == 'cumulant' or method in ('contin', 'nnls', 'lognormal')
        self.copy_box.setVisible(copyable and s is not None)
        try:
            text = self.controller.average_inputs_text(method)
            text = f'Unticked: {text}' if text else ''
        except ValueError as exc:
            # A hand-edited Settings token the seed home refuses: say so here, where the
            # average would otherwise fail on every replicate with the same refusal.
            text = f'Settings cannot seed this fit: {exc}'
        self.settings_label.setText(text)
        self.settings_label.setVisible(bool(text))

    def request(self) -> AverageRequest:
        """The request for what the dialog shows now. The copy is taken HERE, once."""
        method = self.method()
        s = self._sources
        copy = (not self.copy_box.isHidden()) and self.copy_box.isChecked() and s is not None
        order = s.cumulant_order() if copy and method == 'cumulant' else None
        dist = (s.distribution_kwargs(method)
                if copy and method in ('contin', 'nnls', 'lognormal') else None)
        return AverageRequest(method, self.region_box.region_for_run(
            self.region_box.isChecked()), order, dist)


def ask_average_method(parent, controller,
                       sources: Optional[AverageSources] = None) -> Optional[AverageRequest]:
    """Ask which dynamic fit to run on each replicate, and whether it reads the
    Correlogram tab's region and copies the matching sub-tab's per-run settings.
    Returns the request, or None if the user canceled. Cumulant is the default (ISO
    22412's sizing basis, whose size is the scattered-intensity-weighted harmonic mean)."""
    dlg = AverageDialog(parent, controller, sources)
    if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None
    try:
        return dlg.request()
    except ValueError as exc:       # a copied CONTIN α range the engine would refuse
        QtWidgets.QMessageBox.warning(parent, 'Average derived results', str(exc))
        return None


def _grouped_preconditions(notes, n_total: int) -> Tuple[List[Note], list]:
    """The precondition notes of `notes` as the averaging dialog states them, each once
    with the runs that hold it named or counted out of the `n_total` averaged
    (`core.notes.grouped`, the block's rule), and every other note in order. A relayed
    note's `where` names its runs: each becomes one source for the grouping; one with no
    runs named is kept as it is."""
    pairs, whole, rest = [], [], []
    for note in notes:
        spec = REGISTRY.get(note.code) if isinstance(note, Note) else None
        if spec is None or spec.kind != 'precondition':
            rest.append(note)
            continue
        where = note.values.get(WHERE)
        if not (isinstance(where, tuple) and where):
            whole.append(note)
            continue
        bare = dataclasses.replace(
            note, values={k: v for k, v in note.values.items() if k != WHERE})
        pairs.extend((str(name), (bare,)) for name in where)
    return whole + [n for n, _names in grouped(pairs, what='runs', n_total=n_total)], rest


def _average_note_lines(notes, n_total: int) -> List[str]:
    """The averaging dialog's note lines. Each precondition once, with the runs that hold
    it named or counted out of the `n_total` averaged (`_grouped_preconditions`); then, in
    the controller's order, the line saying what the fits read and every other note's
    rendered clause. Six significant figures, so replicate temperatures show the digits
    they differ in."""
    shown, rest = _grouped_preconditions(notes, n_total)
    lines = [f'⚠ {render(n, sig=6, counted=True)}' for n in shown]
    lines += [f'ⓘ {render(n, sig=6, counted=True) if isinstance(n, Note) else n}' for n in rest]
    return lines


def format_average_summary(summary: dict) -> str:
    """Render the controller's average_dls_results summary as display text.

    Kept Qt-free so it can be unit-tested headlessly; the GUI wraps it in a
    QMessageBox. Handles both the parametric summary (a `parameters` list, with its
    fits' reliability verdict in `reliability_warning`) and the distribution summary (a
    `peaks` list with a possible count-disagreement or sweep-edge warning); each draws
    its verdict as a ⚠ line."""
    method = summary.get('method', '?')
    n_ok = summary.get('n_fit_ok', 0)
    n_rep = summary.get('n_replicates', 0)
    lines = [f'DLS replicate average — {method}',
             f'{n_ok} of {n_rep} replicate(s) fit successfully.']
    lines += _average_note_lines(summary.get('notes') or (), n_ok)
    lines.append('')

    peaks = summary.get('peaks')
    if peaks is not None:                          # distribution method
        if not peaks:
            lines.append('No peaks resolved in the replicates.')
        for pk in peaks:
            seen = f"seen in {pk['n_resolved']}/{pk['n_total']} runs"
            rh = format_pm(pk['rh_mean'], pk['rh_sem'], 'nm').strip()
            w = pk['weight_mean']
            wtxt = '' if (w is None or not math.isfinite(w)) else \
                   f", weight ≈ {w * 100:.0f}%"
            # Each clause says what the ± or the position is, once per peak: the grid's
            # resolution when it entered the ±, and the fits whose one weighted grid point is
            # the grid's end, where the position may lie beyond the grid. "grid point", never
            # "node", in user-facing text (ui_style_guide §9).
            clauses = []
            if pk.get('rh_resolution_included'):
                clauses.append('± includes the grid resolution (one weighted grid point per fit)')
            n_edge = pk.get('n_edge_one_node') or 0
            if n_edge:
                clauses.append(f"in {n_edge} fit(s) its weight sits on one grid point at the "
                               f"end of the Rh grid, so its position may lie beyond the grid")
            tail = f" — {'; '.join(clauses)}" if clauses else ''
            # The position's own verdict: a marker at its Rh when an averaged input peak
            # failed a precondition, and the causes on the line under it, each once with
            # its runs, out of the runs that resolved this peak.
            verdict, _rest = _grouped_preconditions(pk.get('notes') or (),
                                                    pk.get('n_resolved') or 0)
            mark = ' ⚠' if verdict else ''
            lines.append(f"Peak {pk['position']} ({seen}): geometric-mean Rh = "
                         f"{rh}{mark}{wtxt}{tail}")
            lines += [f'   ⚠ {render(n, sig=6, counted=True)}' for n in verdict]
        if summary.get('peak_count_warning'):
            lines += ['', f"⚠ {summary['peak_count_warning']}"]
        lines += ['',
                  'Peaks are reported only (not written). Each position is the geometric '
                  'mean of the peak; positions are regularization-dependent, so ± is a '
                  'reproducibility spread, with the grid resolution where stated.']
        return '\n'.join(lines)

    for p in summary.get('parameters', []):        # parametric method
        # ± is the SD/√N across replicates (None when < 2 usable fits, and also when the
        # replicates agree exactly -- a zero spread is withheld, not shown as ± 0).
        # `format_pm` renders None and 0.0 identically, so this shows the bare value. A ±
        # the controller withheld for a stated reason says so (`se_withheld`).
        why = p.get('se_withheld')
        lines.append(f"{p['name']:<10} = "
                     f"{format_pm(p['mean'], p['sem'], p['unit']).strip()}"
                     + (f" (± withheld: {why})" if why else ''))
    # The contributing fits' reliability verdict, as the distribution branch draws its
    # own: the dialog may be the only surface the verdict reaches.
    if summary.get('reliability_warning'):
        lines += ['', f"⚠ {summary['reliability_warning']}"]
    lines.append('')
    if summary.get('rh_written'):
        lines.append('Rh ± SE written to the sample '
                     '(Cross-Sample / ρ = Rg/Rh will use it).')
        # The replacement disclosure is unconditional -- it never turns off; the
        # optional ask-BEFORE confirm is the Settings checkbox (tier rule:
        # SampleResult docstring).
        if summary.get('rh_replaced_pick'):
            lines.append(f"This run replaced your picked Rh "
                         f"({summary['rh_replaced_pick']}).")
    elif summary.get('rh_skip_reason'):
        lines.append(f"Rh not written: {summary['rh_skip_reason']}")
    else:
        # Belt and braces. The controller's chain is exhaustive, so this line should be
        # unreachable -- but a missing reason must never render as SILENCE about the
        # sample Rh.
        lines.append('Rh not written (no reason recorded).')
    return '\n'.join(lines)


def show_average_summary(parent, summary: dict) -> None:
    """Show the derived-results averaging summary in a modal box."""
    QtWidgets.QMessageBox.information(
        parent, 'DLS replicate average', format_average_summary(summary))
