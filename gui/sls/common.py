"""
Helpers the SLS sub-tabs share: the not-applicable cell, the dash-the-value scale
convention, the Rg and Mw qualifiers, and the prompt before a run over an unmeasured blank.
"""
from __future__ import annotations

from typing import Tuple

from PySide6 import QtWidgets

from plotting.plots import scale_causes, scaled_value


# Single not-applicable placeholder for a structurally missing table cell, with a
# reason tooltip beside it (ui_style_guide S9; the cross_module precedent).
_EMPTY = '—'


# The convention's one body now lives in `plotting.plots`: the Debye/Zimm figure titles
# honour it too, a plotter cannot import the GUI, and a second spelling here is exactly the
# family defect it exists to prevent. Aliased under the private names the call sites use.
_scale_causes = scale_causes


# The "dash the value, name the cause" convention, so the ten Mw/A₂ cells on this tab cannot
# disagree about which causes they honour (ui_style_guide §5 R5.2's carve-out: the dash IS the
# disclosure, so a dashed value takes no marker) — and cannot disagree with the figure drawn
# beside them either.
_scaled_value = scaled_value


# Qualifiers about the whole fit that say what its Rg and Mw are: those two rows' labels
# carry the word and the A₂ row's does not; the rule is `core.notes.qualifiers`' Berry clause.
_RG_MW_QUALIFIERS: Tuple[str, ...] = ('berry_past_zimm_limit',)


def _confirm_unmeasured_blank(parent, controller, sid: str, *, reuse: bool) -> bool:
    """Ask before analysing a sample with NO measured c = 0 solvent blank.

    MODULE-level, not a method, because TWO tabs reach the same blankless path and a gate
    on one of them is not a gate: `_ModeResolvedTab` runs `run_mode_partition` through the
    same `masked_rayleigh`, so before this it raised the old flat refusal on a fresh
    session and computed SILENTLY once the Analysis tab had been confirmed -- the
    acknowledgement is controller-wide (round d0ce f12).

    True = go ahead (the controller has been told); False = the caller returns and
    the tab is left exactly as it was.

    This replaced a flat refusal, and it is deliberately the LOUD part of the
    feature. Every other convenience the program offers rests on a published method;
    a blank computed from one 90° value rests on an inference no opened source
    endorses, so a flag on the result alone is too quiet -- the user is told what
    will happen, in the words of the case they are actually in, before any number
    appears.

    Cancel is the DEFAULT button (the house pattern is
    `gui.export_helper.confirm_overwrite`), so Return or Escape declines.

    It fires on EVERY press of Run while the sample has no measured blank -- the
    answer is not remembered across presses. `reuse=True` is the exception the two
    live-refit callers pass: a mask edit after a confirmed Run reuses that answer
    rather than interrupting the user mid-curation.
    """
    if reuse and controller.has_unmeasured_blank_acknowledgement(sid):
        return True
    typed = controller.committed_typed_blank(sid)
    if typed:
        what = (
            f'The solvent blank will be <b>computed from the typed 90° '
            f'intensity</b> ({typed:.10g}), scaled by 1/sinθ at each '
            f'angle — it assumes the solvent scatters like a simple liquid '
            f'and that the scattering volume goes as 1/sinθ.')
    else:
        what = ('<b>No solvent subtraction</b> will be applied — the solvent '
                'blank intensity at 90° on the Data tab is empty. The dark '
                'count is still removed.')
    resp = QtWidgets.QMessageBox.question(
        parent, 'No solvent blank measured',
        f'<p>This sample has no c = 0 solvent measurement.</p><p>{what}</p>'
        f'<p><b>Every result will be flagged unreliable</b>, the radius of '
        f'gyration included — an unmeasured blank is <b>additive</b>, so unlike '
        f'an unknown calibration scale it does not cancel out of a slope.</p>'
        f'<p>The fit will <b>not</b> update this sample\'s Cross-Sample values '
        f'(M<sub>w</sub>, R<sub>g</sub>, A₂) — a blankless fit is not offered '
        f'there. Enter a value by hand on the Cross-Sample tab if you need it in a '
        f'scaling plot.</p>'
        f'<p>Set or clear the value in <i>Solvent blank intensity at 90°</i> '
        f'on the Data tab.</p><p>Continue?</p>',
        QtWidgets.QMessageBox.StandardButton.Yes
        | QtWidgets.QMessageBox.StandardButton.Cancel,
        QtWidgets.QMessageBox.StandardButton.Cancel)
    if resp != QtWidgets.QMessageBox.StandardButton.Yes:
        return False
    controller.acknowledge_unmeasured_blank(sid)
    return True
