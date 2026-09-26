"""
core/naming.py
==============

**One builder for every place a measurement is named**, and one for every place a
sample is. Pure string functions over the data model -- no Qt, no I/O, no lookups --
so the sidebar, the pickers, the legends, the results tables, the warning lines, the
engine-side notes (through a caller-supplied `display_name`) and the export filename
scheme all read from the same body and cannot drift apart.

Before this module there were eleven measurement-name builders and six sample-name
builders, each with its own rules: one unconfirmed sample rendered four different
ways at once, and six replicates of one set read as six identical strings differing
only in `m000#`. The rules here are owner-chosen (2026-09-17):

**One name per measurement** -- the label if set, else the source file **stem**. The
internal ID is the name only when neither exists, because a saved figure or an
exported file reaches readers who have no workspace to look an ID up in. An
unlabelled replicate average is named `avg of N` and then carries no separate
`avg of N` marker, so the phrase appears once.

**A label adds, never replaces.** Polymer and solvent come from the sample header,
the legend title or the export's sample token; angle, concentration and Mw fraction
come from the committed parameters. The label is a short name beside them.

**One display format everywhere** -- the within-sample axes, then the name, then any
markers::

    90° · 3
    90° · 1 mg/mL · 3
    90° · batch 1 (avg of 6)
    PS / toluene @ 298 K · 90° · 3        # sample-prefixed
    0 mg/mL · water blank (solvent ref)

Nothing in a name is computed from which *other* measurements are on screen, so a
name never changes when the ticked set does. The sample prefix is the one
conditional, and it is the **caller's** decision (a legend or list whose ticked set
spans samples passes `sample=`), never a property of the measurement.

**COMMITTED parameters, never working.** Every axis here names what the analysis
computed from; the Data tab owns the pending edit and its highlight.

Residual, stated rather than guarded: a hover card is multi-line plain text, and Qt
auto-detects rich text, so a user label containing a `<tag>`-shaped substring would
be interpreted as markup by the widget that shows it. No instrument writes one and
the label is hand-typed; the cost of being wrong is one mangled tooltip.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional, Sequence

# The separator between the axes and the name, and between the axes themselves. One
# constant so a test can assert the format without re-spelling it.
SEP = ' · '

# What a measurement with no label, no source file and no derived provenance is
# called in a hover card's ID line.
_ID_FIELD = 'ID'

#: What a sample whose identity is not set reads, everywhere. Exported because two
#: GUI surfaces need the STRING and not just the builder: the navigator tints such a
#: node amber, and the picker's group header covers the missing-sample case with the
#: same words. A literal copy at either site would drift from this one.
UNCONFIRMED_SAMPLE = '(unconfirmed sample)'


def _finite(value: Any) -> bool:
    """True for a real, finite number (None, NaN and inf are all not axes)."""
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_replicate_average(lm) -> bool:
    """Is this measurement a replicate average? Keyed on `derived_kind` ALONE, which is
    what every other consumer keys on (`_refuse_an_average_as_a_repeat`, the derived
    route's `replicate_average` reason). Keying the display on a non-empty
    `derived_from` instead made an average with no recorded sources read as an ordinary
    run while those consumers still refused it -- one condition, two predicates."""
    return getattr(lm, 'derived_kind', None) == 'replicate_average'


def _replicate_count(lm) -> int:
    """How many measurements a replicate average was built from (0 if unrecorded)."""
    return len(getattr(lm, 'derived_from', None) or []) if _is_replicate_average(lm) else 0


def source_file_name(lm) -> str:
    """The measurement's source file NAME (never its directory), '' if it has none.

    The name, not the path: a full path buries the part that tells two runs apart,
    which is the whole complaint this module exists to answer. Used by the hover
    card and by the export layout's `Source file` column.

    BOTH separators, not `os.path.basename`, which splits only the one its host uses:
    a session written on Windows carries backslashes, and opened on macOS or Linux the
    whole path came back as the measurement's one NAME -- into the sidebar, the plot
    legends, the Data header, the Summary column and CSV, and the proposed export file
    stem. `exporting/export._bare_file_name` and `core/workspace.blank_label` both
    already split on both for exactly this reason.
    """
    path = getattr(lm, 'source_path', None)
    return str(path).replace('\\', '/').rsplit('/', 1)[-1] if path else ''


def measurement_name(lm) -> str:
    """The measurement's **bare name**: label, else source-file stem, else `avg of N`
    for an unlabelled replicate average, else the internal ID.

    The stem (not the file name) so a rename pattern that strips shared text, and a
    legend, both work on the part that differs; the extension is in the hover card.
    """
    label = str(lm.committed_params.get('label') or '').strip()
    if label:
        return label
    stem = os.path.splitext(source_file_name(lm))[0].strip()
    if stem:
        return stem
    if _is_replicate_average(lm):
        n = _replicate_count(lm)
        # An average whose sources were not recorded still says what it IS, rather than
        # falling through to the internal id as an ordinary run.
        return f'avg of {n}' if n else 'replicate average'
    return lm.item_id


def measurement_axes(lm) -> list:
    """The within-sample axes, in the one fixed order, only when set: angle,
    concentration, Mw fraction.

    Concentration is shown in mg/mL (the unit every GUI surface already used) and
    includes an explicit `0 mg/mL` for a solvent blank -- an axis that is set is
    shown, and a blank's zero is a fact about it, not an absence.
    """
    p = lm.committed_params
    axes = []
    angle = p.get('angle_deg')
    if _finite(angle):
        axes.append(f'{float(angle):g}°')
    conc = p.get('concentration_g_per_mL')
    if _finite(conc):
        axes.append(f'{float(conc) * 1000:.3g} mg/mL')
    frac = p.get('mw_fraction')
    if frac:
        axes.append(f'[{frac}]')
    return axes


def measurement_display_name(lm, *, sample: Optional[str] = None,
                             markers: Sequence[str] = ()) -> str:
    """The one display format: ``[sample · ] axes · name [(markers)]``.

    sample  : the sample's display name, passed by a legend or list whose ticked set
              spans samples (the only conditional in the format). None everywhere a
              sample header already sits above the row.
    markers : extra qualifiers the measurement itself cannot know -- 'solvent ref',
              'extra blank' -- supplied by the surface that knows the sample's slots.
              The `avg of N` marker is added here, and is suppressed when the name IS
              `avg of N`, so the phrase appears once.
    """
    marks = [m for m in markers if m]
    n = _replicate_count(lm)
    name = measurement_name(lm)
    if _is_replicate_average(lm):
        marker = f'avg of {n}' if n else 'replicate average'
        if name != marker:
            marks.insert(0, marker)
    head = ([sample] if sample else []) + measurement_axes(lm) + [name]
    text = SEP.join(head)
    return f'{text} ({", ".join(marks)})' if marks else text


def measurement_hover_card(lm, *, sample: Optional[str] = None,
                           markers: Sequence[str] = ()) -> str:
    """The tooltip shown wherever a measurement's name appears.

    Line 1 is the full, unelided display name -- so this also serves the house
    convention that a name cell's tooltip carries its own full text, however the cell
    elides it. Then the fields a user needs to identify the run: label, file name,
    angle, concentration, temperature, internal ID. Absent fields are omitted; the ID
    line is always present, because it is the one field that always exists and the one
    a bug report needs.
    """
    p = lm.committed_params
    lines = [measurement_display_name(lm, sample=sample, markers=markers)]
    label = str(p.get('label') or '').strip()
    if label:
        lines.append(f'Label: {label}')
    fname = source_file_name(lm)
    if fname:
        lines.append(f'File: {fname}')
    if _finite(p.get('angle_deg')):
        lines.append(f'Angle: {float(p["angle_deg"]):g}°')
    if _finite(p.get('concentration_g_per_mL')):
        lines.append(f'Concentration: {float(p["concentration_g_per_mL"]) * 1000:.3g} mg/mL')
    if _finite(p.get('temperature_K')):
        lines.append(f'Temperature: {float(p["temperature_K"]):g} K')
    n = _replicate_count(lm)
    if n:
        lines.append(f'Averaged from: {n} measurements')
    lines.append(f'{_ID_FIELD}: {lm.item_id}')
    return '\n'.join(lines)


def sample_display_name(sample, fraction: Optional[str] = None) -> str:
    """The one sample name: ``polymer / solvent @ T K``, with ``[fraction]`` when a
    Mw fraction is named.

    A sample whose identity is not yet confirmed in the Data tab reads
    ``(unconfirmed sample)`` -- never the raw `polymer|solvent|temperature` key, which
    used to reach the DLS Summary table and the exported CSV (owner-ruled 2026-09-17).
    It carries whatever identity IS set, because the constant alone cannot tell two
    unconfirmed samples apart, and the DLS Summary's sample-level rows have no other
    identifier: two of them exported as one byte-identical row (owner-ruled 2026-09-18).
    """
    poly, solv, temp = sample.polymer_name, sample.solvent_name, sample.temperature_K
    if poly and solv and _finite(temp):
        base = f'{poly} / {solv} @ {float(temp):g} K'
    else:
        known = [str(v) for v in (poly, solv) if v]
        if _finite(temp):
            known.append(f'{float(temp):g} K')
        base = (f'{UNCONFIRMED_SAMPLE[:-1]}: {" / ".join(known)})' if known
                else UNCONFIRMED_SAMPLE)
    return f'{base} [{fraction}]' if fraction is not None else base
