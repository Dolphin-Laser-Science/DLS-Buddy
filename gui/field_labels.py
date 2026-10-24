"""
gui/field_labels.py
===================

The single source of truth mapping the data model's **internal field names** to the
**user-facing labels** shown in the GUI, plus a small translator that swaps the former
for the latter inside any user-visible message.

Any surface that shows a field's name reads it here (a search of the tree for
`field_labels` finds them): a row's label, or a caught exception whose text embeds the raw
field name (e.g. "polymer_name must be a non-empty string"), which
:func:`humanize_field_names` rewrites so a user reads "Solute name …" instead of the
internal `polymer_name`.

It also holds `CHAIN_TYPE_LABELS`, the label each stored chain-type value is shown under,
and :func:`has_honest_se`, which decides whether a ± is shown beside a value.

Physics-free presentation only; imports nothing but the standard library.
"""

from __future__ import annotations

import math
import re

# Internal data-model field name -> user-facing label. The user-facing name for
# `polymer_name` is "Solute name" (samples need not be polymers); the internal field
# stays `polymer_name` to avoid a data-model/session rename. Unit labels are shown in a
# separate column / tooltip, so no label carries a unit suffix.
PARAM_LABELS = {
    'label':       'Label',
    'polymer_name': 'Solute name',
    'solvent_name': 'Solvent name',
    'concentration_g_per_mL': 'Concentration',
    'temperature_K': 'Temperature',
    'angle_deg': 'Scattering angle',
    'wavelength_nm': 'Wavelength',
    'solvent_refractive_index': 'Solvent refractive index',
    'viscosity_Pa_s': 'Viscosity',
    'mean_count_rate_cps': 'Mean count rate',
    'solute_refractive_index': 'Solute refractive index',
    'shell_refractive_index': 'Shell refractive index',
    'dn_dc_mL_per_g': 'dn/dc',
    'mw_fraction': 'Mw fraction',
    'solvent_blank_item_id': 'Solvent blank',
    'solvent_blank_intensity_90deg': 'Solvent blank intensity at 90°',
    'analyzer_geometry': 'Polarization (DDLS)',
    'chain_type': 'Chain type',
    # Calibration fields, not measurement parameters (no Data-tab row), named as the
    # SLS Calibration panel labels them so a refusal of a value reads the same there.
    'standard_refractive_index': 'Standard refractive index',
    'dark_count_rate': 'Dark count rate',
}

# A stored `chain_type` value -> the label a user reads for it.
CHAIN_TYPE_LABELS = {'linear': 'Linear', 'branched': 'Branched'}

# Match each field name as a whole word-run. The keys carry underscores (a word char), so
# `\b…\b` still anchors correctly and never fires inside a longer identifier. Longest keys
# first so a key that is a prefix of another can't shadow it (none are today, but this
# keeps the alternation order-independent).
_TOKEN = re.compile(
    '|'.join(rf'\b{re.escape(k)}\b'
             for k in sorted(PARAM_LABELS, key=len, reverse=True))
)


def missing_input_lines(missing, counted: str = 'shown') -> list:
    """What a fit lacks, one line per input, as the SLS and DLS tabs print it in a grayed
    entry's tooltip and on the status line: the controller's phrase (its one home is the
    roster, `SLS_FIT_REQUIREMENTS` or `DLS_FIT_REQUIREMENTS`), the clause the requirement
    read off the selection when it wrote one, then the Data-tab row it is entered at, or
    for a count how many the selection holds against the minimum, `counted` naming what
    holds them ('shown' under the SLS mask, 'ticked' on the DLS tab)."""
    lines = []
    for m in missing:
        detail = f': {m.detail}' if getattr(m, 'detail', '') else ''
        if m.param is not None:
            where = f' (Data tab: {PARAM_LABELS[m.param]})'
        elif m.have is not None:
            where = f' ({m.have} {counted}, {m.need} needed)'
        else:
            where = ''
        lines.append(f'• {m.phrase}{detail}{where}')
    return lines


def humanize_field_names(text: str) -> str:
    """Replace internal data-model field tokens (``polymer_name``, ``solvent_name``,
    ``temperature_K`` …) with their user-facing labels in a user-visible message.
    Tokens not in :data:`PARAM_LABELS` (e.g. raw array names) pass through unchanged."""
    return _TOKEN.sub(lambda m: PARAM_LABELS[m.group(0)], text)


def has_honest_se(se) -> bool:
    """Whether a ± is shown beside a value, decided by the SE alone: true for an SE that
    is present, finite and positive. The value's own finiteness is the caller's to test."""
    return bool(se is not None and math.isfinite(se) and se > 0)
