"""
gui/field_labels.py
===================

The single source of truth mapping the data model's **internal field names** to the
**user-facing labels** shown in the GUI, plus a small translator that swaps the former
for the latter inside any user-visible message.

Two callers share this map today:

* the Data tab's parameter table (`gui/data_module.py`), which shows one row per field;
* the skipped-run warnings on the DLS tabs (`gui/dls_module.py`), which surface a caught
  exception whose text embeds the raw field name (e.g. "polymer_name must be a non-empty
  string"). :func:`humanize_field_names` rewrites those tokens so a user reads "Solute
  name …" instead of the internal `polymer_name`.

Physics-free presentation only; imports nothing but the standard library.
"""

from __future__ import annotations

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
}

# Match each field name as a whole word-run. The keys carry underscores (a word char), so
# `\b…\b` still anchors correctly and never fires inside a longer identifier. Longest keys
# first so a key that is a prefix of another can't shadow it (none are today, but this
# keeps the alternation order-independent).
_TOKEN = re.compile(
    '|'.join(rf'\b{re.escape(k)}\b'
             for k in sorted(PARAM_LABELS, key=len, reverse=True))
)


def humanize_field_names(text: str) -> str:
    """Replace internal data-model field tokens (``polymer_name``, ``solvent_name``,
    ``temperature_K`` …) with their user-facing labels in a user-visible message.
    Tokens not in :data:`PARAM_LABELS` (e.g. raw array names) pass through unchanged."""
    return _TOKEN.sub(lambda m: PARAM_LABELS[m.group(0)], text)
