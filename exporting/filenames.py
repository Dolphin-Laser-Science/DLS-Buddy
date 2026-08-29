"""
exporting/filenames.py
======================

Suggested-filename construction for every export surface (CSV export dialogs and
plot saves). Pure string functions — no Qt, no I/O — so the naming scheme is
unit-testable without a GUI.

The house scheme (owner-chosen, 2026-08-11):

    {sample}_{analysis}.{ext}      e.g.  PVP-water-298.1K_debye.csv

where the sample token is `{polymer}-{solvent}-{T:g}K` built from the VERBATIM
(original capitalization) committed names, and the analysis token reuses the
snake_case vocabulary of this package's `export_*` functions. A bare
`{analysis}.{ext}` name is used when no single sample owns the export — the
code's predicate, not a category: sample-independent views (summaries,
cross-sample tables) always, and ticked-measurement views (correlogram,
distribution, α-sweep) exactly when the selection spans more than one sample.

Everything emitted here must be a legal filename on Windows (the strictest
mainstream target): no `\\/:*?"<>|` or control characters, no trailing dots or
spaces, no reserved device stems (CON, COM1, ...). Non-ASCII characters that are
legal on NTFS/ext4/APFS (µ, °, Greek) are deliberately KEPT — polymer and solvent
names are scientific text.
"""

from __future__ import annotations

import math
import re
from typing import Optional, Tuple

# Windows-reserved characters plus ASCII control chars (incl. DEL). A run of
# forbidden characters collapses to a single '-', so 'a/:b' -> 'a-b', never 'a--b'.
_FORBIDDEN_RUN = re.compile(r'[\\/:*?"<>|\x00-\x1f\x7f]+')

# Reserved DOS device stems. Windows reserves the FULL stem only (case-insensitive):
# 'con.csv' is illegal, 'con_gamma_q2.csv' is fine.
_RESERVED_STEMS = frozenset(
    {'con', 'prn', 'aux', 'nul'}
    | {f'com{i}' for i in range(1, 10)}
    | {f'lpt{i}' for i in range(1, 10)}
)

# Cap on the full suggested filename (stem + dot + extension). Comfortably under
# the 255-char per-component filesystem limit, leaving headroom for fan-out
# suffixes ('_m0001_contin') and deep directory paths.
_MAX_NAME_LEN = 120


def sanitize_component(text: str) -> str:
    """Make one filename component Windows-legal; '' if nothing survives.

    Replaces each run of reserved characters (`\\/:*?"<>|`, control chars) with a
    single '-', then strips leading/trailing whitespace, dots, and stray '-' left
    at the edges by the replacement. Interior Unicode is kept verbatim. Fallbacks
    for an empty result are the caller's job.
    """
    cleaned = _FORBIDDEN_RUN.sub('-', text)
    return cleaned.strip(' \t-.')


def sample_token(polymer: str, solvent: str,
                 temperature_K: Optional[float]) -> str:
    """The `{polymer}-{solvent}-{T:g}K` token from verbatim-case sample fields.

    Segments that sanitize to nothing are dropped; the temperature segment is
    dropped when None or non-finite (an uncommitted temperature must not yield
    'nanK'). Returns '' when every segment drops — callers substitute their own
    fallback (in the GUI, the sanitized sample id).
    """
    segments = [sanitize_component(polymer or ''), sanitize_component(solvent or '')]
    if temperature_K is not None and math.isfinite(temperature_K):
        segments.append(f'{temperature_K:g}K')
    return '-'.join(s for s in segments if s)


def suggested_name(sample: Optional[Tuple[str, str, Optional[float]]],
                   analysis: str, ext: str = 'csv') -> str:
    """The suggested filename for one export.

    sample   : (polymer, solvent, temperature_K) verbatim-case fields, or None
               for a sample-independent export (summaries, cross-sample tables).
    analysis : snake_case analysis token (the `export_*` vocabulary).
    ext      : extension without the dot.

    `{sample_token}_{analysis}.{ext}` when a sample is given (falling back to
    bare `{analysis}.{ext}` if the token comes up empty), with the reserved-stem
    guard and the length cap applied. The length cap truncates the sample token,
    never the analysis token or extension.
    """
    token = sample_token(*sample) if sample is not None else ''
    tail = f'{analysis}.{ext}'
    if token:
        overflow = len(token) + 1 + len(tail) - _MAX_NAME_LEN
        if overflow > 0:
            token = token[:max(len(token) - overflow, 0)].rstrip(' -.')
    name = f'{token}_{tail}' if token else tail
    stem = name.rsplit('.', 1)[0]
    if stem.lower() in _RESERVED_STEMS:
        name = '_' + name
    return name
