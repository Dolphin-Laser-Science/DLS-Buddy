"""
exporting/filenames.py
======================

Suggested-filename construction for every export surface (CSV export dialogs and
plot saves). Pure string functions — no Qt, no I/O — so the naming scheme is
unit-testable without a GUI.

**Three rules, no branches** (owner-chosen, 2026-09-16). Which rule applies is
decided by what the export's content belongs to, never by which tab asked:

    one measurement   {polymer}-{solvent}_{analysis}[_{method}]_{name}_{angle:g}deg
                      PMAA PDDA-water_correlogram_cumulant_3_90deg.csv
    one sample        {polymer}-{solvent}_{analysis}
                      PS-toluene_zimm.csv
    cross-sample      {analysis}
    (or sample-free)  dls_summary.csv

`{name}` is the measurement's one name (`core.naming.measurement_name`: the label
if set, else the source-file stem, else `avg of N`, else the internal id), passed
in as a string so this module stays pure over strings. One name is respelled on the
way into a file: an unlabelled average's `avg of N` becomes `avg-of-N` (see
`_file_form`); every other name reaches the file exactly as the screen shows it. `{method}` is present
whenever the analysis offers a choice (`correlogram_cumulant`,
`distribution_contin`, `weighted_distribution_nnls`). The **angle** rides every
per-measurement export whose measurement has a committed angle and is dropped when
it has none; the **temperature never appears** — two samples differing only in
temperature therefore propose the same sample-level name, which the save dialog
shows and the user edits.

**A batch keeps the save dialog and the typed stem.** `batch_export_names` splits a
batch into the **stem** every file shares (`PMAA PDDA-water_correlogram_cumulant`;
for a Distribution batch mixing methods, `PMAA PDDA-water_distribution`) and each
file's **suffix** (`_3_90deg`, `_contin_3_90deg`). Whatever the user types replaces
the stem and every suffix is appended to it, so typing `tuesday-batch` writes
`tuesday-batch_1_90deg.csv` … `tuesday-batch_6_90deg.csv`. With exactly one
measurement the suffix is folded in and the dialog shows the whole name — which is
why a measurement exported alone and the same measurement exported in a batch under
the proposed stem get the **same** file name.

The internal id is **not** part of the scheme: it reaches a file name only through
the in-batch collision fallback, a safety net against a silent overwrite.

Everything emitted here must be a legal filename on Windows (the strictest
mainstream target): no `\\/:*?"<>|` or control characters, no trailing dots or
spaces, no reserved device stems (CON, COM1, ...). Non-ASCII characters that are
legal on NTFS/ext4/APFS (µ, °, Greek) are deliberately KEPT — polymer and solvent
names are scientific text. The length cap truncates the **sample token** first and
then the **head** of `{name}` — a file stem the user never chose can be long, and
names differ at their tails — never the analysis, method, angle or extension.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

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
# the 255-char per-component filesystem limit, leaving headroom for a typed stem
# longer than the proposed one and for deep directory paths.
_MAX_NAME_LEN = 120


class ExportMember(NamedTuple):
    """One measurement of a per-measurement export or batch.

    item_id   : the internal id -- used ONLY as the in-batch collision fallback.
    name      : the measurement's one name (`core.naming.measurement_name`).
    angle_deg : the COMMITTED scattering angle, or None when none is committed
                (the angle token is then dropped).
    method    : the analysis method key when the analysis offers a choice, else
                None. Shared by the whole batch -> it rides the stem; mixed ->
                each file carries its own in its suffix.
    """
    item_id: str = ''
    name: str = ''
    angle_deg: Optional[float] = None
    method: Optional[str] = None


def sanitize_component(text: str) -> str:
    """Make one filename component Windows-legal; '' if nothing survives.

    Replaces each run of reserved characters (`\\/:*?"<>|`, control chars) with a
    single '-', then strips leading/trailing whitespace, dots, and stray '-' left
    at the edges by the replacement. Interior Unicode is kept verbatim. Fallbacks
    for an empty result are the caller's job.
    """
    cleaned = _FORBIDDEN_RUN.sub('-', text)
    return cleaned.strip(' \t-.')


def sample_token(polymer: str, solvent: str) -> str:
    """The `{polymer}-{solvent}` token from verbatim-case sample fields.

    Segments that sanitize to nothing are dropped; '' when both do -- callers
    substitute their own fallback (in the GUI, the sanitized sample id). The
    temperature is deliberately absent: see the module docstring.
    """
    segments = [sanitize_component(polymer or ''), sanitize_component(solvent or '')]
    return '-'.join(s for s in segments if s)


def _angle_token(angle_deg: Optional[float]) -> str:
    """`{angle:g}deg`, or '' when there is no usable committed angle.

    Two-sided by construction: `math.isfinite` rejects NaN and both infinities, so
    neither a missing angle nor a saturated one can write `nandeg` into a file name.
    """
    if angle_deg is None:
        return ''
    try:
        value = float(angle_deg)
    except (TypeError, ValueError):
        return ''
    return f'{value:g}deg' if math.isfinite(value) else ''


# The display spells an unlabelled replicate average `avg of N`; the FILE NAME spells
# it `avg-of-N`. The two forms differ on purpose -- a file name is read in a shell, a
# glob and a sort, where a run of spaces inside one token is a nuisance the on-screen
# form has no reason to carry. This is the one place the two diverge, and it is a
# substitution on a name the builder already produced, not a second naming rule: every
# other name reaches a file exactly as the screen shows it.
_AVERAGE_DISPLAY = re.compile(r'^avg of (\d+)$')

# The average's OTHER computed name, for a session that records no contributors
# (`core.naming.measurement_name`). It is a sibling of `avg of N` -- both are names the
# builder COMPUTES rather than names a user chose -- so it takes the same respelling;
# covering one and not the other was the miss this constant exists to close.
_AVERAGE_PLAIN = 'replicate average'


def _file_form(name: str) -> str:
    """A measurement name as a file-name token.

    Anchored to the WHOLE name, so a label a user typed that merely contains one of
    these phrases is left exactly as they typed it.
    """
    text = (name or '').strip()
    m = _AVERAGE_DISPLAY.match(text)
    if m:
        return f'avg-of-{m.group(1)}'
    return 'replicate-average' if text == _AVERAGE_PLAIN else name


def _suffix(name: str, angle_deg: Optional[float], method: Optional[str]) -> str:
    """One file's suffix: `[_{method}]_{name}[_{angle}deg]`, empty tokens dropped."""
    parts = [sanitize_component(method or ''), sanitize_component(_file_form(name)),
             _angle_token(angle_deg)]
    return ''.join(f'_{p}' for p in parts if p)


def _shared_method(members: Sequence[ExportMember]) -> Optional[str]:
    """The method every member shares, or None when they differ or any names none.

    A shared method belongs in the stem the dialog shows (`correlogram_cumulant`);
    a mixed set has none, so each file carries its own in its suffix (`_contin_…`
    beside `_nnls_…`).
    """
    methods = {m.method or '' for m in members}
    return methods.pop() if len(methods) == 1 and '' not in methods else None


def _head(token: str, fixed: str) -> str:
    """`{sample}_{analysis}[_{method}]`, with either half dropped when it is empty."""
    return f'{token}_{fixed}' if token and fixed else (token or fixed)


def _break_collisions(rows: Sequence[ExportMember],
                      suffixes: Sequence[str]) -> List[str]:
    """Append the internal id to every member of a colliding group; pass the rest through.

    The safety net, not part of the scheme: two files of one batch sharing a suffix
    would be written to one path and silently overwrite each other. Compared
    CASE-INSENSITIVELY because the strictest target (Windows/NTFS) is, so `batch 1`
    and `Batch 1` collide there and are broken here. The whole group is suffixed,
    never all-but-one, so colliding files stay symmetric.
    """
    counts: Dict[str, int] = {}
    for s in suffixes:
        counts[s.casefold()] = counts.get(s.casefold(), 0) + 1
    return [f'{s}_{sanitize_component(m.item_id)}' if counts[s.casefold()] > 1 else s
            for m, s in zip(rows, suffixes, strict=True)]


def _fit_length(token: str, fixed: str, rows: Sequence[ExportMember],
                suffixes: Sequence[str], ext: str) -> Tuple[str, List[str]]:
    """Bring every file's full name under the cap: sample token first, then name heads.

    The sample token is SHARED by the batch, so it is shortened once against the
    longest member and the stem stays one string. Each remaining overflow then eats
    the HEAD of that member's own name. The analysis, method, angle and extension
    are never touched, so a pathologically long fixed part overflows the cap rather
    than mangling what identifies the file.
    """
    budget = _MAX_NAME_LEN - len(ext) - 1
    method = _shared_method(rows)
    over = len(_head(token, fixed)) + max((len(s) for s in suffixes), default=0) - budget
    if over > 0 and token:
        token = token[:max(len(token) - over, 0)].rstrip(' -.')
    out = []
    for m, s in zip(rows, suffixes, strict=True):
        over = len(_head(token, fixed)) + len(s) - budget
        if over > 0 and m.name:
            trimmed = sanitize_component(m.name)[over:].lstrip(' -.')
            s = _suffix(trimmed, m.angle_deg, None if method else m.method)
        out.append(s)
    # Re-run over the trimmed set: two names that differed only in their heads are
    # one name once the heads are cut, and the fallback is what keeps them apart.
    return token, _break_collisions(rows, out)


def batch_export_names(sample: Optional[Tuple[str, str]], analysis: str,
                       members: Iterable[Sequence] = (),
                       ext: str = 'csv') -> Tuple[str, List[str]]:
    """The scheme's one body: `(dialog_name, suffixes)` for any export.

    sample   : (polymer, solvent) verbatim-case fields, or None for a
               sample-independent export (summaries, cross-sample tables, a ticked
               set spanning samples).
    analysis : snake_case analysis token (the `export_*` vocabulary).
    members  : the measurements this export fans out over, as `ExportMember`s (or
               plain `(item_id, name, angle_deg, method)` sequences). EMPTY for a
               sample-level or cross-sample export, which names no measurement.
    ext      : extension without the dot.

    `dialog_name` is what the save dialog is seeded with: the whole name for no
    members or exactly one, the shared stem for a batch. `suffixes` is one string
    per member, aligned with `members`, appended to whatever stem the user finally
    types ('' when there is at most one member).
    """
    rows = [ExportMember(*m) for m in members]
    token = sample_token(*sample) if sample is not None else ''
    method = _shared_method(rows)
    fixed = '_'.join(p for p in (analysis, method or '') if p)
    suffixes = _break_collisions(
        rows, [_suffix(m.name, m.angle_deg, None if method else m.method) for m in rows])
    token, suffixes = _fit_length(token, fixed, rows, suffixes, ext)

    head = _head(token, fixed)
    if head.lower() in _RESERVED_STEMS:
        # Guarded on the STEM the dialog offers, which is every batch file's leading
        # part too, so one guard covers the name the user accepts and the names the
        # fan-out derives from it.
        head = '_' + head
    if len(rows) == 1:
        # One measurement: the dialog shows the whole name, so a run exported alone
        # and the same run exported in a batch under the proposed stem agree.
        head, suffixes = head + suffixes[0], ['']
    return f'{head}.{ext}', suffixes


def measurement_export_name(sample: Optional[Tuple[str, str]], analysis: str,
                            name: str, *, angle_deg: Optional[float] = None,
                            method: Optional[str] = None,
                            ext: str = 'csv') -> str:
    """The one-measurement rule for a single file: the whole name, no fan-out."""
    return batch_export_names(
        sample, analysis, [ExportMember('', name, angle_deg, method)], ext)[0]


def suggested_name(sample: Optional[Tuple[str, str]], analysis: str,
                   ext: str = 'csv') -> str:
    """The one-sample (`{sample}_{analysis}`) and cross-sample (`{analysis}`) rules.

    sample : (polymer, solvent) verbatim-case fields, or None for a
             sample-independent export. A token that sanitizes to nothing falls
             back to the bare `{analysis}` name.
    """
    return batch_export_names(sample, analysis, (), ext)[0]
