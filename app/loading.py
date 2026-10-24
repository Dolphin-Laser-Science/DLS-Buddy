"""
app/loading.py
==============

The file loader: native files in, measurements and count-rate traces in the workspace out.

ONE PATH from a parser's preview to the common model, for every caller. Each loader runs
the format auto-detection over its kind's roster (`parsers/roster.py`), falls back to the
kind's generic parser, collects the parsers' load-time notes, and hands each preview to the
controller in a fixed order: the copy into `raw` and `params`
(`loaded_inputs_from_preview`), the vendor records, `add_loaded`, the file-borne trace's
store and attach, and the load-time count-rate seed. The copy is public so that code
loading a preview without these loops can make the same one.

Qt-free (invariant 5). The widgets keep the file dialog, the busy guard and every message
box; they hand the loader the prompts it needs as callbacks (the generic DLS fallback with
its units prompt, the trace route's detection and generic fallback, each a widget method)
and render the `LoadReport` it returns.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.notes import Note
from core.workspace import DLS_PARAM_KEYS, SLS_PARAM_KEYS, seed_vendor_records
from parsers.autodetect import Detection, autodetect
from parsers.base_parser import (MalformedDataError, ParseError, SLSFilePreview,
                                 collect_parse_notes)
from parsers.brookhaven_dat import BrookhavenDATPreview
from parsers.generic_sls import GenericSLSParser
from parsers.roster import DLS_PARSERS, SLS_PARSERS, TRACE_PARSERS

if TYPE_CHECKING:
    from app.controller import Controller


@dataclass
class LoadReport:
    """What one load did, for the widget's status line and dialogs.

    `first_id` is the first measurement added (None when nothing was); `n_files` and
    `n_meas` count the files read and the measurements they held. `unreadable` lists the
    file names no supported format matched; `malformed` the `'<file>: <why>'` lines of
    files a parser recognised and then diagnosed; `parse_notes` the `(file name, note)`
    pairs from `collect_parse_notes`; `baseline_fallbacks` the Brookhaven `.dat` files whose
    Settings baseline could not be applied. A trace load fills `last_trace_id`, the last
    trace stored, and leaves the measurement fields at their defaults.
    """

    first_id: Optional[str] = None
    n_files: int = 0
    n_meas: int = 0
    unreadable: List[str] = field(default_factory=list)
    malformed: List[str] = field(default_factory=list)
    parse_notes: List[Tuple[str, Note]] = field(default_factory=list)
    baseline_fallbacks: List[str] = field(default_factory=list)
    last_trace_id: Optional[str] = None


def loaded_inputs_from_preview(preview) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """A DLS or SLS parser preview as the `raw` arrays and `params` `add_loaded` takes.

    Pure: the preview is read, never changed. The vendor records are a separate step
    (`seed_vendor_records`), which the DLS loader applies after this copy.
    """
    if isinstance(preview, SLSFilePreview):
        return _sls_loaded_inputs(preview)
    return _dls_loaded_inputs(preview)


def _dls_loaded_inputs(preview) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw = {
        'delay_times_s': [float(x) for x in preview.delay_times_s],
        'correlogram': [float(x) for x in preview.correlogram],
    }
    # A per-channel sample time the FILE states (the Brookhaven .dat's
    # correlator sections) rides in `raw` beside the lags -- an instrument
    # fact, not a parameter -- and is absent when the format states none.
    if getattr(preview, 'sample_times_s', None) is not None:
        raw['sample_times_s'] = [float(x) for x in preview.sample_times_s]
    # The run duration, likewise an instrument fact beside the lags, where the
    # file states it for this correlogram (an ALV one-run acquisition, or a
    # Brookhaven run length that covers its own live channels).
    if getattr(preview, 'duration_s', None) is not None:
        raw['duration_s'] = float(preview.duration_s)
    params = {k: getattr(preview, k, None) for k in DLS_PARAM_KEYS}
    if isinstance(preview, BrookhavenDATPreview):
        # A raw-counts format: which baseline normalized it is absorbed
        # by the default fits but not by a zero-baseline one, and the
        # workspace stores only the correlogram, so the choice has to be
        # recorded here or it is unrecoverable. Provenance, not a param.
        params['correlogram_baseline_source'] = preview.baseline_provenance
    return raw, params


def _sls_loaded_inputs(preview) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw = {'angles_deg': [float(a) for a in preview.angles_deg],
           'intensities': [float(x) for x in preview.intensities]}
    params = {k: getattr(preview, k, None) for k in SLS_PARAM_KEYS}
    return raw, params


def apply_dat_baseline_setting(controller: "Controller", previews) -> list:
    """Seed a Brookhaven .dat correlogram's baseline from Settings.

    The .dat stores raw counts plus both baselines, so a baseline has to be
    picked before a correlogram exists. A file that names one in its header
    has already decided and is left alone; the setting fills only the
    "ask the user" case. Settings strings are deliberately not validated
    against an allow-list (see app/settings.py), so an unrecognized value
    falls back to the parser's own default rather than failing the load.

    Returns a list of human-readable strings, one per preview where the
    requested baseline could NOT be applied. Falling back silently would
    leave the Settings panel reading "Measured" while the numbers came from
    the calculated baseline. The default fits absorb that difference, but the
    linear cumulant does not, so the caller reports these rather than
    swallowing them.
    """
    choice = getattr(controller.settings, 'brookhaven_dat_baseline', None)
    fallbacks = []
    for preview in previews:
        if not isinstance(preview, BrookhavenDATPreview):
            continue
        if preview.baseline_is_file_specified or not choice:
            continue
        if choice == preview.baseline_choice:
            continue                          # already what the parser applied
        try:
            preview.set_baseline_choice(choice)
        except ParseError as exc:
            # This arm knowingly absorbs MalformedDataError too, and it is the only
            # place in the tree that does. set_baseline_choice re-runs apply_data_form,
            # whose data raises are now the subclass -- but a Settings-driven
            # re-normalization that will not take belongs in THIS channel (the file
            # loaded fine under a different baseline), not in the malformed-file dialog
            # that says nothing was loaded.
            fallbacks.append(
                f'{os.path.basename(preview.source_file or "?")}: could not use '
                f'the {choice!r} baseline ({exc}). Used the '
                f'{preview.baseline_choice} baseline instead.')
    return fallbacks


def load_dls_files(controller: "Controller", paths: Sequence[str], *,
                   generic_fallback: Callable[[str], Optional[list]]) -> LoadReport:
    """Load DLS correlogram files, each auto-detected independently.

    `generic_fallback(path)` reads a file no instrument parser recognised as a plain
    two-column table, prompting for its units: it returns the previews, `[]` when the
    file is not a numeric table, or None when there is nothing to load and the user has
    already been dealt with (a cancelled prompt, a rejected choice). A
    `MalformedDataError` it raises is reported with the recognised-but-malformed files.
    """
    first_id, n_files, n_meas, unreadable = None, 0, 0, []
    baseline_fallbacks = []
    parse_notes = []                     # passive load-time ⓘ notices from parsers
    malformed = []                       # recognised, but its data is unusable
    for path in paths:
        # Instrument-agnostic: auto-detect (ALV, Zetasizer, Brookhaven…), then
        # fall back to a plain two-column generic file (with a units prompt).
        det = autodetect(path, DLS_PARSERS)
        previews = det.previews
        if not previews:
            if det.malformed is not None:
                # A parser RECOGNISED this file and diagnosed its data. Its message is
                # the answer, so it goes to the user rather than being swallowed --
                # and we do NOT fall through to the generic parser, which would either
                # re-fail with a worse message or, worse, accept an instrument file as
                # a plain two-column table behind a units prompt and load it wrongly.
                malformed.append(det.reason(os.path.basename(path)))
                continue
            try:
                previews = generic_fallback(path)
            except MalformedDataError as exc:
                malformed.append(f'{os.path.basename(path)}: {exc}')
                continue
        if previews is None:             # canceled the generic prompt -> skip
            continue
        if not previews:
            unreadable.append(os.path.basename(path))
            continue
        collect_parse_notes(previews, path, parse_notes)
        baseline_fallbacks += apply_dat_baseline_setting(controller, previews)
        for preview in previews:
            raw, params = loaded_inputs_from_preview(preview)
            # What the file itself recorded (vendor mean count rate, run
            # length), where the format carries it -- offered, never applied.
            seed_vendor_records(params, preview)
            iid = controller.add_loaded('dls', raw, params, source_path=path)
            # Formats that ship the count-rate trace in the same file (ALV .ASC)
            # hand it over on the preview; store it linked to the measurement we
            # just made. This is the confirmation step the trace parsers defer
            # the measurement_id back-reference to.
            if getattr(preview, 'trace_preview', None) is not None:
                tid = controller.add_trace_from_preview(
                    preview.trace_preview, source_path=path, measurement_id=iid)
                # Through attach_trace, so a file-borne link and a hand-made one
                # seed the mean count rate by exactly one code path; then commit
                # that seed, so a load arrives clean like every other one rather
                # than showing "changes pending" before the user has touched it.
                controller.attach_trace(tid, iid)
            # Unconditional: a format may record its own mean count rate without
            # shipping a trace (Brookhaven .dat), and that seed is due at load too.
            # After the attach above, so a trace still outranks a vendor scalar.
            controller.seed_mean_count_rate_at_load(iid)
            first_id = first_id or iid
        n_files += 1
        n_meas += len(previews)
    return LoadReport(first_id=first_id, n_files=n_files, n_meas=n_meas,
                      unreadable=unreadable, malformed=malformed,
                      parse_notes=parse_notes, baseline_fallbacks=baseline_fallbacks)


def load_sls_files(controller: "Controller", paths: Sequence[str]) -> LoadReport:
    """Load SLS intensity files, each auto-detected, a plain two-column table the fallback."""
    first_id, n_files, n_meas, unreadable = None, 0, 0, []
    parse_notes = []                     # passive load-time ⓘ notices from parsers
    malformed = []                       # recognised, but its data is unusable
    for path in paths:
        det = autodetect(path, SLS_PARSERS)
        previews = det.previews
        if not previews:
            if det.malformed is not None:
                malformed.append(det.reason(os.path.basename(path)))
                continue
            try:                         # fall back to a plain two-column table
                previews = GenericSLSParser().parse(path)
            except MalformedDataError as exc:
                # MUST precede the ParseError arm. Unlike the generic DLS parser this
                # one builds its SLSFilePreview inside parse(), so __post_init__'s
                # require_finite_data fires HERE -- a plain SLS file with an inf
                # intensity is a live route to this handler, not a defensive one.
                malformed.append(f'{os.path.basename(path)}: {exc}')
                continue
            except (ParseError, FileNotFoundError):
                previews = None
        if not previews:
            unreadable.append(os.path.basename(path))
            continue
        for p in previews:
            raw, params = loaded_inputs_from_preview(p)
            iid = controller.add_loaded('sls', raw, params, source_path=path)
            first_id = first_id or iid
        collect_parse_notes(previews, path, parse_notes)   # e.g. negative intensities
        n_files += 1
        n_meas += len(previews)
    return LoadReport(first_id=first_id, n_files=n_files, n_meas=n_meas,
                      unreadable=unreadable, malformed=malformed, parse_notes=parse_notes)


def detect_trace(path: str) -> Detection:
    """The auto-detection over the trace roster.

    `skip_missing_file=True`: the trace route skips a missing file, where the DLS
    and SLS routes let FileNotFoundError propagate.
    """
    return autodetect(path, TRACE_PARSERS, skip_missing_file=True)


def load_trace_files(controller: "Controller", paths: Sequence[str], *,
                     detect: Callable[[str], Optional[Detection]],
                     generic_fallback: Callable[[str], Optional[list]]) -> LoadReport:
    """Load count-rate trace files, each auto-detected independently.

    `detect(path)` returns a `Detection`, or None when a parser crashed and the user has
    already been told (the file is skipped). `generic_fallback(path)` reads a plain
    two-column trace with its units prompt: the previews, `[]` when the file is not a
    numeric table, or None when the user cancelled the prompt, which ends the load and
    keeps the traces already stored.
    """
    last = None
    unreadable = []
    parse_notes = []                         # passive load-time ⓘ notices from parsers
    malformed = []                           # recognised, but its data is unusable
    for path in paths:
        det = detect(path)
        if det is None:                      # a parser crashed; already reported
            continue
        previews = det.previews
        if not previews:                     # no instrument matched
            if det.malformed is not None:
                # Recognised as a trace, then diagnosed. Its message is the answer.
                malformed.append(det.reason(os.path.basename(path)))
                continue
            previews = generic_fallback(path)
            if previews is None:             # user canceled the units prompt
                break                        # keep already-committed traces visible
            if not previews:
                unreadable.append(os.path.basename(path))
                continue
        collect_parse_notes(previews, path, parse_notes)
        try:
            for p in previews:
                last = controller.add_trace_from_preview(p, source_path=path)
        except MalformedDataError as exc:
            # THIS route validates LATE. Its two sibling fallbacks surface a bad
            # column inside the guarded call -- the DLS one inside
            # `apply_data_conversion`, the SLS one inside `parse()` -- but a
            # GenericTraceFilePreview holds its raw columns and defers the unit
            # conversion to `build()`, which the CONTROLLER calls. Without this arm the
            # diagnosis arrives outside every `try` this loader has and leaves the Qt
            # slot as an uncaught exception with no message box at all. Arm order: it
            # must precede any `except ParseError`, and there is none here.
            malformed.append(f'{os.path.basename(path)}: {exc}')
            continue
    return LoadReport(unreadable=unreadable, malformed=malformed, parse_notes=parse_notes,
                      last_trace_id=last)
