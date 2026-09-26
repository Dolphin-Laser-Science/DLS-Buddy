"""
parsers/alv_asc.py
==================

Parser for ALV correlator ``.ASC`` exports (ALV-5000/6000/7000 family). Two
header layouts are supported and auto-detected:

* **Multi-angle** (e.g. **ALV-7012 CGS-12F**): one run captures several detection
  angles at once, with per-detector keys ``Angle(1)``..``Angle(12)`` and one data
  column per detector. Yields one preview per active angle.
* **Single-angle** (e.g. **ALV-7004/USB**): one file is one angle, with a bare
  ``Angle [deg]`` key and a single active data column (CH0, column 1). Yields one
  preview. Detected only when no indexed ``Angle(i)`` key is present.

Its layout:

    <line 1>          instrument / mode descriptor
    Key : value       header metadata (temperature, viscosity, n, wavelength,
                      Angle(1..12), MeanCR1..12, dark counts, ...)
    "Correlation"     lag time + g2(tau)-1 for each of the 12 detector channels
    "Count Rate"      time + count rate (kHz) for each of the 12 channels
    "Cumulant ..."    the ALV's own per-detector cumulant fits (IGNORED -- this
                      platform computes its own; vendor fit results are display-only)

Because one file holds both a correlogram *and* a count-rate trace for every
active angle, it feeds two of the common-model object types. Mirroring the
two-layer parser design (and the Brookhaven DLS/trace split), there are two
parsers that share one file read:

  * ``ALVCorrelatorParser`` -> one ``DLSFilePreview`` per active angle (g2-1 curve)
  * ``ALVTraceParser``      -> one ``TraceFilePreview`` per active angle (count rate)

An "active" channel is one whose ``Angle(i)`` is non-zero (the ALV sets unused
detectors to angle 0). Both parsers leave the **sample identity** (polymer,
solvent, concentration) empty -- the ``.ASC`` file does not record it; the user
supplies it at the confirmation step, exactly as for the Brookhaven DLS parser.

Units (converted to the platform's canonical units here, in the parser)
-----------------------------------------------------------------------
* **Lag time: milliseconds -> seconds.** The ALV stores the correlation lag in
  **ms** (the shortest lag ``2.5e-5`` ms = 25 ns matches the ``25 ns STC`` sample
  time in the Mode string). This is the easy-to-miss one -- reading it as seconds
  would scale every Rh by 1000.
* Temperature: kelvin (kept). Viscosity: centipoise -> Pa.s. Wavelength: nm
  (kept). Angle: degrees (kept). Count rate: kHz -> cps (x1000). Count-rate trace
  time: seconds (kept; 0..Duration).

The file is read as latin-1: the ``Angle(i)[deg]`` keys carry a degree symbol
(byte 0xB0) that is not valid UTF-8. Keys are matched by prefix, so the exact
bytes of the unit bracket do not matter.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from parsers.base_parser import (
    BaseDLSParser,
    BaseTraceParser,
    DLSFilePreview,
    TraceFilePreview,
    MalformedDataError,
    ParseError,
    carried_run_duration,
    convert_delay_times,
    convert_viscosity,
)

_ENCODING = 'latin-1'
_KHZ_TO_CPS = 1000.0                 # 1 kHz of photon counts = 1000 counts/s
_SECTION_CORRELATION = '"Correlation"'
_SECTION_COUNT_RATE = '"Count Rate"'
_ANGLE_KEY_RE = re.compile(r'^Angle\((\d+)\)')
_MEANCR_KEY_RE = re.compile(r'^MeanCR(\d+)\b')


# ---------------------------------------------------------------------------
# Shared parse (one file read, used by both the correlator and trace parsers)
# ---------------------------------------------------------------------------

@dataclass
class _ALVChannel:
    """One active detector channel within an ALV run."""
    index: int                       # 1-based detector index (column in the data)
    angle_deg: float
    correlogram: np.ndarray          # g2(tau) - 1
    count_rates_cps: np.ndarray      # count-rate trace, converted to cps
    # The file's OWN mean count rate for this channel (its MeanCR scalar), in cps.
    # None when the file did not record a usable one. A record of what the vendor
    # reported, never a substitute for the trace's own mean.
    mean_count_rate_cps: Optional[float] = None


@dataclass
class _ALVData:
    """Everything extracted from one ALV .ASC file (common, canonical units)."""
    instrument: str
    sample_label: Optional[str]
    temperature_K: Optional[float]
    viscosity_Pa_s: Optional[float]
    refractive_index: Optional[float]
    wavelength_nm: Optional[float]
    delay_times_s: np.ndarray        # correlation lag, converted ms -> s
    trace_times_s: np.ndarray        # count-rate trace time (s)
    channels: List[_ALVChannel]
    source_file: str
    # The run's NOMINAL length from the header, in seconds. Not the trace's own
    # span: a file reporting 10 s carries a Count Rate section covering 9.88 s.
    duration_s: Optional[float] = None
    # The same length as the correlogram's run duration T -- the input the exact-in-Gamma
    # weight's 1/(M - k) reads -- but only for a ONE-run acquisition (`Runs : 1`). With
    # several runs the header does not say whether `Duration` is
    # one run's length or the total the averaged correlogram was built over, so nothing
    # is carried and the weight takes its T -> infinity form, which the fit records.
    run_duration_s: Optional[float] = None
    # Passive load-time notices (a data block cut short by a bad row), carried onto
    # every preview built from this file.
    notes: tuple = ()


def _read_lines(file_path: str) -> List[str]:
    """Read the file as latin-1. A missing file is FileNotFoundError (the contract);
    any other OS refusal -- a file the acquisition software still holds open, a share
    the user can list but not read -- is a plain ParseError, the one class `autodetect`
    skips past. This parser is first in the DLS list, so a bare OSError here escaped
    the whole load, for every format."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path!r}")
    try:
        with open(file_path, encoding=_ENCODING, newline='') as fh:
            return fh.read().splitlines()
    except OSError as exc:
        raise ParseError(f"Could not read {file_path!r}: {exc}") from exc


def _find_section(lines: List[str], marker: str, start: int = 0) -> int:
    for i in range(start, len(lines)):
        if lines[i].strip() == marker:
            return i
    raise MalformedDataError(f"ALV file is missing its {marker} section.")


def _parse_header_block(header_lines: List[str]) -> Dict[str, str]:
    """`Key : value` lines -> {stripped key: stripped value}, split on the FIRST
    colon (so a value like a "2:05:10 PM" timestamp is preserved)."""
    header: Dict[str, str] = {}
    for line in header_lines:
        if ':' not in line:
            continue
        key, _, value = line.partition(':')
        key = key.strip()
        if key:
            header.setdefault(key, value.strip())
    return header


def _get(header: Dict[str, str], prefix: str) -> Optional[str]:
    """First header value whose key starts with `prefix` (keys carry unit
    brackets, e.g. 'Temperature [K]')."""
    for key, value in header.items():
        if key.startswith(prefix):
            return value
    return None


def _mean_count_rates_cps(header: Dict[str, str]) -> List[Optional[float]]:
    """The file's own per-channel mean count rates, in cps, in COLUMN order.

    The MeanCR keys are POSITIONAL, not detector-numbered: the k-th key (sorted by
    its integer suffix) describes Count Rate column k, whatever that suffix happens
    to be. Measured 2026-08-14 against all three real ALV families -- the ALV-7012
    multi-angle files number them `MeanCR1..12` while both ALV-7004 families start
    at `MeanCR0`, so reading `MeanCR<detector index>` picks the neighbouring
    detector on a 7004 file and lands 30 % off its own trace.

    A zero, negative or non-finite entry reads as ABSENT rather than as a count
    rate of zero: 7004 files write `MeanCR2/3 = 0.000` for detectors they do not
    use. Matching is by regex, never by _get's prefix rule -- 'MeanCR1' is a prefix
    of 'MeanCR12'.
    """
    indexed: List[Tuple[int, Optional[float]]] = []
    for key, text in header.items():
        m = _MEANCR_KEY_RE.match(key)
        if m:
            indexed.append((int(m.group(1)), _to_float(text)))
    out: List[Optional[float]] = []
    for _, khz in sorted(indexed, key=lambda pair: pair[0]):
        if khz is None or not np.isfinite(khz) or khz <= 0.0:
            out.append(None)
        else:
            out.append(khz * _KHZ_TO_CPS)
    return out


def _to_float(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    try:
        return float(text.strip().strip('"'))
    except ValueError:
        return None


def _read_numeric_block(lines: List[str], start: int) -> Tuple[np.ndarray, int, bool]:
    """Read a contiguous block of whitespace-separated numeric rows from `start`.

    Stops at the first blank line, a quoted section header, or a non-numeric line.
    Returns (rows array, index of the stopping line, cut_short). Rows must share a
    column count; a ragged row ends the block. `cut_short` is True when the stop was
    a non-numeric or ragged ROW rather than the block's natural end (blank line,
    header, end of file) -- the one case where measured data after the stop is
    silently left unread, which the caller discloses.
    """
    rows: List[List[float]] = []
    ncols: Optional[int] = None
    i = start
    cut_short = False
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith('"'):
            break
        parts = stripped.split()
        try:
            values = [float(p) for p in parts]
        except ValueError:
            cut_short = True
            break
        if ncols is None:
            ncols = len(values)
        elif len(values) != ncols:
            cut_short = True
            break
        rows.append(values)
        i += 1
    if not rows:
        raise MalformedDataError(
            f"ALV file: expected numeric data starting at line {start + 1}, "
            f"found none.")
    return np.asarray(rows, dtype=float), i, cut_short


def _parse_alv_file(file_path: str) -> _ALVData:
    lines = _read_lines(file_path)
    # ALV .ASC files open with a line like "ALV-7004 / USB Data" (or "ALV-7012...").
    # Sniff on the leading "ALV" token; take everything before " Data" as the
    # instrument name.
    if not lines or not lines[0].lstrip().upper().startswith('ALV'):
        first = lines[0] if lines else '<empty file>'
        raise ParseError(
            f"{file_path!r} is not an ALV .ASC file (first line: {first!r}).")

    # --- format commit point ---------------------------------------------------
    # Above this line a failure means "not my format" and stays a plain ParseError,
    # so autodetect can hand the file to the next parser. Below it the file has
    # proved it IS this format, so a failure is malformed DATA and raises
    # MalformedDataError, whose message is meant for the user.
    instrument = lines[0].split(' Data')[0].strip() or 'ALV'

    idx_corr = _find_section(lines, _SECTION_CORRELATION)
    header = _parse_header_block(lines[1:idx_corr])

    # active angles (channel index -> angle); unused detectors are angle 0
    angles: Dict[int, float] = {}
    for key, value in header.items():
        m = _ANGLE_KEY_RE.match(key)
        if m:
            a = _to_float(value)
            if a is not None:
                angles[int(m.group(1))] = a
    # Single-angle instruments (e.g. ALV-7004/USB) write a bare `Angle [deg]`
    # key instead of the multi-detector `Angle(1..12)` keys. Fall back to it only
    # when no indexed key matched, and treat it as the one channel in column 1
    # (CH0) -- the rest of the pipeline is index-driven and works unchanged.
    if not angles:
        a = _to_float(_get(header, 'Angle'))
        if a is not None:
            angles[1] = a
    active = sorted(i for i, a in angles.items() if a != 0.0)
    if not active:
        raise MalformedDataError(
            f"{file_path!r}: no active detection angles "
            f"(no non-zero Angle(i) or bare Angle key).")

    # correlation block: lag (ms) + one g2-1 column per detector
    corr, after_corr, corr_cut = _read_numeric_block(lines, idx_corr + 1)
    idx_cr = _find_section(lines, _SECTION_COUNT_RATE, after_corr)
    cr, after_cr, cr_cut = _read_numeric_block(lines, idx_cr + 1)

    # A block that ended on a bad row (not its blank-line / header end) has rows
    # after the stop that were never read: a truncated correlogram or trace, which
    # every fit would take at face value. Disclosed keep-both -- on `notes` for the
    # GUI's passive box and as a UserWarning for a headless caller.
    found_notes = []
    for name, cut, block, stop in (('Correlation', corr_cut, corr, after_corr),
                                   ('Count Rate', cr_cut, cr, after_cr)):
        if cut:
            found_notes.append(
                f"the {name} block ended at line {stop + 1} on a row that is not "
                f"numeric or has a different number of columns; {block.shape[0]} row(s) "
                f"were read before it and the rest of the block was not loaded. If the "
                f"file was not edited by hand, re-export it."
            )
    notes = tuple(found_notes)
    for note in notes:
        warnings.warn(f"{file_path!r}: {note}", UserWarning, stacklevel=2)

    need_cols = max(active) + 1          # column 0 is lag/time, channel i is column i
    for name, block in (('Correlation', corr), ('Count Rate', cr)):
        if block.shape[1] < need_cols:
            raise MalformedDataError(
                f"{file_path!r}: the {name} block has {block.shape[1]} columns "
                f"but detector {max(active)} needs at least {need_cols}.")

    delay_times_s = convert_delay_times(corr[:, 0], 'ms')   # ms -> s (the gotcha)
    trace_times_s = cr[:, 0].astype(float)                  # already seconds

    # Channel i reads Count Rate column i, so it takes the i-th positional MeanCR.
    # A list too short to reach it yields None rather than a guess: a short list
    # means the pairing is unknown, and a wrong mean count rate is worse than none.
    mean_crs = _mean_count_rates_cps(header)
    channels = [
        _ALVChannel(
            index=i, angle_deg=angles[i],
            correlogram=corr[:, i].astype(float),
            count_rates_cps=cr[:, i].astype(float) * _KHZ_TO_CPS,
            mean_count_rate_cps=(mean_crs[i - 1] if i <= len(mean_crs) else None),
        )
        for i in active
    ]

    duration_s = _to_float(_get(header, 'Duration'))
    if duration_s is not None and (not np.isfinite(duration_s) or duration_s <= 0):
        duration_s = None
    run_duration_s = duration_s if _to_float(_get(header, 'Runs')) == 1.0 else None

    visc = _to_float(_get(header, 'Viscosity'))
    label = _get(header, 'Samplename')
    label = label.strip().strip('"') if label else ''

    # `is not None` (not `if visc`): a header viscosity of 0.0 is PRESENT-but-invalid,
    # not absent, so it must reach convert_viscosity to be rejected — and its
    # ValueError (0/negative, or a bad unit) surfaces as a ParseError, this layer's
    # contract, rather than a bare ValueError leaking to the caller.
    try:
        viscosity_Pa_s = convert_viscosity(visc, 'cp') if visc is not None else None
    except ValueError as exc:
        raise MalformedDataError(
            f"{file_path!r}: invalid viscosity {visc!r} cP in the ALV header ({exc})."
        ) from exc

    return _ALVData(
        instrument=instrument,
        sample_label=label or None,
        temperature_K=_to_float(_get(header, 'Temperature')),
        viscosity_Pa_s=viscosity_Pa_s,
        refractive_index=_to_float(_get(header, 'Refractive Index')),
        wavelength_nm=_to_float(_get(header, 'Wavelength')),
        delay_times_s=delay_times_s,
        trace_times_s=trace_times_s,
        channels=channels,
        source_file=os.path.abspath(file_path),
        duration_s=duration_s,
        run_duration_s=run_duration_s,
        notes=notes,
    )


def _trace_preview_for(data: _ALVData, ch: _ALVChannel) -> TraceFilePreview:
    """The count-rate trace of one channel, labelled with its angle.

    One builder for both parsers, so a trace ingested through the DLS load and the
    same trace loaded on its own can never end up with different labels or units.
    """
    tag = f'{ch.angle_deg:g}°'
    label = f'{data.sample_label} @ {tag}' if data.sample_label else tag
    return TraceFilePreview(
        source_file=data.source_file,
        instrument_name=data.instrument,
        times_s=data.trace_times_s.copy(),
        count_rates_cps=ch.count_rates_cps.copy(),
        sample_label=label,
        notes=data.notes,
    )


# ---------------------------------------------------------------------------
# Public parsers
# ---------------------------------------------------------------------------

class ALVCorrelatorParser(BaseDLSParser):
    """ALV ``.ASC`` -> one ``DLSFilePreview`` per active angle (the g2-1 curve).

    Sample identity (polymer/solvent/concentration) is not in the file and is left
    for the confirmation step. The several angles of one file share temperature,
    viscosity, wavelength, and refractive index, and group into a single sample.

    An ALV file carries the count-rate trace beside the correlogram, so each
    preview also carries ITS OWN channel's trace (``trace_preview``) and the
    vendor's mean count rate and nominal duration -- the duration also as the run
    duration ``duration_s`` when the file records one run. The loader links that trace to
    the measurement it creates. Pairing is structural rather than positional --
    both halves come off one ``_ALVChannel``, so no zip across two independently
    filtered channel lists can attach one angle's trace to another angle's curve.
    """

    def parse(self, file_path: str) -> List[DLSFilePreview]:
        data = _parse_alv_file(file_path)
        # The run duration is decided PER CHANNEL:
        # the angles of one file share a `Duration` and a delay axis, but each carries its
        # own correlogram, and the measurement is the unit the weight runs on. The rule is
        # `base_parser.carried_run_duration`, the same one the Brookhaven reader applies --
        # spelled here on `g2 - 1 > -1`, which is what this format ships.
        durations, duration_notes = [], []
        for ch in data.channels:
            carried, note = carried_run_duration(
                data.run_duration_s, data.delay_times_s,
                np.asarray(ch.correlogram, dtype=float) > -1.0,
                "the header's Duration for a one-run acquisition")
            durations.append(carried)
            duration_notes.append(note)          # '' where the duration was carried
        # The note rides ONLY the previews that actually withheld a duration. The angles
        # of one file share a `Duration` and an axis but not their liveness, so a
        # file-wide note lands on angles whose duration WAS carried and tells them the
        # opposite of what their own result records.
        base_notes = tuple(data.notes)
        previews = [
            DLSFilePreview(
                source_file=data.source_file,
                instrument_name=data.instrument,
                delay_times_s=data.delay_times_s.copy(),
                correlogram=ch.correlogram.copy(),
                sample_label=data.sample_label,
                angle_deg=ch.angle_deg,
                wavelength_nm=data.wavelength_nm,
                solvent_refractive_index=data.refractive_index,
                temperature_K=data.temperature_K,
                viscosity_Pa_s=data.viscosity_Pa_s,
                vendor_mean_count_rate_cps=ch.mean_count_rate_cps,
                vendor_duration_s=data.duration_s,
                duration_s=carried,
                trace_preview=_trace_preview_for(data, ch),
                notes=base_notes + ((note,) if note else ()),
            )
            for ch, carried, note in zip(data.channels, durations, duration_notes,
                                         strict=True)
        ]
        return previews


class ALVTraceParser(BaseTraceParser):
    """ALV ``.ASC`` -> one ``TraceFilePreview`` per active angle (count-rate trace).

    Each trace is labeled with its angle. This is the surface for loading an ALV
    file's traces ON THEIR OWN; loading the same file as a correlogram links each
    trace to the measurement it belongs to instead (the confirmation step this
    parser leaves the ``measurement_id`` back-reference for). Count rates are
    converted kHz -> cps.
    """

    def parse(self, file_path: str) -> List[TraceFilePreview]:
        data = _parse_alv_file(file_path)
        return [_trace_preview_for(data, ch) for ch in data.channels]
