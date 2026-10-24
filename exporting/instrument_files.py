"""
exporting/instrument_files.py
=============================

Loadable instrument files from the synthetic generator's artifacts, and the dataset
orchestrators that write a whole ``test-data/Synthetic`` folder.

This is an I/O module: its functions write (or, for the depolarized pair, read back) files
and return the paths. The artifacts they take are built by :mod:`analysis.synthetic_dataset`,
which holds the forward models, the builders, the default test system and the seeding
convention.

What lives here
---------------
* **Writers**: a built artifact as a file a real parser loads. ALV ``.ASC`` for a
  multi-angle DLS set (:func:`write_alv_asc`); the two Brookhaven SLS ``.csv`` layouts
  (:func:`write_brookhaven_sls`, :func:`write_brookhaven_sls_scan`); the Brookhaven raw
  ASCII ``.dat`` (:func:`write_brookhaven_dat`); a two-column count-rate trace
  (:func:`write_trace_csv`); the generic DLS parser's two-column correlogram CSV
  (:func:`export_synthetic_correlogram_csv`, with :func:`write_kww_csv` its adapter for a
  stretched-exponential set); and the corpus's two-line VV/VH stand-in file with its reader
  (:func:`write_depolarized_pair_csv`, :func:`read_depolarized_pair_csv`).
* **Dataset orchestrators**: regenerate a committed synthetic folder from its fixed recipe
  and seeds (:func:`generate_full_dataset`, :func:`generate_alpha_selection_dataset`,
  :func:`generate_architecture_dataset`, :func:`generate_form_factor_dataset`,
  :func:`generate_mode_resolved_dataset`), or write a built mode-resolved tuple as a
  loadable folder (:func:`save_mode_resolved_dataset`). Picture rendering is not here: it
  needs the plotting layer, and a developer wrapper adds it.

Why here: the analysis package performs no file I/O (architecture invariant 1), and the
parsers may not import the analysis types these writers take (invariant 2), so they sit in
the export layer, which the Qt-free controller already calls. What keeps a writer honest is
the round trip through the real parser, not adjacency to it.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from analysis.synthetic_dataset import (
    ALPHA_SELECTION_CASES, DEFAULT_DLS_ANGLES, DEFAULT_DLS_CONCS_MG, DEFAULT_DLS_SINGLE_MG,
    DEFAULT_DN_DC, DEFAULT_N_SOLVENT, DEFAULT_SERIES, DEFAULT_SLS_ANGLES,
    DEFAULT_SLS_CONCS_MG, DEFAULT_SLS_SCAN_MG, DEFAULT_SLS_SCAN_SERIES,
    DEFAULT_SOLVENT_INTENSITY_90, DEFAULT_TEMPERATURE_C, DEFAULT_TEMPERATURE_K,
    DEFAULT_VISCOSITY_CP, DEFAULT_VISCOSITY_PA_S, DEFAULT_WAVELENGTH_NM, PROFILES,
    CalibrationSpec, DepolarizedPair, ModeResolvedTuple, MultiAngleDLS, SyntheticKWW,
    SyntheticSLSSet, SyntheticTrace,
    build_architecture_series, build_count_rate_trace, build_form_factor_series,
    build_mode_resolved_tuple, build_multi_angle_dls, build_sls_set, rh_at_concentration,
)
from analysis.utilities import SyntheticCorrelogramResult, SyntheticPopulation


# ===========================================================================
# Writers (a built artifact -> a loadable instrument file)
# ===========================================================================
def write_alv_asc(path: str, dls: MultiAngleDLS, *,
                  temperature_K: Optional[float] = None,
                  viscosity_cp: Optional[float] = None) -> str:
    """Write a MultiAngleDLS as an ALV ``.ASC`` (correlation block + count-rate block).

    Mirrors the ALV multi-angle format the project's ALV parser reads: the lag in
    the Correlation block is in **ms**, the Count Rate block in **kHz**.
    """
    angles = dls.angles_deg
    T_K = float(temperature_K if temperature_K is not None else dls.temperature_K)
    eta_cp = float(viscosity_cp if viscosity_cp is not None
                   else dls.viscosity_Pa_s * 1.0e3)
    delay = dls.delay_times_s
    t = dls.trace_times_s

    L = ['ALV-7012 CGS-12F Data, Synthetic (DLS Buddy test data)',
         'Date :\t"1/1/2026"', 'Time :\t"12:00:00 PM"',
         f'Samplename : \t"{dls.label}, c = {dls.concentration_g_per_mL * 1e3:g} mg/mL"']
    L += [f'SampMemo({i}) : \t""' for i in range(10)]
    L += [f'Temperature [K] :\t{T_K:.5f}', f'Viscosity [cp]  :\t{eta_cp:.5f}',
          f'Refractive Index:\t{dls.n_solvent:.5f}',
          f'Wavelength [nm] :\t{dls.wavelength_nm:.5f}']
    L += [f'Angle({i})[deg]     :\t{a:.5f}' for i, a in enumerate(angles, 1)]
    L += [f'Duration [s]    :\t{float(dls.run_duration_s)}', 'Runs            :\t1',
          'Mode            :\t"synthetic"']
    L += [f'MeanCR{i} [kHz]   :\t{dls.mean_cr_kHz[a]:.5f}' for i, a in enumerate(angles, 1)]
    L += [f'DC[{i}][kHz]      :\t0.00000' for i in range(1, len(angles) + 1)]
    L += ['', '"Correlation"']
    for k in range(delay.size):
        row = [f'{delay[k] * 1000:.5E}'] + [f'{dls.signals[a][k]:.5E}' for a in angles]
        L.append('  ' + '\t'.join(row))
    L.append('"Count Rate"')
    for k in range(t.size):
        row = [f'{t[k]:.5f}'] + [f'{dls.trace_cps[a][k] / 1000:.5f}' for a in angles]
        L.append('  ' + '\t'.join(row))
    with open(path, 'w', encoding='latin-1', newline='\n') as fh:
        fh.write('\n'.join(L) + '\n')
    return path


def write_brookhaven_sls(path: str, sls: SyntheticSLSSet) -> str:
    """Write a SyntheticSLSSet as a Brookhaven SLS ``.csv`` (key,value rows).

    Calibration metadata (constant, standard, Rayleigh) is written only when the
    set is calibrated (``cal.include_in_files``); an uncalibrated set omits it, so
    a user who loads it without entering a calibrant gets the uncalibrated flag. The
    dn/dc row is written only when ``sls.dn_dc_in_files``; a dn/dc-less set omits it,
    so the parser leaves dn/dc None and the analysis runs on the arbitrary optical scale.
    """
    angles = [float(a) for a in sls.angles_deg]
    concs_mg = [c * 1e3 for c in sls.concentrations_g_per_mL]
    rows: List[Tuple[str, str]] = [
        ('Date/Time', '1/1/2026 12:00:00 PM'),
        ('Sample ID', sls.label or f'{sls.polymer_name} in {sls.solvent_name}'),
        ('Operator ID', 'Synthetic'),
        ('Sample Liquid', sls.solvent_name or 'Water'),
        ('Refractive Index of Sample Liquid', f'{sls.n_solvent}'),
    ]
    if sls.dn_dc_in_files:
        rows.append(('Refractive Index Inc. (dn/dc) (mL/g)', f'{sls.dn_dc}'))
    if sls.calibrated:
        rows += [
            ('Calibration Constant', f'{sls.cal.k_c():.4E}'),
            ('Calibration Liquid', 'Toluene'),
            ('Refractive Index of Calibration Liquid', f'{sls.cal.n_standard}'),
            ('Rayleigh Ratio of Calibration Liquid', f'{sls.cal.rayleigh():.4E}'),
        ]
    rows += [('Wavelength (nm)', f'{sls.wavelength_nm:g}'),
             ("'A' Dark Count Rate", '0'),
             ('Number of Angles Measured', str(len(angles)))]
    rows += [(f'Angle {i} (degrees)', f'{a:g}') for i, a in enumerate(angles, 1)]
    rows += [('Number of Concentrations Measured', str(len(concs_mg)))]
    rows += [(f'Concentration {j} (mg/mL)', f'{c:g}') for j, c in enumerate(concs_mg, 1)]
    for j, c in enumerate(sls.concentrations_g_per_mL, 1):
        col = sls.intensities[c]
        for i in range(len(angles)):
            rows.append((f'Intensity - Concentration {j} - Angle {i + 1}', f'{col[i]:.0f}'))
    with open(path, 'w', encoding='latin-1', newline='\n') as fh:
        fh.write('\n'.join(f'{k},{v}' for k, v in rows) + '\n')
    return path


def write_brookhaven_sls_scan(path: str, sls: SyntheticSLSSet,
                              conc_g_per_mL: float) -> str:
    """Write ONE concentration of a SyntheticSLSSet as a Brookhaven angular-scan ``.csv``.

    The instrument's second SLS layout (``BrookhavenSLSScanParser``): a single
    acquisition plus the solvent blank measured in the same run, with no concentration
    table and no dn/dc. The set's ``c = 0`` column becomes ``Liquid Intensity`` and the
    column at ``conc_g_per_mL`` becomes ``Sample Intensity``.

    Deliberately omits the calibration block as well as dn/dc: layout B carries neither,
    so a file that shipped them would exercise a path no real export produces.

    Raises
    ------
    ValueError
        If the set has no ``c = 0`` blank, or no column at ``conc_g_per_mL`` -- either
        way the caller asked for a file that cannot represent this layout.
    """
    if 0.0 not in sls.intensities:
        raise ValueError('layout B needs the set\'s c = 0 blank; this set has none.')
    if conc_g_per_mL not in sls.intensities:
        raise ValueError(
            f'no intensity column at c = {conc_g_per_mL!r} g/mL; '
            f'available: {sorted(sls.intensities)}.')
    angles = [float(a) for a in sls.angles_deg]
    rows: List[Tuple[str, str]] = [
        ('Date/Time', '1/1/2026 12:00:00 PM'),
        ('Sample ID', sls.label or f'{sls.polymer_name} in {sls.solvent_name}'),
        ('Operator ID', 'Synthetic'),
        ('Sample Liquid', sls.solvent_name or 'Water'),
        ('Refractive Index of Sample Liquid', f'{sls.n_solvent}'),
        ('Wavelength (nm)', f'{sls.wavelength_nm:g}'),
        ("'A' Dark Count Rate", '0'),
        ('Number of Angles Measured', str(len(angles))),
    ]
    rows += [(f'Angle {i} (degrees)', f'{a:g}') for i, a in enumerate(angles, 1)]
    blank = sls.intensities[0.0]
    sample = sls.intensities[conc_g_per_mL]
    rows += [(f'Liquid Intensity - Angle {i + 1}', f'{blank[i]:.1f}')
             for i in range(len(angles))]
    rows += [(f'Sample Intensity - Angle {i + 1}', f'{sample[i]:.1f}')
             for i in range(len(angles))]
    with open(path, 'w', encoding='latin-1', newline='\n') as fh:
        fh.write('\n'.join(f'{k},{v}' for k, v in rows) + '\n')
    return path


def write_trace_csv(path: str, trace: SyntheticTrace, *, count_rate_unit: str = 'kcps') -> str:
    """Write a count-rate trace as a plain two-column CSV (time s, count rate).

    The default writes kcps (matching the bundled trace test file); load it via the
    generic trace parser, choosing time = s and the matching count-rate unit.
    """
    factor = {'cps': 1.0, 'kcps': 1e-3, 'Mcps': 1e-6}[count_rate_unit]
    cps = np.asarray(trace.count_rates_cps, dtype=float) * factor
    t = np.asarray(trace.times_s, dtype=float)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(f'{ti:.2f},{ci:.3f}' for ti, ci in zip(t, cps, strict=True)) + '\n')
    return path


# Conversion factors: seconds -> output unit (multiply by these)
_DELAY_OUTPUT_FACTORS = {'s': 1.0, 'ms': 1.0e3, 'us': 1.0e6, 'ns': 1.0e9}


def export_synthetic_correlogram_csv(
    result: SyntheticCorrelogramResult,
    file_path: str,
    delay_unit: str = 's',
    delimiter: str = ',',
) -> str:
    """Write a synthetic correlogram as a two-column CSV for the generic parser.

    The output has no header rows and two columns -- delay time and the
    correlation value -- exactly matching the generic DLS parser's contract.
    Load it back with GenericDLSParser, setting delay_time_unit to match
    `delay_unit` and data_form to match the result's output_form (and beta or
    baseline_B if the form is 'g1' or 'g2').

    Parameters
    ----------
    result : SyntheticCorrelogramResult
        The output of generate_synthetic_correlogram().
    file_path : str
        Destination path for the CSV.
    delay_unit : str
        Unit to write the delay times in: 's' (default), 'ms', 'us', or 'ns'.
        When loading, set the generic parser's delay_time_unit to the same value.
    delimiter : str
        Column delimiter, ',' (default) or '\\t'.

    Returns
    -------
    str
        The file path written.

    Raises
    ------
    ValueError
        If delay_unit is unrecognized.
    """
    if delay_unit not in _DELAY_OUTPUT_FACTORS:
        raise ValueError(
            f"delay_unit must be one of {sorted(_DELAY_OUTPUT_FACTORS)}, "
            f"got {delay_unit!r}."
        )
    factor = _DELAY_OUTPUT_FACTORS[delay_unit]
    delays_out = result.delay_times_s * factor

    with open(file_path, 'w', newline='') as fh:
        for t, v in zip(delays_out, result.signal, strict=True):
            fh.write(f"{t:.10e}{delimiter}{v:.10e}\n")

    return file_path


def write_kww_csv(path: str, kww: SyntheticKWW, *, delay_unit: str = 's') -> str:
    """Write a :class:`SyntheticKWW` as the generic parser's two-column CSV.

    A THIN ADAPTER over :func:`export_synthetic_correlogram_csv`, not a second copy of it:
    that function already accepts a KWW result **at runtime** -- its body reads only
    ``delay_times_s`` and ``signal``, both of which :class:`SyntheticKWW` has -- so the
    only thing missing was a contract saying so, and duplicating the write loop to add one
    would put two spellings of the same format in the tree.

    Why a KWW set needs this at all: :class:`SyntheticKWW` is deliberately not a
    :class:`~analysis.utilities.SyntheticCorrelogramResult` (a stretched exponential is a
    continuous relaxation-time spectrum with no discrete population list), so
    :func:`write_alv_asc` cannot take one either -- and that one is not adaptable, because
    it needs angles, count-rate traces and per-angle mean count rates a single correlogram
    does not have.

    Load the result with ``GenericDLSParser``, setting ``delay_time_unit`` to match
    ``delay_unit`` and ``data_form`` to ``'g2m1'``.
    """
    return export_synthetic_correlogram_csv(kww, path, delay_unit=delay_unit)


# ---------------------------------------------------------------------------
# Brookhaven raw ASCII .dat (the writer axis: internal model -> vendor file)
# ---------------------------------------------------------------------------
# This writer sits beside the other writers in this module rather than in `parsers/`: that layer
# is declared as the one translating a native file INTO the common model (architecture
# invariant 2), and what keeps a writer honest is the round trip driving the REAL
# parser, not adjacency.

# Header positions this writer fills, by the manual's 1-based numbering (Appendix A,
# "Raw Data File Formats"). Named here rather than imported from the parser: the writer
# and the parser are meant to be independent transcriptions of one published layout, so
# that a round trip is a check on both rather than a tautology on one shared constant.
_DAT_N_PARAMS_LESS_ONE = 36          # item 17; header is this + 1 lines
_DAT_DELAY_MODE = -2                 # item 19, constant-ratio spacing (written bare)
_DAT_ANALYSIS_MODE = 2               # item 20, autocorrelation
_DAT_TRAILER_LINES = 4               # Sample ID, Operator ID, Date, Time
_DAT_BASELINE_MODE_CODE = {'prompt': 0, 'calculated': 1, 'measured': 2}


def _fmt_dat_scalar(value: float) -> str:
    """One header or data value in the vendor's '%.9e' form with a 3-digit exponent.

    Three digits ('1.000000000e+000') is what the real exports carry; Python's own '%e'
    emits two. The parser reads with ``float()`` and does not care, so this is
    faithfulness to the vendor's shape rather than a requirement.
    """
    text = f'{float(value):.9e}'
    mantissa, _, exponent = text.partition('e')
    sign, digits = exponent[0], exponent[1:]
    return f'{mantissa}e{sign}{int(digits):03d}'


#: The run duration a synthetic file states when its draw names none -- a TYPICAL DLS
#: acquisition (owner, 2026-09-12). Consumed by `write_brookhaven_dat`, whose format
#: cannot express "no duration": items 4 and 25 are always written, and since 2026-09-12
#: the parser reads them as Eq. (9)'s T. The builders' own `run_duration_s` default is
#: `analysis.synthetic_dataset.build_multi_angle_dls`'s.
_DEFAULT_SYNTHETIC_RUN_S = 60.0


def write_brookhaven_dat(path: str, corr: SyntheticCorrelogramResult, *,
                         baseline: float = 2.0e7,
                         baseline_mode: str = 'calculated',
                         viscosity_cp: Optional[float] = None,
                         n_samples: Optional[float] = None,
                         sampling_time_us: float = 5.0,
                         first_channel_used: int = 1,
                         n_baseline_channels: int = 6,
                         sample_label: str = 'synthetic',
                         operator: str = 'DLS Buddy',
                         acquired_date: str = '01/01/26',
                         acquired_time: str = '00:00:00') -> str:
    """Write a generated correlogram as a Brookhaven raw ASCII ``.dat`` export.

    The inverse of ``parsers/brookhaven_dat.py``'s normalization: that parser recovers
    ``C(tau) = contents(tau)/B - 1`` from raw correlator counts, so this writes
    ``contents = B * (1 + C)`` with ``B = baseline``. The point of having it is that the
    ``.dat``'s numerics were pinned only against whichever real file happened to stay
    committed -- a genuine numeric transform on the way to a result, with nothing
    asserting that it inverts, and one ``git rm`` from being pinned against nothing.

    ``corr`` must be in ``'g2m1'`` form. The other two forms are refused rather than
    converted: a ``'g2'`` column differs by exactly the 1 this format's normalization
    removes, so writing one would produce a file whose recovered correlogram is off by
    unity with nothing to say so, and ``'g1'`` is not an intensity correlation at all.

    ``baseline_mode`` sets header item 8 -- what the acquiring software intended.
    ``'calculated'`` and ``'measured'`` name one and the parser honours it; ``'prompt'``
    (code 0, what both real exports carry) leaves the choice to the application default,
    which is the case worth being able to generate.

    BOTH baselines are written and both are honest. Item 22, the calculated baseline, is
    ``baseline``, and the total counts are chosen so the manual's identity
    ``A * B / n_samples`` reproduces it. Item 23, the measured baseline, is the actual
    mean of the last ``n_baseline_channels`` written channels, computed from the ROUNDED
    counts so the file is self-consistent at its own precision -- so a reader that
    re-derives either baseline from the data gets the header value back, and
    ``set_baseline_choice('measured')`` returns the affine transform the parser's
    docstring describes rather than an arbitrary second number.

    Header slots this parser does not read are written as ``0.0``, and so is ONE it does:
    item 15, the particle refractive index, which ``parsers/brookhaven_dat.py`` reads onto
    the preview as ``solute_refractive_index_from_file`` and deliberately does not
    propagate (invariant 3 keeps a solute index hand-entered). A generated file therefore
    parses back with that field at 0.0 rather than at a real index, which is honest --
    the generator has no solute index to state -- but it is a READING, not an absent
    value, so do not treat a parsed 0.0 there as "the file said nothing". Everything else
    written as ``0.0`` is a slot the parser never looks at, and that half is the sheet's
    scope line: a writer emits what its parser reads, and inventing plausible values for
    fields whose meaning we have not established would be a fabrication dressed as
    fidelity.

    ``viscosity_cp`` defaults to the correlogram's own ``viscosity_Pa_s`` in centipoise,
    so the written file describes the medium the data was generated in.

    ``n_samples`` DEFAULTS FROM THE DRAW. Items
    4 and 25 are the run duration this format states -- ``n_samples * sampling_time_us`` --
    and since 2026-09-12 the parser reads them onto the measurement as Eq. (9)'s ``T``.
    A fixed default was harmless while nothing read them and a fabrication the moment
    something did: it implied a 54-minute acquisition whatever the draw actually was, so a
    correlogram drawn with ``duration_s=60`` round-tripped back claiming 3250 s. Left at
    ``None`` the count is derived from ``corr.duration_s``; a draw that states none takes
    ``_DEFAULT_SYNTHETIC_RUN_S``, a typical DLS run rather than a long one (owner,
    2026-09-12: a typical measurement is never half an hour). Passing a count explicitly
    still works and is not second-guessed.

    A default that does not cover the drawn lag axis is NOT refused here, because the
    reader already owns that boundary: a header disagreeing with its own data is exactly
    what ``base_parser.carried_run_duration`` withholds ``T`` for, with a note. A second
    predicate policing the same edge is the gap rule 12 says to remove, not to double.
    """
    if corr.output_form != 'g2m1':
        raise ValueError(
            f"write_brookhaven_dat needs a 'g2m1' correlogram -- the .dat's "
            f"contents/B - 1 normalization IS g2-1, so any other form would be written "
            f"as data it is not; got output_form={corr.output_form!r}.")
    if not (math.isfinite(baseline) and baseline > 0):
        raise ValueError(
            f"baseline must be a positive finite number of counts (the parser divides "
            f"by it); got {baseline!r}.")
    if baseline_mode not in _DAT_BASELINE_MODE_CODE:
        raise ValueError(
            f"baseline_mode must be one of {tuple(_DAT_BASELINE_MODE_CODE)}, got "
            f"{baseline_mode!r}.")
    tau_us = np.asarray(corr.delay_times_s, dtype=float) * 1.0e6
    n_channels = int(tau_us.size)
    if not (1 <= n_baseline_channels <= n_channels):
        raise ValueError(
            f"n_baseline_channels must be between 1 and the channel count "
            f"({n_channels}); got {n_baseline_channels!r}.")
    counts = baseline * (1.0 + np.asarray(corr.signal, dtype=float))

    # The measured baseline is the mean of the channels it names, read back at the
    # precision the file stores -- so re-deriving it from the written rows reproduces
    # the written header value instead of missing it by a rounding.
    written_counts = [_fmt_dat_scalar(c) for c in counts]
    tail = [float(x) for x in written_counts[n_channels - n_baseline_channels:]]
    baseline_measured = float(np.mean(tail))
    # Items 4 and 25 ARE the run duration this format states, so they come from the draw
    # rather than from a constant -- see the docstring.
    if n_samples is None:
        drawn = getattr(corr, 'duration_s', None)
        duration_s = float(drawn) if drawn else _DEFAULT_SYNTHETIC_RUN_S
        n_samples = duration_s / (float(sampling_time_us) * 1.0e-6)
    # A * B / n_samples must reproduce the calculated baseline (the manual's identity).
    total_counts = math.sqrt(float(baseline) * float(n_samples))

    visc = (viscosity_cp if viscosity_cp is not None
            else float(corr.viscosity_Pa_s) * 1.0e3)

    header = {
        1: 1.0,
        2: total_counts, 3: total_counts, 4: float(n_samples),
        6: float(tau_us[0]), 7: float(n_channels),
        8: float(_DAT_BASELINE_MODE_CODE[baseline_mode]),
        9: float(corr.angle_deg), 10: float(corr.wavelength_nm),
        11: float(corr.temperature_K), 12: float(visc),
        14: float(corr.solvent_refractive_index),
        17: float(_DAT_N_PARAMS_LESS_ONE), 18: float(first_channel_used),
        20: float(_DAT_ANALYSIS_MODE),
        22: float(baseline), 23: baseline_measured,
        24: float(tau_us[-1]), 25: float(sampling_time_us),
        35: float(n_channels - n_baseline_channels + 1), 36: float(n_channels),
    }
    lines = []
    for item in range(1, _DAT_N_PARAMS_LESS_ONE + 2):
        if item == 19:
            lines.append(str(_DAT_DELAY_MODE))       # item 19 is a bare int
        else:
            lines.append(_fmt_dat_scalar(header.get(item, 0.0)))
    for t_us, count_text in zip(tau_us, written_counts, strict=True):
        lines.append(f'{_fmt_dat_scalar(t_us)}, {count_text}')
    lines.extend([sample_label, operator, acquired_date, acquired_time])
    assert len(lines) == _DAT_N_PARAMS_LESS_ONE + 1 + n_channels + _DAT_TRAILER_LINES

    with open(path, 'w', encoding='latin-1', newline='\r\n') as fh:
        fh.write('\n'.join(lines) + '\n')
    return path


def write_depolarized_pair_csv(path: str, pair: DepolarizedPair) -> str:
    """Write a static VV/VH intensity pair as the corpus's two-line stand-in file.

        VV,<I_VV>
        VH,<I_VH>

    DPLS has no instrument file format -- the BI-200SM is VU, with no analyzer -- so
    there is nothing to be byte-faithful TO. This two-line shape is the convention the
    committed corpus already uses for its two adversarial pairs
    (``Synthetic/Stress/2 - Unphysical/dpls-*.csv``), and writing it from the generator
    is what gives the static Cabannes path an end-to-end synthetic case built from known
    truth rather than only from hand-typed pathologies: the static builder returns
    NUMBERS, and numbers are not a loadable file.

    Full precision, deliberately: the pair's whole content is a RATIO, so rounding the
    two intensities for display would move rho_v and make the file disagree with the
    ground truth it was written from. Nothing about this format is a vendor's, so there
    is no fixed-decimal convention to honour.
    """
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(f'VV,{pair.i_vv!r}\nVH,{pair.i_vh!r}\n')
    return path


def read_depolarized_pair_csv(path: str) -> Tuple[float, float]:
    """Read back a ``write_depolarized_pair_csv`` file as ``(I_VV, I_VH)``.

    The inverse lives beside the writer: a reader that drifts from its writer is the
    failure mode to design against, and there is no ``parsers/`` module to put this one
    in -- the format is the corpus's own stand-in, not an instrument's.
    Order-insensitive and case-insensitive on the row labels; anything else in the file
    is ignored, which is what makes it usable on the hand-typed stress pairs too.
    """
    vals: Dict[str, float] = {}
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if ',' in line:
                k, v = line.strip().split(',', 1)
                vals[k.strip().upper()] = float(v)
    missing = {'VV', 'VH'} - set(vals)
    if missing:
        raise ValueError(
            f"{path}: a depolarized pair file needs a VV and a VH row; missing "
            f"{sorted(missing)}.")
    return vals['VV'], vals['VH']


# ===========================================================================
# Full-dataset orchestrator (writes a whole test-data/Synthetic * folder)
# ===========================================================================
def generate_full_dataset(out_dir: str, profile_name: str, *,
                          series: Optional[Dict[str, dict]] = None) -> List[str]:
    """Regenerate a complete synthetic dataset folder (no pictures).

    Writes one SLS Zimm ``.csv`` per series sample, one SLS Guinier ``.csv`` re-emitting
    :data:`DEFAULT_SLS_SCAN_SERIES` in Brookhaven's OTHER layout (the single angular scan
    plus its blank), the DLS ``.ASC`` files (a concentration series for one sample, a
    single concentration for the rest), a bimodal DLS sample, a standalone trace, and
    ``parameters.txt``. Returns the list of file paths written. ``profile_name`` keys into
    :data:`PROFILES`. Picture rendering is added by a separate developer wrapper, which
    may import the plotting layer.
    """
    import os
    prof = PROFILES[profile_name]
    series = series or DEFAULT_SERIES
    rng = np.random.default_rng(prof['seed'])
    os.makedirs(out_dir, exist_ok=True)
    cal = CalibrationSpec()          # default calibrated test system
    written: List[str] = []

    # SLS Zimm sets (one file per sample)
    for name, p in series.items():
        sls = build_sls_set(
            mw=p['mw'], rg_nm=p['rg'], a2_mol_mL_per_g2=p['a2'],
            angles_deg=DEFAULT_SLS_ANGLES, concentrations_g_per_mL=[c * 1e-3 for c in DEFAULT_SLS_CONCS_MG],
            wavelength_nm=DEFAULT_WAVELENGTH_NM, temperature_K=DEFAULT_TEMPERATURE_K,
            n_solvent=DEFAULT_N_SOLVENT, dn_dc=DEFAULT_DN_DC, cal=cal,
            solvent_intensity_90=DEFAULT_SOLVENT_INTENSITY_90, noise_level=prof['sls_noise'],
            seed=rng.integers(1 << 31), polymer_name=name, solvent_name='water',
            label=f'{name} in Water')
        written.append(write_brookhaven_sls(
            os.path.join(out_dir, f'SLS Zimm - {name}.csv'), sls))
        # ... and one series again in the angular-scan layout, so the corpus covers
        # Brookhaven's second SLS format. Written from the SAME built `sls`, so no extra
        # rng.integers draw is taken and every other generated file stays byte-identical.
        if name == DEFAULT_SLS_SCAN_SERIES:
            written.append(write_brookhaven_sls_scan(
                os.path.join(out_dir, f'SLS Guinier - {name}.csv'), sls,
                DEFAULT_SLS_SCAN_MG * 1e-3))

    # DLS: a full concentration series for one sample, one concentration for the rest.
    # One seed per RUN, drawn from the profile stream exactly as the SLS half above
    # draws one per set: every run is an independent measurement (the literal profile
    # seed on every call gave four "different" samples bit-identical traces and a
    # concentration ladder with perfectly correlated correlogram noise).
    for name, p in series.items():
        concs = DEFAULT_DLS_CONCS_MG.get(name, [DEFAULT_DLS_SINGLE_MG])
        for c_mg in concs:
            c = c_mg * 1e-3
            pops = [SyntheticPopulation(rh_nm=rh_at_concentration(p['rh'], c), weight=1.0)]
            dls = build_multi_angle_dls(
                pops, angles_deg=DEFAULT_DLS_ANGLES, wavelength_nm=DEFAULT_WAVELENGTH_NM,
                solvent_refractive_index=DEFAULT_N_SOLVENT, temperature_K=DEFAULT_TEMPERATURE_K,
                viscosity_Pa_s=DEFAULT_VISCOSITY_PA_S, noise_level=prof['corr_noise'],
                trace_cv=prof['trace_cv'], drift=prof['drift'], spikes=prof['spikes'],
                label=f'{name} in Water', concentration_g_per_mL=c,
                seed=int(rng.integers(1 << 31)))
            written.append(write_alv_asc(
                os.path.join(out_dir, f'DLS - {name} - {c_mg:g} mg per mL.ASC'), dls,
                viscosity_cp=DEFAULT_VISCOSITY_CP))

    # bimodal DLS sample (CONTIN / NNLS / double-exp / KWW / peak picking)
    bimodal = build_multi_angle_dls(
        [SyntheticPopulation(rh_nm=20.0, weight=1.0),
         SyntheticPopulation(rh_nm=200.0, weight=2.0)],
        angles_deg=DEFAULT_DLS_ANGLES, wavelength_nm=DEFAULT_WAVELENGTH_NM,
        solvent_refractive_index=DEFAULT_N_SOLVENT, temperature_K=DEFAULT_TEMPERATURE_K,
        viscosity_Pa_s=DEFAULT_VISCOSITY_PA_S, noise_level=prof['corr_noise'],
        trace_cv=prof['trace_cv'], drift=prof['drift'], spikes=prof['spikes'],
        label='Bimodal latex mix in Water', concentration_g_per_mL=DEFAULT_DLS_SINGLE_MG * 1e-3,
        seed=int(rng.integers(1 << 31)))
    written.append(write_alv_asc(
        os.path.join(out_dir, 'DLS - Bimodal 20nm + 200nm.ASC'), bimodal,
        viscosity_cp=DEFAULT_VISCOSITY_CP))

    # standalone two-column trace (generic-trace load path)
    trace = build_count_rate_trace(
        mean_kHz=300.0, trace_cv=prof['trace_cv'], drift=prof['drift'],
        spikes=prof['spikes'], seed=int(rng.integers(1 << 31)))
    written.append(write_trace_csv(
        os.path.join(out_dir, 'Trace - count rate (kcps vs s).csv'), trace))

    written.append(write_parameters_txt(out_dir, profile_name, series))
    return written


def write_parameters_txt(out_dir: str, profile_name: str,
                         series: Optional[Dict[str, dict]] = None) -> str:
    """Write the human-readable parameters.txt (what to enter + ground truth)."""
    import os
    from physics import solvents as solvent_lib
    series = series or DEFAULT_SERIES
    messy = 'Messy' in profile_name
    cal = CalibrationSpec()
    # The text follows the Quickstart's path through the Data tab (its reader picks the
    # solvent and accepts what the program proposes), so the proposed values are the
    # library's at the set's wavelength and temperature, printed as the program shows
    # them, beside the round values the set was generated with. The committed Clean file
    # is pinned byte-for-byte to this writer (tests/test_synthetic_generator.py).
    n_water = solvent_lib.refractive_index_solvent('water', DEFAULT_WAVELENGTH_NM, DEFAULT_TEMPERATURE_C)
    eta_water_mpas = solvent_lib.viscosity_solvent('water', DEFAULT_TEMPERATURE_C) * 1e3
    n_toluene = solvent_lib.refractive_index_solvent('toluene', DEFAULT_WAVELENGTH_NM, DEFAULT_TEMPERATURE_C)
    lines = [
        f'DLS Buddy — {profile_name} test data', '=' * 50, '',
        'System: PEG in water, calibrated against a toluene standard.', '',
        'GLOBAL PARAMETERS (the Data-tab confirmation step; the Quickstart Guide walks through it):',
        '  Solvent                 : water   (pick it from the Solvent name dropdown, under',
        '                                     "Measured properties")',
        f'  Solvent refractive index: {n_water:.4f}  (proposed by the program when you pick water;',
        f'                                     the set was generated with {DEFAULT_N_SOLVENT} -- either is fine)',
        f'  Viscosity               : {eta_water_mpas:.3f} mPa.s (proposed likewise; generated with {DEFAULT_VISCOSITY_CP} cP)',
        f'  Temperature             : {DEFAULT_TEMPERATURE_C} C  ({DEFAULT_TEMPERATURE_K} K)',
        f'  Wavelength              : {DEFAULT_WAVELENGTH_NM:g} nm',
        f'  dn/dc                   : {DEFAULT_DN_DC} mL/g', '',
        'SLS CALIBRATION (enter in the SLS tab calibration panel):',
        f'  Standard                : toluene   (geometry {cal.geometry})',
        f'  Standard refractive index: {n_toluene:.5f}   (the program proposes this from its solvent',
        f'                                        library at {DEFAULT_WAVELENGTH_NM:g} nm / {DEFAULT_TEMPERATURE_C:g} C -- accept it)',
        f'  Calibrant intensity     : {cal.calibrant_intensity:g}  at 90 deg',
        '  Dark count rate         : 0',
        f'  (the program then computes k_c = {cal.k_c():.4e})', '',
        'Notes:',
        '  - DLS files are ALV .ASC: all angles + a count-rate trace are in one file.',
        '    Enter the polymer name and pick the solvent (water); temperature and',
        '    wavelength are read from the file, and picking the solvent proposes n and',
        '    the viscosity from the library. The concentration (in the file name) is',
        '    optional for a cumulant fit; enter it for the D-vs-c series.',
        '  - SLS Zimm files are Brookhaven .csv (a full Zimm set: solvent reference at',
        '    c=0 plus 5 concentrations x 12 angles). Enter polymer name + temperature.',
        "  - The SLS Guinier file is the SAME data in Brookhaven's other .csv layout:",
        '    one angular scan plus the solvent blank measured beside it. It loads as TWO',
        '    measurements. That layout records no concentration and no dn/dc, so enter',
        '    polymer name + temperature AND:',
        f'        concentration = {DEFAULT_SLS_SCAN_MG:g} mg/mL   (for the sample, not the blank)',
        f'        dn/dc         = {DEFAULT_DN_DC} mL/g',
        '  - To pair rho = Rg/Rh for a sample, give its DLS and SLS files the SAME',
        '    polymer name + solvent + temperature so they group into one sample.',
        '  - The two-column trace loads via the Utilities > Traces tab (time = s,',
        '    count rate = kcps).',
    ]
    if messy:
        lines += ['', 'This is the MESSY set: ~2.5% intensity noise, noisier',
                  'correlograms, and traces with slow drift + occasional dust spikes.',
                  'Fits should still work but with more scatter (lower R^2, some',
                  'flagged outliers). Compare to the Clean set.']
    else:
        lines += ['', 'This is the CLEAN set: low noise (good, not perfect). Fits',
                  'should recover the ground-truth values below closely.']
    lines += ['', 'GROUND TRUTH (what the analyses should recover):', '',
              f'  {"Sample":<12}{"Mw (g/mol)":>14}{"Rg (nm)":>10}{"Rh (nm)":>10}{"A2 (mol mL/g^2)":>18}']
    for name, p in series.items():
        lines.append(f'  {name:<12}{p["mw"]:>14.2e}{p["rg"]:>10.1f}{p["rh"]:>10.1f}{p["a2"]:>18.2e}')
    lines += ['',
              '  Bimodal DLS sample: two populations at Rh = 20 nm and 200 nm',
              '    (intensity weights ~1 : 2). Use CONTIN or NNLS to resolve them.', '',
              '  rho = Rg/Rh ~ 1.5 for every series sample (the THETA-solvent linear-coil',
              '    value; a good-solvent coil sits nearer 1.78 -- the series pairs this rho',
              '    with a good-solvent size exponent, and the two are not made consistent).',
              '  Rg-Mw scaling exponent nu ~ 0.585; A2-Mw exponent ~ -0.2.',
              '    The SLS sets follow the Zimm form exactly, so a Zimm fit recovers the planted',
              '    Rg at any qRg while a Berry fit is biased past qRg ~ 1. Cross-Sample Scaling',
              '    fits Zimm and switches to Berry where the Zimm Rg is flagged past the qRg',
              '    limit, so on PEG 1M and 3M it reports the smaller Berry Rg and an exponent',
              '    near 0.5, not the planted 0.585.',
              '  DLS shows a mild concentration dependence (kD ~ +30 mL/g), so the',
              '  D-vs-c extrapolation has a small positive slope (PEG 1M has 3 concs).']
    path = os.path.join(out_dir, 'parameters.txt')
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:   # LF, as committed
        fh.write('\n'.join(lines) + '\n')
    return path


# ===========================================================================
# Alpha-selection gap-case dataset (its cases: analysis.synthetic_dataset)
# ===========================================================================
def generate_alpha_selection_dataset(out_dir: str) -> List[str]:
    """Regenerate the alpha-selection gap-case set (test-data/Synthetic/Alpha-Selection/).

    Writes one loadable ALV ``.ASC`` per :data:`ALPHA_SELECTION_CASES` entry (same
    format and system as the Clean/Messy sets: PEG-in-water optics at 532 nm, all
    angles + a count-rate trace per file) plus a ``parameters.txt``. Fixed seeds, so
    re-running reproduces the same bytes. Returns the file paths written.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    for stem, pops, noise, seed, label in ALPHA_SELECTION_CASES:
        dls = build_multi_angle_dls(
            pops, angles_deg=DEFAULT_DLS_ANGLES, wavelength_nm=DEFAULT_WAVELENGTH_NM,
            solvent_refractive_index=DEFAULT_N_SOLVENT, temperature_K=DEFAULT_TEMPERATURE_K,
            viscosity_Pa_s=DEFAULT_VISCOSITY_PA_S, noise_level=noise, trace_cv=0.03,
            drift=0.0, spikes=0, label=label,
            concentration_g_per_mL=DEFAULT_DLS_SINGLE_MG * 1e-3, seed=seed)
        written.append(write_alv_asc(
            os.path.join(out_dir, stem + '.ASC'), dls, viscosity_cp=DEFAULT_VISCOSITY_CP))
    written.append(_write_alpha_selection_parameters(out_dir))
    return written


def _write_alpha_selection_parameters(out_dir: str) -> str:
    """Human-readable parameters.txt for the alpha-selection gap-case set."""
    import os
    lines = [
        'DLS Buddy — Synthetic Alpha-Selection test data', '=' * 50, '',
        'DLS correlograms that stress CONTIN\'s regularization-parameter (alpha)',
        'selection. Same optical system as the Clean/Messy sets (PEG-in-water,',
        '532 nm); load the .ASC files exactly as those (all angles are in one file;',
        'analyze the ~90 deg angle). Enter polymer name + solvent (water); temperature,',
        'viscosity, refractive index and wavelength are read from the file.', '',
        'GLOBAL PARAMETERS (Data-tab confirmation step):',
        f'  Solvent refractive index: {DEFAULT_N_SOLVENT}',
        f'  Viscosity               : {DEFAULT_VISCOSITY_CP} cP',
        f'  Temperature             : {DEFAULT_TEMPERATURE_C} C',
        f'  Wavelength              : {DEFAULT_WAVELENGTH_NM:g} nm', '',
        'CASES (ground truth — what the distribution methods should recover):', '',
        '  Broad unimodal            : single mode, median Rh = 50 nm, log-normal',
        '                              CV 0.5 (broad). Needs heavier smoothing; a good',
        '                              selector keeps ONE peak and does not under-smooth.',
        '  Close bimodal             : 40 + 90 nm (~2x separation, near the resolution',
        '                              limit) — the hard inversion case.',
        '  Wide bimodal (x3)         : 30 + 200 nm at noise 0.001 / 0.003 / 0.010.',
        '                              A good selector keeps TWO peaks at 0.001 and',
        '                              0.003; 0.010 is the margin (GCV resolves both',
        '                              at some angles and merges them at others);',
        '                              an over-smoothing one merges them into a single',
        '                              intermediate peak.', '',
        'Use CONTIN and compare the GCV (default), L-curve, and F-test selectors',
        '(DLS tab -> Distribution -> CONTIN alpha selection). GCV is the robust default;',
        'the L-curve tends to smooth more heavily on this kernel. See the',
        'Theory-and-Equations-Guide (CONTIN) for the methods.',
    ]
    path = os.path.join(out_dir, 'parameters.txt')
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:   # LF, as committed
        fh.write('\n'.join(lines) + '\n')
    return path


# ===========================================================================
# Branching-architecture dataset (writes a whole test-data/Synthetic/Architecture)
# ===========================================================================
def save_mode_resolved_dataset(tup: ModeResolvedTuple, out_dir: str, *,
                               viscosity_Pa_s: float = DEFAULT_VISCOSITY_PA_S,
                               recipe_note: str = '') -> List[str]:
    """Write a ModeResolvedTuple as a loadable folder (the Mode-Resolved set).

    Two multi-angle ALV ``.ASC`` files WITH count-rate traces (the two-mode
    sample and the c = 0 solvent blank), the calibrant trace CSV, and a
    ``parameters.txt`` recording the full ground truth. Deliberately NO
    pre-derived SLS file: static points are derived in-app from the traces, so
    the acceptance oracle stays the generator's truth rather than a check of
    our own derivation arithmetic.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    conc = tup.concentration_g_per_mL
    eta_cp = viscosity_Pa_s * 1.0e3
    written.append(write_alv_asc(
        os.path.join(out_dir, f'DLS two-mode c={conc * 1e3:g} mgml.ASC'),
        tup.dls[conc], viscosity_cp=eta_cp))
    written.append(write_alv_asc(
        os.path.join(out_dir, 'DLS solvent blank.ASC'),
        tup.dls[0.0], viscosity_cp=eta_cp))
    if tup.calibrant_trace is not None:
        written.append(write_trace_csv(
            os.path.join(out_dir, 'Calibrant trace (toluene).csv'),
            tup.calibrant_trace))

    d = tup.dls[conc]
    lines = [
        'Mode-Resolved synthetic set -- ground truth and how it was established',
        '=' * 70, '',
        'One linked truth, three faces: two-mode correlograms whose per-angle',
        'amplitude fractions ARE the static partition fractions; count-rate',
        'traces whose means ARE the static intensities (exact-mean matched);',
        'and the per-mode excess Rayleigh ratios below. Load both .ASC files,',
        'commit, run the mode map, pick the two families, record the mode set,',
        'derive the static series from the traces, and the partition must',
        'reproduce the table (the amplitude-Guinier recovers the slow Rg).', '',
        'Deliberately NO pre-derived SLS file: derive the static points in-app',
        'from the traces, so the check is against this truth, not against a',
        'stored copy of the derivation arithmetic.', '',
        'Recipe (build_mode_resolved_tuple): ' + (recipe_note or 'defaults'), '',
        'Scalar truth',
        '-' * 12,
        f'slow-mode Guinier size Rg_s     : {tup.slow_rg_nm:g} nm '
        f'(what the amplitude-Guinier fit recovers)',
        f'fast fraction at q -> 0         : {tup.a_fast_q0:g}',
        f'fast-mode Rh                    : {tup.rh_fast_nm:g} nm '
        f'(D_f = {tup.d_fast_m2_s:.6g} m^2/s)',
        f'slow-mode dynamics              : {tup.slow_kind}'
        + (f', crossing the fast branch at {tup.crossing_angle_deg:g} deg'
           if math.isfinite(tup.crossing_angle_deg) else ''),
        f'concentration                   : {conc * 1e3:g} mg/mL',
        f'wavelength / n / T / viscosity  : {d.wavelength_nm:g} nm / '
        f'{d.n_solvent:g} / {d.temperature_K:g} K / {viscosity_Pa_s:g} Pa*s',
        f'dn/dc (enter by hand)           : {tup.dn_dc:g} mL/g',
        f'calibrant intensity (toluene)   : {tup.cal.calibrant_intensity:g} cps '
        f'(k_c = {tup.cal.k_c():.6g}; geometry {tup.cal.geometry}, '
        f'{tup.cal.wavelength_nm:g} nm, {tup.cal.temperature_C:g} C)', '',
        'Per-angle truth (fractions are domain-free; dR in cm^-1, I in cps)',
        '-' * 68,
        'angle   A_fast   A_slow   dR_fast      dR_slow      dR_total     '
        'I_sample     I_solvent',
    ]
    for a in tup.angles_deg:
        lines.append(
            f'{a:5.1f}   {tup.a_fast[a]:.4f}   {tup.a_slow[a]:.4f}   '
            f'{tup.dr_fast_cm_inv[a]:.5e}  {tup.dr_slow_cm_inv[a]:.5e}  '
            f'{tup.dr_total_cm_inv[a]:.5e}  '
            f'{tup.static_intensity_cps[(conc, a)]:.6g}   '
            f'{tup.static_intensity_cps[(0.0, a)]:.6g}')
    lines += [
        '',
        'Gamma_fast per angle (1/s): '
        + ', '.join(f'{a:g}: {tup.gamma_fast_s_inv[a]:.5g}'
                    for a in tup.angles_deg),
        'Gamma_slow per angle (1/s): '
        + ', '.join(f'{a:g}: {tup.gamma_slow_s_inv[a]:.5g}'
                    for a in tup.angles_deg),
        '',
        f'exact_mean = {tup.exact_mean}; trace_cv = {tup.trace_cv:g}; '
        f'max count rate = {tup.max_count_rate_cps:.6g} cps',
    ]
    ppath = os.path.join(out_dir, 'parameters.txt')
    with open(ppath, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(lines) + '\n')
    written.append(ppath)
    return written


def generate_mode_resolved_dataset(out_dir: str, seed: int = 23) -> List[str]:
    """Regenerate the committed ``test-data/Synthetic/Mode-Resolved/`` set.

    Fixed recipe and seed, so re-running reproduces the same bytes: this function
    IS the committed set's recipe, and ``tests/test_synthetic_generator.py`` pins
    that the folder regenerates byte-identical from it. The in-app save path
    (``controller.save_synth_mode_resolved``) reaches ``build_mode_resolved_tuple``
    + ``save_mode_resolved_dataset`` directly with the user's own recipe, and no
    developer script wraps this one. The angle grid spans the 82-degree crossing
    (the association-breaking configuration stays the test-drivable one).
    """
    tup = build_mode_resolved_tuple(seed=seed, polymer_name='Two-mode demo',
                                    solvent_name='water',
                                    label='Mode-resolved demo')
    return save_mode_resolved_dataset(
        tup, out_dir,
        recipe_note=f'defaults, seed={seed} (slow_rg_nm=120, a_fast_q0=0.3, '
                    f'rh_fast_nm=5, crossing at 82 deg)')


def generate_architecture_dataset(out_dir: str, seed: int = 17) -> List[str]:
    """Regenerate the branching-architecture test set (SLS Zimm + multi-angle DLS per
    sample of :func:`build_architecture_series`, plus a ``parameters.txt``). Fixed seeds,
    so re-running reproduces the same bytes. Shared by the in-app Synthetic Generator and
    the developer test-data generator script. Returns the file
    paths written.
    """
    import os
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    cal = CalibrationSpec()
    series = build_architecture_series()
    written: List[str] = []
    for name, p in series.items():
        sls = build_sls_set(
            mw=p['mw'], rg_nm=p['rg'], a2_mol_mL_per_g2=p['a2'],
            angles_deg=DEFAULT_SLS_ANGLES,
            concentrations_g_per_mL=[c * 1e-3 for c in DEFAULT_SLS_CONCS_MG],
            wavelength_nm=DEFAULT_WAVELENGTH_NM, temperature_K=DEFAULT_TEMPERATURE_K,
            n_solvent=DEFAULT_N_SOLVENT, dn_dc=DEFAULT_DN_DC, cal=cal,
            solvent_intensity_90=DEFAULT_SOLVENT_INTENSITY_90, noise_level=0.005,
            seed=int(rng.integers(1 << 31)), polymer_name=name, solvent_name='water',
            label=f'{name} in water')
        written.append(write_brookhaven_sls(
            os.path.join(out_dir, f'SLS Zimm - {name}.csv'), sls))
        written.append(_write_water_dls(out_dir, name, p['rh'], rng))

    written.append(_write_architecture_parameters(out_dir, series, cal))
    return written


def _write_water_dls(out_dir: str, name: str, rh_nm: float,
                     rng: np.random.Generator) -> str:
    """Write a series sample's paired multi-angle DLS ``.ASC`` (one population of
    ``rh_nm`` in the default water system, at the single DLS concentration) and return
    its path. The run's seed is drawn from the series writer's own stream ``rng``, right
    after that sample's SLS set took its draw, so each run is an independent measurement
    and the stream's order fixes every generated file's bytes.
    """
    import os
    pops = [SyntheticPopulation(rh_nm=rh_nm, weight=1.0)]
    dls = build_multi_angle_dls(
        pops, angles_deg=DEFAULT_DLS_ANGLES, wavelength_nm=DEFAULT_WAVELENGTH_NM,
        solvent_refractive_index=DEFAULT_N_SOLVENT, temperature_K=DEFAULT_TEMPERATURE_K,
        viscosity_Pa_s=DEFAULT_VISCOSITY_PA_S, noise_level=0.005, trace_cv=0.03,
        drift=0.0, spikes=0, label=f'{name} in water',
        concentration_g_per_mL=DEFAULT_DLS_SINGLE_MG * 1e-3,
        seed=int(rng.integers(1 << 31)))       # one seed per run, like the SLS half
    return write_alv_asc(
        os.path.join(out_dir, f'DLS - {name}.ASC'), dls,
        viscosity_cp=DEFAULT_VISCOSITY_CP)


def _write_architecture_parameters(out_dir: str, series: Dict[str, dict],
                                   cal: CalibrationSpec) -> str:
    """Human-readable parameters.txt (ground truth + what to enter) for the set."""
    import os
    lines = [
        'DLS Buddy — Synthetic Architecture (branching contraction factor g) test data',
        '=' * 76, '',
        'A homologous LINEAR Rg-Mw series (Rg = 0.05 * Mw^0.5), two REGULAR STARS, and two',
        'RANDOMLY-BRANCHED FRACTIONS (tri- and tetra-functional), built to exercise the',
        'Cross-Sample Architecture tab: tag the linear samples Linear and a branched sample',
        'Branched (Chain type, Data tab), and g = (Rg_branched / Rg_linear)^2 is recovered.',
        'Its reading depends on the selected branching model:',
        '  * regular star            -> f_eff  (arm number, from g = (3f-2)/f^2)',
        '  * random tri / tetra      -> n      (branch units/molecule; '
        'Theory-and-Equations-Guide §5.3.1)', '',
        'GLOBAL PARAMETERS (enter at the Data-tab confirmation step):',
        '  Solvent                 : water',
        f'  Solvent refractive index: {DEFAULT_N_SOLVENT}',
        f'  Viscosity               : {DEFAULT_VISCOSITY_CP} cP',
        f'  Temperature             : {DEFAULT_TEMPERATURE_C} C ({DEFAULT_TEMPERATURE_K} K)',
        f'  Wavelength              : {DEFAULT_WAVELENGTH_NM:g} nm',
        f'  dn/dc                   : {DEFAULT_DN_DC} mL/g', '',
        'SLS CALIBRATION (SLS tab calibration panel):',
        f'  Standard toluene ({cal.geometry}); calibrant intensity {cal.calibrant_intensity:g} at 90 deg',
        f'  (the program computes k_c = {cal.k_c():.4e})', '',
        'GROUND TRUTH (Role -> tag on the Architecture tab):',
        f'  {"sample":<18}{"role":<10}{"Mw":<12}{"Rg (nm)":<10}{"Rh (nm)":<10}'
        f'{"model":<14}{"f":<4}{"n":<4}{"g":<8}',
    ]
    _model_disp = {None: '', 'regular_star': 'regular star',
                   'random_tri': 'random tri', 'random_tetra': 'random tetra'}
    for name, p in series.items():
        f = '' if p['f'] is None else str(p['f'])
        n = '' if p.get('n') is None else str(p['n'])
        g = '' if p['g'] is None else f'{p["g"]:.4f}'
        lines.append(f'  {name:<18}{p["role"]:<10}{p["mw"]:<12.3g}'
                     f'{p["rg"]:<10.3f}{p["rh"]:<10.3f}'
                     f'{_model_disp[p.get("model")]:<14}{f:<4}{n:<4}{g:<8}')
    lines += ['',
              'Expected on the Architecture tab (linear series tagged Linear, one Branched):',
              '  PS star 3-arm    -> g ≈ 0.778;  regular star  -> f_eff ≈ 3',
              '  PS star 4-arm    -> g ≈ 0.625;  regular star  -> f_eff ≈ 4',
              '  PS random tri    -> g ≈ 0.672;  random tri    -> n ≈ 6 branch units/molecule',
              '  PS random tetra  -> g ≈ 0.578;  random tetra  -> n ≈ 4 branch units/molecule',
              '  (all four branched Mw sit inside the linear reference span 1e5-9e5)', '',
              'MODEL-DEPENDENCE (the headline caveat): the SAME g inverts to a DIFFERENT n',
              'under a different model. E.g. PS random tri (g ≈ 0.672) reads as n ≈ 6 under the',
              'tri model but n ≈ 2.4 under the tetra model, and f_eff ≈ 3.6 as a regular star —',
              'so the readout must always name the assumed model.', '']
    path = os.path.join(out_dir, 'parameters.txt')
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:   # LF, as committed
        fh.write('\n'.join(lines))
    return path


# ===========================================================================
# Form-factor SETS (its series: analysis.synthetic_dataset)
# ===========================================================================
def generate_form_factor_dataset(out_dir: str, seed: int = 23) -> List[str]:
    """Regenerate the form-factor SET test data (an SLS form-factor CSV + a paired
    multi-angle DLS ASC per sample of :func:`build_form_factor_series`, plus a
    ``parameters.txt``). Fixed seeds -> reproducible bytes. Shared by the in-app Synthetic
    Generator and the developer test-data generator script.
    Returns the file paths written.
    """
    import os
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    series = build_form_factor_series()
    written: List[str] = []
    for name, p in series.items():
        cal = CalibrationSpec(include_in_files=bool(p['calibrated']))
        concs_mg = ([DEFAULT_DLS_SINGLE_MG] if p['single_conc']
                    else [c for c in DEFAULT_SLS_CONCS_MG if c > 0])
        sls = build_sls_set(
            mw=p['mw'], rg_nm=p['rg'], a2_mol_mL_per_g2=p['a2'],
            angles_deg=DEFAULT_SLS_ANGLES,
            concentrations_g_per_mL=[c * 1e-3 for c in concs_mg],
            wavelength_nm=DEFAULT_WAVELENGTH_NM, temperature_K=DEFAULT_TEMPERATURE_K,
            n_solvent=DEFAULT_N_SOLVENT, dn_dc=DEFAULT_DN_DC, cal=cal,
            solvent_intensity_90=DEFAULT_SOLVENT_INTENSITY_90, noise_level=0.005,
            seed=int(rng.integers(1 << 31)), polymer_name=name, solvent_name='water',
            label=f'{name} in water',
            kind=('single_concentration' if p['single_conc'] else 'zimm'),
            form_factor_model=p['model'], **p['params'])
        written.append(write_brookhaven_sls(
            os.path.join(out_dir, f'SLS form factor - {name}.csv'), sls))
        written.append(_write_water_dls(out_dir, name, p['rh'], rng))

    written.append(_write_form_factor_parameters(out_dir, series, CalibrationSpec()))
    return written


def _write_form_factor_parameters(out_dir: str, series: Dict[str, dict],
                                  cal: CalibrationSpec) -> str:
    """Human-readable parameters.txt (ground truth + what each sample exercises)."""
    import os
    _degen_disp = {False: 'no', 'divergent': 'yes (Rg-divergent)',
                   'consensus': 'yes (Rg-agree)'}
    lines = [
        'DLS Buddy — Synthetic Form-factor SETS (SLS P(q) + paired DLS) test data',
        '=' * 76, '',
        'Each sample is an SLS form-factor set PLUS a paired multi-angle DLS whose Rh is',
        'set from a textbook rho = Rg/Rh for that shape. Load a pair, fit the Form factor',
        'P(q) for Rg (SLS tab) and a cumulant / Gamma-q^2 for Rh (DLS tab), pick the',
        'form-factor Rg on the Cross-Sample tab, and rho = Rg/Rh reproduces the shape',
        'verdict below. The panel exercises the cross-sample tagging / validation axes:',
        '  * rho shape verdict          -> sphere / soft-sphere / coil / rod',
        '  * model-dependent flag        -> a form-factor Rg is flagged (neutral i) everywhere',
        '  * degenerate selection        -> ellipsoid (Rg-divergent) + core-shell (Rg-agree)',
        '  * model-dependent scaling     -> the FF sphere ladder fits nu ~ 1/3 (tag Linear)',
        '  * apparent vs thermodynamic   -> FF sphere apparent is single-concentration',
        '  * uncalibrated                -> FF sphere uncal (Mw arbitrary scale; Rg/rho survive)',
        '', 'GLOBAL PARAMETERS (enter at the Data-tab confirmation step):',
        '  Solvent                 : water',
        f'  Solvent refractive index: {DEFAULT_N_SOLVENT}',
        f'  Viscosity               : {DEFAULT_VISCOSITY_CP} cP',
        f'  Temperature             : {DEFAULT_TEMPERATURE_C} C ({DEFAULT_TEMPERATURE_K} K)',
        f'  Wavelength              : {DEFAULT_WAVELENGTH_NM:g} nm',
        f'  dn/dc                   : {DEFAULT_DN_DC} mL/g', '',
        'SLS CALIBRATION (SLS tab calibration panel; the calibrated samples carry it):',
        f'  Standard toluene ({cal.geometry}); calibrant intensity {cal.calibrant_intensity:g} at 90 deg',
        f'  (the program computes k_c = {cal.k_c():.4e})', '',
        'GROUND TRUTH:',
        f'  {"sample":<20}{"model":<12}{"Mw":<11}{"Rg (nm)":<9}{"Rh (nm)":<9}'
        f'{"rho":<6}{"verdict":<26}{"degenerate":<20}{"cal":<5}{"shape params"}',
    ]
    for name, p in series.items():
        # Shape params are a VARIABLE-length string, so they get their own TRAILING column --
        # never concatenated into the fixed-width `model` field (that would overflow the pad and
        # run the next column into it, exactly on the degenerate ellipsoid/core-shell rows).
        params = ''
        if p['model'] == 'ellipsoid':
            params = f"eps={p['params']['ellipsoid_axial_ratio']:g}"
        elif p['model'] == 'mie':
            params = f"m={p['params']['m']:g}"
        elif p['model'] == 'core_shell':
            params = (f"Rc={p['params']['r_core_nm']:g} Ro={p['params']['r_outer_nm']:g}"
                      f" m1={p['params']['m1']:g} m2={p['params']['m2']:g}")
        lines.append(
            f'  {name:<20}{p["model"]:<12}{p["mw"]:<11.3g}{p["rg"]:<9.3g}'
            f'{p["rh"]:<9.3g}{p["rho"]:<6.2f}{p["verdict"]:<26}'
            f'{_degen_disp[p["degenerate"]]:<20}{"yes" if p["calibrated"] else "NO":<5}{params}')
    lines += [
        '', 'WHAT TO EXPECT ON THE CROSS-SAMPLE TAB:',
        '  * FF coil        -> rho ~ 1.5  -> "random (Gaussian) coil"',
        '  * FF ellipsoid   -> rho ~ 2.5  -> "rod / extended"; the ellipsoid fit is DEGENERATE',
        '      and Rg-divergent, so each admissible solution is an explicit pick (no consensus).',
        '  * FF core-shell  -> rho ~ 1.0  -> "dendrimer / soft sphere"; DEGENERATE but the',
        '      solutions AGREE on Rg, so a "size determined, shape ambiguous" consensus is offered.',
        '  * FF Mie latex   -> rho ~ 0.78 "hard sphere" (a solid latex sphere); qRg ~ 7.5 sits',
        '      past the first Mie minimum, so Rg recovers while Mw/A2 carry the deep-minimum',
        '      ill-conditioning flag -- expected physics, not a defect.',
        '  * FF sphere 0.5M/1M/2M -> rho ~ 0.78 "hard sphere"; tagged Linear, the scaling plot',
        '      fits nu ~ 1/3 (Rg-Mw and Rh-Mw), carrying the MODEL-DEPENDENT flag.',
        '  * FF sphere apparent -> a single-concentration form-factor Rg -> flagged APPARENT.',
        '  * FF sphere uncal    -> load in its OWN session (or clear the SLS calibration): Mw and',
        '      A2 go arbitrary-scale (uncalibrated warning), but Rg and rho survive.',
        '', 'NOTE: SLS calibration is session-global, so do not load "FF sphere uncal" in the',
        'same session as the calibrated samples if you want to see its uncalibrated behaviour.', '']
    path = os.path.join(out_dir, 'parameters.txt')
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:   # LF, as committed
        fh.write('\n'.join(lines))
    return path
