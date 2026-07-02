"""Regression tests for the instrument parser layer (parsers/).

Every committed instrument format is parsed and round-tripped into the common
internal data model (DLSMeasurement / SLSMeasurement / IntensityTrace), asserting
the key fields each parser is contracted to extract and the unit conversions it
performs at load time (the two-layer parser design: a parser's only job is native
file -> preview -> build() into the common model).

The real-file cases are marked ``realdata`` (they read a file from ``test-data/``).
The generic plain-text parsers have no committed exemplar of their exact strict
two-column contract, so those and the malformed-input cases build tiny files under
``tmp_path`` instead and are NOT ``realdata``.
"""
from __future__ import annotations

import numpy as np
import pytest

from fixtures.data_paths import (
    ALV_LATEX_DIR, ALV_PS_TOLUENE_DIR, ALV_NAPSS_DIR,
    BROOKHAVEN_DIR, MALVERN_DIR, SYNTH_CLEAN_DIR, require,
)

from parsers.base_parser import ParseError
from parsers.alv_asc import ALVCorrelatorParser, ALVTraceParser
from parsers.brookhaven_dls import BrookhavenDLSParser, BrookhavenTraceParser
from parsers.brookhaven_sls import BrookhavenSLSParser
from parsers.zetasizer_clipboard import ZetasizerClipboardParser
from parsers.zetasizer_export import ZetasizerExportParser
from parsers.generic_dls import GenericDLSParser
from parsers.generic_sls import GenericSLSParser
from parsers.generic_trace import GenericTraceParser


# ---------------------------------------------------------------- ALV .ASC ---

@pytest.mark.realdata
def test_alv_correlator_roundtrip():
    path = require(ALV_LATEX_DIR / "Noisy050Latex0004_0001.ASC")
    previews = ALVCorrelatorParser().parse(str(path))

    # One preview per ACTIVE angle; the file declares 8 (44..147), the other four
    # detector channels are angle 0 and are dropped.
    assert len(previews) == 8
    angles = sorted(round(p.angle_deg) for p in previews)
    assert angles == [44, 50, 64, 81, 99, 117, 134, 147]

    for p in previews:
        # header optics/thermal, shared across all angles of the run
        assert p.temperature_K == pytest.approx(298.16, abs=1e-2)
        assert p.wavelength_nm == pytest.approx(685.0, abs=1e-6)
        assert p.solvent_refractive_index == pytest.approx(1.332, abs=1e-3)
        # viscosity 0.89 cp -> Pa.s
        assert p.viscosity_Pa_s == pytest.approx(0.89e-3, rel=1e-6)
        # ~200-point correlogram; lag axis aligned with the g2-1 column
        assert p.delay_times_s.size == p.correlogram.size
        assert 150 <= p.delay_times_s.size <= 300
        # THE GOTCHA: lag is stored in ms and converted ms -> s. The first lag
        # 2.5e-5 ms = 25 ns becomes 2.5e-8 s; the whole axis lands in seconds, not
        # thousands-of-ms. Read as seconds it would be 1000x too large.
        assert p.delay_times_s[0] == pytest.approx(2.5e-8, rel=1e-3)
        assert p.delay_times_s[-1] < 100.0

    # round-trip into the common data model (fill the identity the file lacks)
    p0 = previews[0]
    p0.polymer_name = "Latex"
    p0.solvent_name = "water"
    p0.concentration_g_per_mL = 1.0e-3   # preview requires a value (may be 0.0)
    meas = p0.build()
    assert meas.delay_times_s.size == p0.delay_times_s.size
    assert meas.angle_deg == pytest.approx(p0.angle_deg)


@pytest.mark.realdata
def test_alv_trace_roundtrip():
    path = require(ALV_LATEX_DIR / "Noisy050Latex0004_0001.ASC")
    previews = ALVTraceParser().parse(str(path))

    assert len(previews) == 8            # one count-rate trace per active angle
    for p in previews:
        assert p.times_s.size == p.count_rates_cps.size
        assert p.times_s.size > 1
        # count rate is stored kHz in the file and converted kHz -> cps (x1000),
        # so the values are in the hundreds-of-thousands, not hundreds.
        assert np.nanmax(p.count_rates_cps) > 1.0e4
        assert np.all(p.times_s >= 0.0)

    trace = previews[0].build()
    assert trace.count_rates_cps.size == previews[0].count_rates_cps.size


@pytest.mark.realdata
def test_alv_rejects_non_alv_file(tmp_path):
    bad = tmp_path / "not_alv.ASC"
    bad.write_text("this is not an ALV file\n1,2,3\n", encoding="latin-1")
    with pytest.raises(ParseError):
        ALVCorrelatorParser().parse(str(bad))


@pytest.mark.realdata
def test_alv_single_angle_7004_roundtrip():
    # ALV-7004/USB single-angle format: one file = one angle, bare `Angle [deg]`
    # header key (not the multi-detector Angle(1..12) form). The parser must yield
    # exactly ONE preview, using the single CH0 data column.
    path = require(ALV_PS_TOLUENE_DIR / "DLS - PS 290k Toluene - 1.5 mg per mL - 090 deg (avg).ASC")
    previews = ALVCorrelatorParser().parse(str(path))
    assert len(previews) == 1
    p = previews[0]
    assert p.angle_deg == pytest.approx(90.0, abs=1e-6)
    assert p.wavelength_nm == pytest.approx(660.0, abs=1e-6)
    assert p.solvent_refractive_index == pytest.approx(1.496, abs=1e-3)   # toluene
    assert p.temperature_K == pytest.approx(293.2, abs=0.2)
    # same ms -> s lag conversion as the multi-angle path
    assert p.delay_times_s[0] == pytest.approx(2.5e-8, rel=1e-3)
    assert p.delay_times_s.size == p.correlogram.size

    # a NaPSS single-angle file parses too (water optics)
    npath = require(ALV_NAPSS_DIR / "DLS - NaPSS 40 mg per mL Water - 090 deg.ASC")
    npv = ALVCorrelatorParser().parse(str(npath))
    assert len(npv) == 1
    assert npv[0].solvent_refractive_index == pytest.approx(1.332, abs=1e-3)
    assert npv[0].angle_deg == pytest.approx(90.0, abs=1e-6)


# --------------------------------------------------------- Brookhaven DLS ---

@pytest.mark.realdata
def test_brookhaven_dls_roundtrip():
    path = require(BROOKHAVEN_DIR / "Correlation Function - PVP (40k) in Water.csv")
    previews = BrookhavenDLSParser().parse(str(path))

    assert len(previews) == 1
    p = previews[0]
    assert p.instrument_name == "Brookhaven Particle Explorer"
    assert p.sample_label and "PnVP" in p.sample_label
    assert p.delay_times_s.size == p.correlogram.size
    assert p.delay_times_s.size > 50
    # delay stored in microseconds, converted us -> s (first real lag 0.25 us).
    assert p.delay_times_s[0] == pytest.approx(0.25e-6, rel=1e-6)
    # the file carries NO physical parameters -> all left None for the user.
    assert p.polymer_name is None
    assert p.temperature_K is None
    assert p.angle_deg is None

    # round-trip into the common data model
    p.polymer_name = "PVP"
    p.solvent_name = "water"
    p.concentration_g_per_mL = 5.0e-4
    p.temperature_K = 298.15
    p.angle_deg = 90.0
    p.wavelength_nm = 532.0
    p.solvent_refractive_index = 1.33
    meas = p.build()
    assert meas.correlogram.size == p.correlogram.size


@pytest.mark.realdata
def test_brookhaven_trace_roundtrip():
    path = require(BROOKHAVEN_DIR / "Count Rate History - PVP (40k) in Water.csv")
    previews = BrookhavenTraceParser().parse(str(path))

    assert len(previews) == 1
    p = previews[0]
    assert p.times_s.size == p.count_rates_cps.size
    assert p.times_s.size > 10
    # kcps -> cps: the file's ~19 kcps first sample becomes ~19500 cps.
    assert np.nanmax(p.count_rates_cps) > 1.0e4
    # elapsed seconds parsed from HH:MM:SS timestamps (first sample ~1 s, not 0).
    assert p.times_s[0] == pytest.approx(1.157, abs=1e-2)
    trace = p.build()
    assert trace.times_s.size == p.times_s.size


@pytest.mark.realdata
def test_brookhaven_dls_rejects_plain_two_column(tmp_path):
    # A generic two-column CSV lacks the Brookhaven 't(us),C(t),...' header row;
    # the parser must reject it (Session 36 strictness) rather than eat two rows.
    bad = tmp_path / "plain.csv"
    bad.write_text("1.0,0.5\n2.0,0.4\n3.0,0.3\n", encoding="latin-1")
    with pytest.raises(ParseError):
        BrookhavenDLSParser().parse(str(bad))


# --------------------------------------------------------- Brookhaven SLS ---

@pytest.mark.realdata
def test_brookhaven_sls_roundtrip():
    path = require(BROOKHAVEN_DIR / "Zimm Plot - PS (900k) in Toluene Intensities.csv")
    previews = BrookhavenSLSParser().parse(str(path))

    # 7 concentrations (incl. the c = 0 solvent), each an angular series of 13.
    assert len(previews) == 7
    for p in previews:
        assert p.angles_deg.size == 13
        assert p.intensities.size == 13
        assert p.wavelength_nm == pytest.approx(532.0)
        assert p.solvent_refractive_index == pytest.approx(1.502, rel=1e-6)
        assert p.dn_dc_mL_per_g == pytest.approx(0.11, rel=1e-6)
        assert p.solvent_name == "Toluene"
        # informational calibration reference values from the header
        assert p.calibration_constant == pytest.approx(3.224e-10, rel=1e-4)
        assert p.standard_rayleigh_ratio_file == pytest.approx(2.803e-05, rel=1e-4)
        # the file lacks polymer + temperature (user supplies them)
        assert p.polymer_name is None
        assert p.temperature_K is None

    # exactly one concentration is the c = 0 solvent reference
    concs = sorted(p.concentration_g_per_mL for p in previews)
    assert concs[0] == pytest.approx(0.0)
    assert sum(1 for c in concs if c == 0.0) == 1

    # round-trip a non-zero-concentration preview into the common model
    p = next(p for p in previews if p.concentration_g_per_mL > 0)
    p.polymer_name = "PS"
    p.temperature_K = 298.15
    meas = p.build()
    assert meas.angles_deg.size == 13


# --------------------------------------------------------- Zetasizer ---

@pytest.mark.realdata
def test_zetasizer_clipboard_multi():
    path = require(MALVERN_DIR / "Zetasizer Multi Clipboard.txt")
    previews = ZetasizerClipboardParser().parse(str(path))

    assert len(previews) > 1                  # many record columns
    lag = previews[0].delay_times_s
    for p in previews:
        assert p.instrument_name == "Malvern Zetasizer"
        assert p.correlogram.size == lag.size
        np.testing.assert_allclose(p.delay_times_s, lag)
    # lag stored in us, converted us -> s (first lag 0.5 us -> 5e-7 s).
    assert lag[0] == pytest.approx(0.5e-6, rel=1e-6)


@pytest.mark.realdata
def test_zetasizer_clipboard_single():
    path = require(MALVERN_DIR / "Correlation Function - PEG (8k) in Water.txt")
    previews = ZetasizerClipboardParser().parse(str(path))
    assert len(previews) == 1
    p = previews[0]
    assert p.delay_times_s.size == p.correlogram.size
    assert p.delay_times_s[0] == pytest.approx(0.5e-6, rel=1e-6)


@pytest.mark.realdata
def test_zetasizer_export_roundtrip():
    path = require(MALVERN_DIR / "Zetasizer Multi Export.txt")
    previews = ZetasizerExportParser().parse(str(path))

    assert len(previews) >= 1                 # one measurement per data row
    for p in previews:
        assert p.instrument_name == "Malvern Zetasizer"
        assert p.delay_times_s.size == p.correlogram.size
        assert p.delay_times_s.size > 1
        # export carries temperature (deg C -> K) and viscosity (cP -> Pa.s)
        assert p.temperature_K is not None and p.temperature_K > 273.0
        assert p.viscosity_Pa_s is not None and p.viscosity_Pa_s > 0.0
        # lag converted us -> s: sub-second lags
        assert np.nanmax(p.delay_times_s) < 100.0


# --------------------------------------------------------- generic parsers ---

@pytest.mark.realdata
def test_generic_trace_roundtrip():
    # A committed plain two-column (time_s, count_rate_kcps) trace.
    path = require(SYNTH_CLEAN_DIR / "Trace - count rate (kcps vs s).csv")
    previews = GenericTraceParser().parse(str(path))
    assert len(previews) == 1
    p = previews[0]
    assert not p.is_ready()                   # units not yet confirmed
    p.time_unit = "s"
    p.count_rate_unit = "kcps"
    trace = p.build()
    assert trace.times_s.size == trace.count_rates_cps.size
    assert trace.times_s.size > 100
    # kcps -> cps: a ~284 kcps first sample becomes ~284000 cps.
    assert np.nanmax(trace.count_rates_cps) > 1.0e5


def test_generic_dls_roundtrip(tmp_path):
    # Strict two-column, no-header table (delay_us, g2m1).
    f = tmp_path / "correlogram.csv"
    tau_us = np.geomspace(0.25, 1.0e5, 64)
    g = 0.8 * np.exp(-tau_us / 500.0)
    f.write_text("\n".join(f"{t},{v}" for t, v in zip(tau_us, g, strict=True)), encoding="utf-8")

    previews = GenericDLSParser().parse(str(f))
    assert len(previews) == 1
    p = previews[0]
    p.delay_time_unit = "us"
    p.data_form = "g2m1"
    p.polymer_name = "test"
    p.solvent_name = "water"
    p.concentration_g_per_mL = 1.0e-3
    p.temperature_K = 298.15
    p.angle_deg = 90.0
    p.wavelength_nm = 633.0
    p.solvent_refractive_index = 1.33
    meas = p.build()
    assert meas.delay_times_s.size == tau_us.size
    # us -> s conversion applied
    assert meas.delay_times_s[0] == pytest.approx(0.25e-6, rel=1e-9)


def test_generic_dls_rejects_headered_file(tmp_path):
    # A file dominated by non-numeric (header/comment) rows violates the strict
    # two-column contract and must raise.
    f = tmp_path / "headered.csv"
    f.write_text("time,signal\nlabel,here\nfoo,bar\nbaz,qux\n", encoding="utf-8")
    with pytest.raises(ParseError):
        GenericDLSParser().parse(str(f))


def test_generic_sls_roundtrip(tmp_path):
    # Strict two-column, no-header table (angle_deg, intensity).
    f = tmp_path / "sls.csv"
    angles = [30, 45, 60, 75, 90, 105, 120, 135]
    inten = [1000.0 / np.sin(np.radians(a)) for a in angles]
    f.write_text("\n".join(f"{a},{i}" for a, i in zip(angles, inten, strict=True)), encoding="utf-8")

    previews = GenericSLSParser().parse(str(f))
    assert len(previews) == 1
    p = previews[0]
    assert p.angles_deg.size == len(angles)
    p.polymer_name = "PS"
    p.solvent_name = "toluene"
    p.concentration_g_per_mL = 5.0e-4
    p.temperature_K = 298.15
    p.wavelength_nm = 532.0
    p.solvent_refractive_index = 1.502
    p.dn_dc_mL_per_g = 0.11
    meas = p.build()
    assert meas.angles_deg.size == len(angles)


def test_generic_sls_rejects_bad_angles(tmp_path):
    # Column 1 out of the (0, 180) angle range -> ParseError (columns swapped?).
    f = tmp_path / "badangles.csv"
    f.write_text("500,1.0\n600,2.0\n700,3.0\n", encoding="utf-8")
    with pytest.raises(ParseError):
        GenericSLSParser().parse(str(f))
