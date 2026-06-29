"""
analysis/synthetic_dataset.py
=============================

Synthetic light-scattering **dataset** generation: the reusable engine behind
both the developer test-data tool (``internal/generate_synthetic_test_data.py``) and
the in-app *Utilities ▸ Synthetic data* generator.

What lives here
---------------
* **Forward models** (pure): the same optical-constant / Rayleigh / Zimm physics
  the analysis code uses, run *forwards* (parameters → intensities/correlograms),
  so that loading and analysing the output recovers the ground truth.
* **Builders** (pure): assemble the artifacts the GUI offers — a single
  correlogram (delegated to :func:`analysis.utilities.generate_synthetic_correlogram`),
  a multi-angle DLS set, an SLS intensity set (a full Zimm grid or a single-angle
  / single-concentration slice), and a count-rate trace — each returned as a plain
  data object, never written or plotted here.
* **Writers** (file I/O): turn a built artifact into a *loadable instrument file*
  (ALV ``.ASC`` for DLS, Brookhaven ``.csv`` for SLS, a two-column CSV for a
  standalone trace). These are the file-I/O exception in the analysis package, in
  the same spirit as ``utilities.export_synthetic_correlogram_csv``; they are kept
  here so the controller (which may not import plotting) can save without the GUI.
* **Full-dataset orchestrator** :func:`generate_full_dataset`: regenerate a whole
  ``test-data/Synthetic *`` folder (the homologous Mw series + bimodal sample +
  trace + ``parameters.txt``). Picture rendering is intentionally *not* here (it
  needs the plotting layer); the ``internal/`` wrapper adds it.

Deliberately **no plotting and no analysis-engine imports** — preview figures are
drawn by the plotting layer from the GUI, and the ground-truth round-trip is
validated by the test harness, not by this module analysing its own output.

The default constants describe the canonical test system (PEG in water, calibrated
against a toluene VU standard at 532 nm). They are defaults only: every builder
takes the physical parameters explicitly, so the in-app generator drives them from
user input and nothing system-specific is baked into a generated artifact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from physics import constants as phys
from analysis.utilities import (
    SyntheticPopulation, SyntheticCorrelogramResult,
    generate_synthetic_correlogram,
)


# ===========================================================================
# Default test-data system (PEG in water vs a toluene VU standard at 532 nm)
# ===========================================================================
DEFAULT_WAVELENGTH_NM = 532.0
DEFAULT_TEMPERATURE_C = 25.0
DEFAULT_TEMPERATURE_K = 298.15
DEFAULT_N_SOLVENT = 1.33
DEFAULT_VISCOSITY_CP = 0.89
DEFAULT_VISCOSITY_PA_S = 0.89e-3
DEFAULT_N_STANDARD = 1.496                 # toluene at 532 nm
DEFAULT_GEOMETRY = 'VU'                     # BI-200SM: no analyser
DEFAULT_DN_DC = 0.135
DEFAULT_CALIBRANT_INTENSITY = 1.0e5        # toluene count rate at 90° (a.u.)
DEFAULT_SOLVENT_INTENSITY_90 = 6000.0      # water baseline at 90° (a.u.)
DEFAULT_KD_ML_G = 30.0                      # diffusion interaction kD (good solvent)

# Angle / concentration grids for the full test set.
DEFAULT_SLS_ANGLES = [35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145]
DEFAULT_DLS_ANGLES = [35, 50, 65, 80, 95, 110, 125, 140]
DEFAULT_SLS_CONCS_MG = [0.0, 0.2, 0.4, 0.6, 1.0, 1.4]    # mg/mL; the 0.0 is the ref

# Homologous Mw series for the full test set (Mw g/mol, Rg nm, Rh nm, A2 mol mL/g^2).
DEFAULT_SERIES = {
    'PEG 100k': dict(mw=1.0e5, rg=14.0, rh=9.5, a2=1.5e-4),
    'PEG 300k': dict(mw=3.0e5, rg=27.0, rh=18.0, a2=1.2e-4),
    'PEG 1M':   dict(mw=1.0e6, rg=55.0, rh=37.0, a2=9.5e-5),
    'PEG 3M':   dict(mw=3.0e6, rg=105.0, rh=70.0, a2=7.0e-5),
}
DEFAULT_DLS_CONCS_MG = {'PEG 1M': [0.2, 0.6, 1.4]}      # one sample gets a c-series
DEFAULT_DLS_SINGLE_MG = 0.6                              # the rest: one concentration

# Noise/realism profiles for the full test set.
PROFILES = {
    'Synthetic Clean': dict(corr_noise=0.005, sls_noise=0.005, trace_cv=0.03,
                            drift=0.0, spikes=0, seed=11),
    'Synthetic Messy': dict(corr_noise=0.030, sls_noise=0.025, trace_cv=0.06,
                            drift=0.12, spikes=4, seed=29),
}


# ===========================================================================
# Calibration specification
# ===========================================================================
@dataclass
class CalibrationSpec:
    """The SLS calibration used to scale generated excess intensities.

    ``k_c`` (= R_standard / I_calibrant) maps an intensity to a Rayleigh ratio,
    exactly as the analysis calibration does in reverse. ``include_in_files``
    controls only whether the *written* Brookhaven file carries calibration
    metadata: with it on (the default) the file documents the standard + constant,
    so a user who enters the matching calibrant gets an absolute Mw; with it off
    the file omits that metadata, so an analysis run is flagged uncalibrated
    (arbitrary-scale Mw). Either way the intensities themselves are generated with
    the same ``k_c`` — "uncalibrated" is a property of how the data is *used*, not
    of the numbers.
    """
    wavelength_nm: float = DEFAULT_WAVELENGTH_NM
    temperature_C: float = DEFAULT_TEMPERATURE_C
    geometry: str = DEFAULT_GEOMETRY
    n_standard: float = DEFAULT_N_STANDARD
    calibrant_intensity: float = DEFAULT_CALIBRANT_INTENSITY
    include_in_files: bool = True

    def rayleigh(self) -> float:
        """Rayleigh ratio of the toluene standard (cm^-1) for this geometry."""
        return phys.rayleigh_ratio_toluene(
            self.wavelength_nm, self.temperature_C, geometry=self.geometry)

    def k_c(self) -> float:
        """Calibration constant k_c = R_standard / I_calibrant."""
        return self.rayleigh() / self.calibrant_intensity


# ===========================================================================
# Forward models (pure)
# ===========================================================================
def solvent_intensity(angle_deg: float, i_ref_90: float) -> float:
    """Isotropic solvent baseline: I ∝ 1/sin θ (constant scattering volume × 1/sinθ)."""
    return i_ref_90 / math.sin(math.radians(angle_deg))


def excess_rayleigh(angle_deg: float, c_g_per_mL: float, mw: float, rg_nm: float,
                    a2_mol_mL_per_g2: float, *, wavelength_nm: float,
                    n_solvent: float, dn_dc: float) -> float:
    """Excess Rayleigh ratio ΔR (cm^-1) from the Zimm equation, run forwards.

        Kc/ΔR = (1/Mw)(1 + q² Rg²/3) + 2 A2 c   ⇒   ΔR = K c / (that ordinate)
    """
    q = phys.scattering_vector_q(angle_deg, wavelength_nm, n_solvent)   # nm^-1
    K = phys.optical_constant_K(n_solvent, dn_dc, wavelength_nm)
    ordinate = (1.0 / mw) * (1.0 + q * q * rg_nm * rg_nm / 3.0) + 2.0 * a2_mol_mL_per_g2 * c_g_per_mL
    return K * c_g_per_mL / ordinate


def sample_intensity(angle_deg: float, c_g_per_mL: float, mw: float, rg_nm: float,
                     a2_mol_mL_per_g2: float, *, wavelength_nm: float,
                     n_solvent: float, dn_dc: float, k_c: float, n_standard: float,
                     i_ref_90: float) -> float:
    """Measured intensity = solvent baseline + the excess from ΔR, un-calibrated.

    Inverts the analysis path ΔR = k_c · sinθ · (I_sample − I_solvent) · (n/n_std)²,
    so the program recovers ΔR (and hence Mw/Rg/A2) from the generated intensity.
    """
    base = solvent_intensity(angle_deg, i_ref_90)
    if c_g_per_mL == 0:
        return base
    dR = excess_rayleigh(angle_deg, c_g_per_mL, mw, rg_nm, a2_mol_mL_per_g2,
                         wavelength_nm=wavelength_nm, n_solvent=n_solvent, dn_dc=dn_dc)
    s = math.sin(math.radians(angle_deg))
    return base + dR / (k_c * s * (n_solvent / n_standard) ** 2)


def rh_at_concentration(rh0_nm: float, c_g_per_mL: float,
                        kd_mL_g: float = DEFAULT_KD_ML_G) -> float:
    """Apparent Rh shrinking with concentration: D(c)=D0(1+kD c) ⇒ Rh = Rh0/(1+kD c)."""
    return rh0_nm / (1.0 + kd_mL_g * c_g_per_mL)


# ===========================================================================
# Artifact data containers
# ===========================================================================
@dataclass
class SyntheticTrace:
    """A generated count-rate trace in canonical units (s, cps)."""
    times_s: np.ndarray
    count_rates_cps: np.ndarray
    label: str = ''


@dataclass
class MultiAngleDLS:
    """Correlograms (and a count-rate trace) at several angles, one ALV-file's worth."""
    angles_deg: List[float]
    delay_times_s: np.ndarray
    signals: Dict[float, np.ndarray]          # angle -> g2-1 column
    trace_times_s: np.ndarray
    trace_cps: Dict[float, np.ndarray]        # angle -> count rate (cps)
    mean_cr_kHz: Dict[float, float]
    # context (for writing + building measurements)
    label: str
    concentration_g_per_mL: float
    temperature_K: float
    viscosity_Pa_s: float
    n_solvent: float
    wavelength_nm: float
    output_form: str = 'g2m1'


@dataclass
class SyntheticSLSSet:
    """An SLS intensity set: intensities over an angle grid for several concentrations.

    ``concentrations_g_per_mL`` always includes 0.0 (the solvent reference) first.
    A full Zimm set has many angles × many concentrations; a single-concentration
    slice has many angles × one concentration; a single-angle slice has one angle ×
    many concentrations. The writer and the measurement-builder handle all three.
    """
    angles_deg: np.ndarray
    concentrations_g_per_mL: List[float]
    intensities: Dict[float, np.ndarray]      # concentration -> intensity over angles
    # context
    label: str
    polymer_name: str
    solvent_name: str
    temperature_K: float
    wavelength_nm: float
    n_solvent: float
    dn_dc: float
    cal: CalibrationSpec
    kind: str = 'zimm'                        # 'zimm' | 'single_concentration' | 'single_angle'

    @property
    def calibrated(self) -> bool:
        return self.cal.include_in_files


# ===========================================================================
# Builders (pure — return data, no I/O, no plotting)
# ===========================================================================
def build_correlogram(populations: Sequence[SyntheticPopulation], *, angle_deg: float,
                      wavelength_nm: float, solvent_refractive_index: float,
                      temperature_K: float, viscosity_Pa_s: float, beta: float = 0.8,
                      noise_level: float = 0.0, n_points: int = 200,
                      output_form: str = 'g2m1',
                      seed: Optional[int] = None) -> SyntheticCorrelogramResult:
    """A single correlogram (thin pass-through to the utilities generator)."""
    return generate_synthetic_correlogram(
        populations, angle_deg=angle_deg, wavelength_nm=wavelength_nm,
        solvent_refractive_index=solvent_refractive_index, temperature_K=temperature_K,
        viscosity_Pa_s=viscosity_Pa_s, beta=beta, noise_level=noise_level,
        n_points=n_points, output_form=output_form, seed=seed)


def build_count_rate_trace(*, duration_s: float = 120.0, dt_s: float = 0.1,
                           mean_kHz: float = 300.0, trace_cv: float = 0.03,
                           drift: float = 0.0, spikes: int = 0,
                           seed: Optional[int] = None,
                           label: str = 'synthetic trace') -> SyntheticTrace:
    """A noisy count-rate trace: shot-noise scatter + optional slow drift + dust spikes."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, dt_s)
    cr = mean_kHz * (1.0 + trace_cv * rng.standard_normal(t.size))
    if drift and t.size:
        cr = cr + mean_kHz * drift * (t / t[-1])
    for _ in range(int(spikes)):
        cr[rng.integers(t.size)] *= rng.uniform(2.0, 3.5)
    cr = np.clip(cr, 1.0, None)
    return SyntheticTrace(times_s=t, count_rates_cps=cr * 1000.0, label=label)


def build_multi_angle_dls(populations_at_angle, *, angles_deg: Sequence[float],
                          wavelength_nm: float, solvent_refractive_index: float,
                          temperature_K: float, viscosity_Pa_s: float, beta: float = 0.85,
                          noise_level: float = 0.0, n_points: int = 200,
                          trace_cv: float = 0.03, drift: float = 0.0, spikes: int = 0,
                          label: str = '', concentration_g_per_mL: float = 0.0,
                          seed: Optional[int] = None) -> MultiAngleDLS:
    """Correlograms at every angle (+ a count-rate trace per angle), for one sample.

    ``populations_at_angle`` is either a fixed list of SyntheticPopulation (same
    sizes at every angle) or a callable ``angle_deg -> list`` (rarely needed). The
    per-angle seed is offset by the integer angle so the curves are independent.
    """
    angles = [float(a) for a in angles_deg]
    signals: Dict[float, np.ndarray] = {}
    delay = None
    for a in angles:
        pops = populations_at_angle(a) if callable(populations_at_angle) else populations_at_angle
        g = generate_synthetic_correlogram(
            pops, angle_deg=a, wavelength_nm=wavelength_nm,
            solvent_refractive_index=solvent_refractive_index, temperature_K=temperature_K,
            viscosity_Pa_s=viscosity_Pa_s, beta=beta, noise_level=noise_level,
            n_points=n_points, output_form='g2m1',
            seed=(None if seed is None else seed + int(a)))
        delay = g.delay_times_s
        signals[a] = g.signal

    # count-rate traces: more scattering at low angle; shot noise + drift/spikes
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 60.0, 0.05)
    trace_cps: Dict[float, np.ndarray] = {}
    mean_cr: Dict[float, float] = {}
    for a in angles:
        base = 350.0 * (math.sin(math.radians(95)) / math.sin(math.radians(a))) ** 0.5
        cr = base * (1.0 + trace_cv * rng.standard_normal(t.size))
        if drift:
            cr = cr + base * drift * (t / t[-1])
        for _ in range(int(spikes)):
            cr[rng.integers(t.size)] *= rng.uniform(2.0, 3.5)
        cps = np.clip(cr, 1.0, None) * 1000.0
        trace_cps[a] = cps
        mean_cr[a] = float(cps.mean() / 1000.0)        # kHz

    return MultiAngleDLS(
        angles_deg=angles, delay_times_s=delay, signals=signals,
        trace_times_s=t, trace_cps=trace_cps, mean_cr_kHz=mean_cr,
        label=label, concentration_g_per_mL=concentration_g_per_mL,
        temperature_K=temperature_K, viscosity_Pa_s=viscosity_Pa_s,
        n_solvent=solvent_refractive_index, wavelength_nm=wavelength_nm)


def build_sls_set(*, mw: float, rg_nm: float, a2_mol_mL_per_g2: float,
                  angles_deg: Sequence[float], concentrations_g_per_mL: Sequence[float],
                  wavelength_nm: float, temperature_K: float, n_solvent: float,
                  dn_dc: float, cal: CalibrationSpec, solvent_intensity_90: float,
                  noise_level: float = 0.0, seed: Optional[int] = None,
                  polymer_name: str = '', solvent_name: str = '', label: str = '',
                  kind: str = 'zimm') -> SyntheticSLSSet:
    """Build an SLS intensity set (a Zimm grid or a single-angle/-concentration slice).

    The solvent reference (c = 0) is always included as the first concentration; it
    is what the analysis subtracts to form the excess Rayleigh ratio.
    """
    rng = np.random.default_rng(seed)
    k_c = cal.k_c()
    angles = np.asarray([float(a) for a in angles_deg], dtype=float)
    concs = sorted({float(c) for c in concentrations_g_per_mL})
    if not any(c == 0.0 for c in concs):
        concs = [0.0] + concs
    intensities: Dict[float, np.ndarray] = {}
    for c in concs:
        I = np.array([sample_intensity(
            a, c, mw, rg_nm, a2_mol_mL_per_g2, wavelength_nm=wavelength_nm,
            n_solvent=n_solvent, dn_dc=dn_dc, k_c=k_c, n_standard=cal.n_standard,
            i_ref_90=solvent_intensity_90) for a in angles], dtype=float)
        if noise_level and c > 0:
            I = I * (1.0 + noise_level * rng.standard_normal(I.size))
        intensities[c] = np.clip(I, 1.0, None)
    return SyntheticSLSSet(
        angles_deg=angles, concentrations_g_per_mL=concs, intensities=intensities,
        label=label, polymer_name=polymer_name, solvent_name=solvent_name,
        temperature_K=temperature_K, wavelength_nm=wavelength_nm, n_solvent=n_solvent,
        dn_dc=dn_dc, cal=cal, kind=kind)


# ===========================================================================
# Writers (file I/O — the loadable instrument-file exception)
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
    L += ['Duration [s]    :\t60.0', 'Runs            :\t1',
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
    a user who loads it without entering a calibrant gets the uncalibrated flag.
    """
    angles = [float(a) for a in sls.angles_deg]
    concs_mg = [c * 1e3 for c in sls.concentrations_g_per_mL]
    rows: List[Tuple[str, str]] = [
        ('Date/Time', '1/1/2026 12:00:00 PM'),
        ('Sample ID', sls.label or f'{sls.polymer_name} in {sls.solvent_name}'),
        ('Operator ID', 'Synthetic'),
        ('Sample Liquid', sls.solvent_name or 'Water'),
        ('Refractive Index of Sample Liquid', f'{sls.n_solvent}'),
        ('Refractive Index Inc. (dn/dc) (mL/g)', f'{sls.dn_dc}'),
    ]
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


def write_trace_csv(path: str, trace: SyntheticTrace, *, count_rate_unit: str = 'kcps') -> str:
    """Write a count-rate trace as a plain two-column CSV (time s, count rate).

    The default writes kcps (matching the bundled trace test file); load it via the
    generic trace parser, choosing time = s and the matching count-rate unit.
    """
    factor = {'cps': 1.0, 'kcps': 1e-3, 'Mcps': 1e-6}[count_rate_unit]
    cps = np.asarray(trace.count_rates_cps, dtype=float) * factor
    t = np.asarray(trace.times_s, dtype=float)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(f'{ti:.2f},{ci:.3f}' for ti, ci in zip(t, cps)) + '\n')
    return path


# ===========================================================================
# Full-dataset orchestrator (writes a whole test-data/Synthetic * folder)
# ===========================================================================
def generate_full_dataset(out_dir: str, profile_name: str, *,
                          series: Optional[Dict[str, dict]] = None) -> List[str]:
    """Regenerate a complete synthetic dataset folder (no pictures).

    Writes one SLS Zimm ``.csv`` per series sample, the DLS ``.ASC`` files (a
    concentration series for one sample, a single concentration for the rest), a
    bimodal DLS sample, a standalone trace, and ``parameters.txt``. Returns the list
    of file paths written. ``profile_name`` keys into :data:`PROFILES`. Picture
    rendering is added by the ``internal/`` wrapper, which may import the plotting layer.
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

    # DLS: a full concentration series for one sample, one concentration for the rest
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
                label=f'{name} in Water', concentration_g_per_mL=c, seed=prof['seed'])
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
        seed=prof['seed'])
    written.append(write_alv_asc(
        os.path.join(out_dir, 'DLS - Bimodal 20nm + 200nm.ASC'), bimodal,
        viscosity_cp=DEFAULT_VISCOSITY_CP))

    # standalone two-column trace (generic-trace load path)
    trace = build_count_rate_trace(
        mean_kHz=300.0, trace_cv=prof['trace_cv'], drift=prof['drift'],
        spikes=prof['spikes'], seed=prof['seed'])
    written.append(write_trace_csv(
        os.path.join(out_dir, 'Trace - count rate (kcps vs s).csv'), trace))

    written.append(write_parameters_txt(out_dir, profile_name, series))
    return written


def write_parameters_txt(out_dir: str, profile_name: str,
                         series: Optional[Dict[str, dict]] = None) -> str:
    """Write the human-readable parameters.txt (what to enter + ground truth)."""
    import os
    series = series or DEFAULT_SERIES
    messy = 'Messy' in profile_name
    cal = CalibrationSpec()
    lines = [
        f'DLS Buddy — {profile_name} test data', '=' * 50, '',
        'System: PEG in water, calibrated against a toluene standard.', '',
        'GLOBAL PARAMETERS (enter these at the Data-tab confirmation step):',
        f'  Solvent                 : water',
        f'  Solvent refractive index: {DEFAULT_N_SOLVENT}',
        f'  Viscosity               : {DEFAULT_VISCOSITY_CP} cP   (= {DEFAULT_VISCOSITY_PA_S:g} Pa.s)',
        f'  Temperature             : {DEFAULT_TEMPERATURE_C} C  ({DEFAULT_TEMPERATURE_K} K)',
        f'  Wavelength              : {DEFAULT_WAVELENGTH_NM:g} nm',
        f'  dn/dc                   : {DEFAULT_DN_DC} mL/g', '',
        'SLS CALIBRATION (enter in the SLS tab calibration panel):',
        f'  Standard                : toluene   (geometry {cal.geometry})',
        f'  Standard refractive index: {cal.n_standard}',
        f'  Calibrant intensity     : {cal.calibrant_intensity:g}  at 90 deg',
        f'  Dark count rate         : 0',
        f'  (the program then computes k_c = {cal.k_c():.4e})', '',
        'Notes:',
        '  - DLS files are ALV .ASC: all angles + a count-rate trace are in one file.',
        '    Enter polymer name, solvent (water), and the concentration (in the file',
        '    name) for each. Temperature/viscosity/n/wavelength are read from the file.',
        '  - SLS files are Brookhaven .csv (a full Zimm set: solvent reference at c=0',
        '    plus 5 concentrations x 12 angles). Enter polymer name + temperature.',
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
              '  rho = Rg/Rh ~ 1.5 for every series sample (good-solvent coil).',
              '  Rg-Mw scaling exponent nu ~ 0.585; A2-Mw exponent ~ -0.2.',
              '  DLS shows a mild concentration dependence (kD ~ +30 mL/g), so the',
              '  D-vs-c extrapolation has a small positive slope (PEG 1M has 3 concs).']
    path = os.path.join(out_dir, 'parameters.txt')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    return path
