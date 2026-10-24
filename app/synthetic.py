"""
app/synthetic.py
================

The Synthetic Generator's recipe: which builder each artifact kind calls, with which
inputs, how a ticked artifact is saved, and in which order the single-sample injection
adds built artifacts to the workspace.

Qt-free (invariant 5). The Utilities tab reads its fields into a `SynthRequest` and hands
it here: `build` and `save` run on the worker thread, `inject` on the main thread, since it
changes the workspace the widgets iterate. The physics is `analysis/synthetic_dataset.py`'s
and the file formats `exporting/instrument_files.py`'s, both reached through the
controller's generator methods.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from app.controller import Controller


# Default save filenames per single-file artifact (written into the chosen folder).
# 'depolarized' writes two ALV files (VV + VH), 'architecture'/'full_set' write subfolders,
# so they are handled specially in save (no single-name entry here).
_SYNTH_FILENAMES = {
    'correlogram': 'Synthetic correlogram.csv',
    'trace': 'Synthetic trace (kcps vs s).csv',
    'multi_dls': 'Synthetic DLS multi-angle.ASC',
    'two_mode': 'Synthetic DLS two-mode.ASC',
    'sls_zimm': 'Synthetic SLS Zimm.csv',
    'sls_ff_sphere': 'Synthetic SLS form-factor sphere.csv',
    'sls_ff_debye': 'Synthetic SLS form-factor coil.csv',
    'sls_ff_ellipsoid': 'Synthetic SLS form-factor ellipsoid.csv',
    'sls_ff_mie': 'Synthetic SLS form-factor Mie sphere.csv',
    'sls_ff_core_shell': 'Synthetic SLS form-factor core-shell.csv',
    'sls_single_conc': 'Synthetic SLS single-concentration.csv',
    'sls_single_angle': 'Synthetic SLS single-angle.csv',
}


@dataclass(kw_only=True)
class SynthRequest:
    """The generator's inputs, read from the Utilities tab's fields as plain values.

    Units are the builders' own: wavelength in nm, temperature in K, viscosity in Pa·s,
    concentrations in g/mL, sizes in nm. `specs` holds the size populations
    (`rh_nm`, `weight`, `spread_cv`), and `polymer` and `solvent` the sample's names.
    `with_calibration` is the generator's calibration box: a saved SLS file carries the
    calibration metadata, and an added SLS set prefills the session calibration. `build`
    reads only the fields its artifact kind needs.
    """

    specs: List[Dict[str, float]]
    polymer: str
    solvent: str
    wl: float
    n: float
    T: float
    eta: float
    angle: float
    beta: float
    noise: float
    baseline: float
    npts: int
    axis_kw: Dict[str, Any]
    mw: float
    rg: float
    a2: float
    dn_dc: float
    angles: List[float]
    concs: List[float]
    with_calibration: bool
    dn_dc_in_files: bool
    seed: Optional[int]
    rho_v: float
    vh_noise: float
    shape: str
    dims_nm: Tuple[float, float]
    trace_kw: Dict[str, Any]
    weight_basis: str = 'intensity'
    comp_model: str = 'rayleigh'
    comp_solute_ri: Optional[float] = None


def build(controller: 'Controller', key: str, request: SynthRequest):
    """Build one artifact kind from `request`; the object the tab previews and saves."""
    c = controller
    specs, poly, solv = request.specs, request.polymer, request.solvent
    # A non-intensity weight basis (Number/Volume/Surface) treats the population
    # weights as a composition: forward-model the intensity (Rayleigh, angle-
    # independent) so a generated DLS set reweights back to this composition. Only the
    # DLS correlogram artifacts consume the size populations, so convert just for them.
    basis = request.weight_basis
    if key in ('correlogram', 'multi_dls') and basis != 'intensity' and specs:
        angle = request.angle if key == 'correlogram' else (
            request.angles[0] if request.angles else 90.0)
        comp_model = request.comp_model
        # RDG/Mie form the contrast m from the solute RI; Rayleigh ignores it. Distinguish
        # "not entered" (None) from "entered a non-physical value" (0, negative, or a
        # non-finite inf/nan) so the error is honest rather than reporting an entered 0
        # as "missing".
        n_sol = request.comp_solute_ri
        m = None
        if comp_model in ('rdg', 'mie'):
            if n_sol is None:
                raise ValueError(
                    f"The {comp_model.upper()} composition model needs the solute refractive "
                    "index (enter it next to the weight basis).")
            if not (math.isfinite(n_sol) and n_sol > 0):
                raise ValueError(
                    f"The solute refractive index must be a positive finite number, "
                    f"got {n_sol:g}.")
            m = n_sol / request.n
        specs = c.synth_composition_to_intensity_specs(
            specs, basis, angle_deg=angle, wavelength_nm=request.wl,
            n_solvent=request.n, model=comp_model, m=m)
    if key == 'correlogram':
        if not specs:
            raise ValueError('Add at least one size population for the correlogram.')
        return c.synth_correlogram(
            specs, angle_deg=request.angle, wavelength_nm=request.wl,
            solvent_refractive_index=request.n, temperature_K=request.T,
            viscosity_Pa_s=request.eta, beta=request.beta, noise_level=request.noise,
            baseline=request.baseline,
            n_points=request.npts, seed=request.seed, **request.axis_kw)
    if key == 'trace':
        return c.synth_trace(seed=request.seed, label=f'{poly} / {solv}',
                             **request.trace_kw)
    if key == 'depolarized':
        if not request.angles:
            raise ValueError('Enter at least one angle for the DDLS set.')
        # Same concentration rule as 'multi_dls' below: the VV/VH correlograms are the
        # sample's DLS side, so they carry the entered c rather than 0, and the Data
        # tab shows one concentration across the sample's DLS and SLS rows.
        return c.synth_depolarized(
            rho_v=request.rho_v, shape=request.shape, dims_nm=request.dims_nm,
            angles_deg=request.angles, wavelength_nm=request.wl, n_solvent=request.n,
            temperature_K=request.T, viscosity_Pa_s=request.eta, beta=request.beta,
            vh_noise_level=request.vh_noise, n_points=request.npts, seed=request.seed,
            label=f'{poly} {solv} DDLS',
            concentration_g_per_mL=(request.concs[0] if request.concs else 6e-4),
            baseline=request.baseline)
    if key == 'mode_resolved':
        # The linked set (two-mode correlograms + exact-mean traces +
        # per-angle static truth from ONE partition ground truth). The
        # truth knobs are fixed (slow Rg 120 nm, fast fraction 0.3 at
        # q -> 0, crossing at 82°) -- the linkage is the point, the same
        # fixed-parameter idiom as 'two_mode' above.
        if not request.angles:
            raise ValueError('Enter at least one angle.')
        conc0 = request.concs[0] if request.concs else 5e-3
        return c.synth_mode_resolved_tuple(
            angles_deg=request.angles, wavelength_nm=request.wl,
            n_solvent=request.n, temperature_K=request.T,
            viscosity_Pa_s=request.eta, beta=request.beta,
            noise_level=request.noise, n_points=request.npts,
            concentration_g_per_mL=conc0, seed=request.seed,
            polymer_name=poly, solvent_name=solv,
            label=f'{poly} in {solv} (mode-resolved)', **request.axis_kw)
    if key == 'architecture':
        # The ground-truth series spec (a small dict) — drives the preview + readout;
        # the heavy per-sample build happens at save/inject time.
        return c.synth_architecture_series()
    if key == 'form_factor':
        return c.synth_form_factor_series()
    if key == 'multi_dls':
        if not specs:
            raise ValueError('Add at least one size population for the DLS set.')
        if not request.angles:
            raise ValueError('Enter at least one angle.')
        conc0 = request.concs[0] if request.concs else 6e-4
        return c.synth_multi_angle_dls(
            specs, angles_deg=request.angles, wavelength_nm=request.wl,
            solvent_refractive_index=request.n, temperature_K=request.T,
            viscosity_Pa_s=request.eta, beta=request.beta, noise_level=request.noise,
            baseline=request.baseline,
            n_points=request.npts, label=f'{poly} in {solv}',
            concentration_g_per_mL=conc0, seed=request.seed, **request.axis_kw)
    if key == 'two_mode':
        # A diffusive fast mode (the entered β/noise apply) sweeping past a
        # q-independent slow mode crossing at 82° with amplitudes 0.6/0.4 —
        # fixed here rather than via new inputs (the crossing is the point;
        # the same fixed-parameter idiom as the ellipsoid's axial ratio 3).
        if not request.angles:
            raise ValueError('Enter at least one angle.')
        conc0 = request.concs[0] if request.concs else 6e-4
        return c.synth_two_mode_dls(
            angles_deg=request.angles, wavelength_nm=request.wl,
            n_solvent=request.n, temperature_K=request.T,
            viscosity_Pa_s=request.eta, beta=request.beta,
            noise_level=request.noise, n_points=request.npts,
            label=f'{poly} in {solv} (two-mode)',
            concentration_g_per_mL=conc0, seed=request.seed,
            **request.axis_kw)
    if key.startswith('sls'):
        if not request.angles:
            raise ValueError('Enter at least one angle.')
        if not request.concs:
            raise ValueError('Enter at least one (non-zero) concentration.')
        ff_model = 'guinier'
        ax_ratio = 1.0
        if key == 'sls_zimm':
            a_list, c_list, kind = request.angles, request.concs, 'zimm'
        elif key == 'sls_ff_sphere':
            # A full Zimm grid whose intensities carry the FULL sphere P(q), so the
            # Form factor P(q) fit is exercised end-to-end from in-app ground truth.
            a_list, c_list, kind, ff_model = request.angles, request.concs, 'zimm', 'sphere'
        elif key == 'sls_ff_debye':
            a_list, c_list, kind, ff_model = request.angles, request.concs, 'zimm', 'debye'
        elif key == 'sls_ff_ellipsoid':
            # A prolate ellipsoid (axial ratio 3) so the Ellipsoid P(q) fit is exercised
            # end-to-end and its prolate/oblate degeneracy is reachable from in-app ground
            # truth. (Fixed here rather than via a new input — the shape is the point.)
            a_list, c_list, kind, ff_model = request.angles, request.concs, 'zimm', 'ellipsoid'
            ax_ratio = 3.0
        elif key == 'sls_ff_mie':
            # A latex-in-water-like high-contrast Mie sphere (relative index m ~ 1.19, a
            # fixed representative solute RI 1.585 over the chosen solvent n), so the full-Mie
            # P(q) fit is exercised end-to-end from in-app ground truth in the regime RDG and
            # the analytic sphere cannot reach. (Fixed here rather than via a new input — the
            # high contrast is the point; enter the solute RI 1.585 on the Data tab to fit it.)
            a_list, c_list, kind, ff_model = request.angles, request.concs, 'zimm', 'mie'
            mie_m = 1.585 / request.n
            return c.synth_sls_set(
                mw=request.mw, rg_nm=request.rg, a2_mol_mL_per_g2=request.a2, angles_deg=a_list,
                concentrations_g_per_mL=c_list, wavelength_nm=request.wl,
                temperature_K=request.T, n_solvent=request.n, dn_dc=request.dn_dc,
                calibrated=request.with_calibration, dn_dc_in_files=request.dn_dc_in_files,
                noise_level=request.noise, seed=request.seed,
                polymer_name=poly, solvent_name=solv, label=f'{poly} in {solv}',
                kind=kind, form_factor_model='mie', m=mie_m)
        elif key == 'sls_ff_core_shell':
            # A representative coated particle (silica-like core n ~ 1.45 in a polymer
            # shell n ~ 1.40, R_core = 40 nm, R_outer = 70 nm) so the core-shell P(q) fit
            # -- and its two-radius degeneracy -- is reachable end-to-end from in-app ground
            # truth. (Fixed here rather than via new inputs -- the coated structure is the
            # point; enter the core [solute] RI 1.45 and shell RI 1.40 on the Data tab to
            # fit it. The best solution recovers R_core / R_outer exactly.)
            n_core, n_shell = 1.45, 1.40
            return c.synth_sls_set(
                mw=request.mw, rg_nm=request.rg, a2_mol_mL_per_g2=request.a2,
                angles_deg=request.angles, concentrations_g_per_mL=request.concs,
                wavelength_nm=request.wl, temperature_K=request.T, n_solvent=request.n,
                dn_dc=request.dn_dc, calibrated=request.with_calibration,
                dn_dc_in_files=request.dn_dc_in_files, noise_level=request.noise,
                seed=request.seed, polymer_name=poly, solvent_name=solv,
                label=f'{poly} in {solv}', kind='zimm', form_factor_model='core_shell',
                r_core_nm=40.0, r_outer_nm=70.0,
                m1=n_core / request.n, m2=n_shell / request.n)
        elif key == 'sls_single_conc':
            a_list, c_list, kind = request.angles, [request.concs[0]], 'single_concentration'
        else:   # sls_single_angle: the angle nearest 90°
            a_near = min(request.angles, key=lambda x: abs(x - 90.0))
            a_list, c_list, kind = [a_near], request.concs, 'single_angle'
        return c.synth_sls_set(
            mw=request.mw, rg_nm=request.rg, a2_mol_mL_per_g2=request.a2, angles_deg=a_list,
            concentrations_g_per_mL=c_list, wavelength_nm=request.wl, temperature_K=request.T,
            n_solvent=request.n, dn_dc=request.dn_dc, calibrated=request.with_calibration,
            dn_dc_in_files=request.dn_dc_in_files,
            noise_level=request.noise, seed=request.seed, polymer_name=poly,
            solvent_name=solv, label=f'{poly} in {solv}', kind=kind,
            form_factor_model=ff_model, ellipsoid_axial_ratio=ax_ratio)
    raise ValueError(f'Unknown artifact {key!r}.')


def save(controller: 'Controller', save_keys: Sequence[str], built: Dict[str, Any],
         folder: str, eta_cp: float) -> List[str]:
    """Write the save-ticked artifacts. Takes plain values (no widget reads) so
    it can run on the worker thread; the caller hoists the folder/viscosity."""
    if not save_keys:
        return []
    if not folder:
        raise ValueError('Choose a "Save to" folder first.')
    os.makedirs(folder, exist_ok=True)
    c = controller
    saved = []
    for key in save_keys:
        if key == 'full_set':
            for prof in ('Synthetic Clean', 'Synthetic Messy'):
                c.generate_full_test_set(os.path.join(folder, prof), prof)
                saved.append(prof)
            continue
        if key == 'architecture':
            c.save_synth_architecture(os.path.join(folder, 'Architecture'))
            saved.append('Architecture/')
            continue
        if key == 'mode_resolved':
            c.save_synth_mode_resolved(
                built['mode_resolved'],
                os.path.join(folder, 'Mode-Resolved'),
                viscosity_Pa_s=eta_cp / 1e3)
            saved.append('Mode-Resolved/')
            continue
        if key == 'form_factor':
            c.save_synth_form_factor(os.path.join(folder, 'Form-Factor'))
            saved.append('Form-Factor/')
            continue
        if key == 'depolarized':
            for p in c.save_synth_depolarized(built['depolarized'], folder,
                                              viscosity_cp=eta_cp):
                saved.append(os.path.basename(p))
            continue
        path = os.path.join(folder, _SYNTH_FILENAMES[key])
        obj = built[key]
        if key == 'correlogram':
            c.export_synthetic(obj, path)
        elif key == 'trace':
            c.save_synth_trace(obj, path)
        elif key == 'multi_dls':
            c.save_synth_multi_angle_dls(obj, path, viscosity_cp=eta_cp)
        elif key == 'two_mode':
            # A TwoModeDLS wraps an ordinary MultiAngleDLS; the ground
            # truth stays in-app (the preview names it), the file is a
            # normal multi-angle .ASC any DLS loader reads.
            c.save_synth_multi_angle_dls(obj.data, path, viscosity_cp=eta_cp)
        else:    # sls_*
            c.save_synth_sls_set(obj, path)
        saved.append(os.path.basename(path))
    return saved


def inject(controller: 'Controller', built: Dict[str, Any], calibrated: bool, poly: str,
           solv: str, conc0: float, T: float, eta: float) -> str:
    """Inject a coherent single sample: the primary DLS artifact (multi-angle
    preferred), the primary SLS artifact (Zimm preferred), and any trace.
    Injecting every SLS slice would pile duplicate concentrations into one
    sample, so only the primary of each kind is added.

    Takes plain values (no widget reads). It mutates the workspace, so the
    Utilities tab calls it on the main thread once the worker's build has
    landed, then emits workspaceChanged and refreshes the trace list."""
    c = controller
    parts = []
    if 'multi_dls' in built:
        c.inject_multi_angle_dls(built['multi_dls'], polymer_name=poly, solvent_name=solv)
        parts.append('multi-angle DLS')
    elif 'two_mode' in built:
        c.inject_multi_angle_dls(built['two_mode'].data, polymer_name=poly,
                                 solvent_name=solv)
        parts.append('two-mode DLS')
    elif 'correlogram' in built:
        c.inject_correlogram(built['correlogram'], polymer_name=poly, solvent_name=solv,
                             concentration_g_per_mL=conc0, temperature_K=T,
                             viscosity_Pa_s=eta)
        parts.append('correlogram')
    if 'depolarized' in built:
        c.inject_depolarized(built['depolarized'], polymer_name=poly, solvent_name=solv)
        parts.append('depolarized VV/VH (DDLS)')
    if 'mode_resolved' in built:
        n = len(c.inject_mode_resolved_tuple(
            built['mode_resolved'], polymer_name=poly, solvent_name=solv))
        parts.append(f'mode-resolved linked set ({n} DLS runs with '
                     f'traces incl. the c = 0 blank, + calibrant trace)')
    sls_added = False
    for key in ('sls_zimm', 'sls_ff_sphere', 'sls_ff_debye', 'sls_ff_ellipsoid',
                'sls_ff_mie', 'sls_ff_core_shell', 'sls_single_conc', 'sls_single_angle'):
        if key in built:
            c.inject_sls_set(built[key], polymer_name=poly, solvent_name=solv,
                             prefill_calibration=calibrated)
            parts.append(key.replace('sls_', 'SLS '))
            sls_added = True
            break
    if 'trace' in built:
        c.inject_trace(built['trace'], label=f'{poly} / {solv}')
        parts.append('trace')
    if not parts:
        return ''
    cal_note = ' (session calibration set)' if (calibrated and sls_added) else ''
    return f'added to workspace as “{poly} / {solv}”: ' + ', '.join(parts) + cal_note
