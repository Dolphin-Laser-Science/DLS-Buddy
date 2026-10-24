"""The notes registry: one row per note code, its kind, clause, guide section and the flags
it fails. The program-wide notes repository; the model is ``core/notes.py``'s docstring and
the wording rule ui_style_guide §5.

Adding or changing a note is one row here (and its remedy in ``gui/note_text.py``). A
clause is one clause of at most ``CLAUSE_MAX`` characters rendered for the screen; a
``section`` is a heading number of the Theory-and-Equations-Guide; ``fails`` names flag
FIELDS, and a class that does not carry one of them is not concerned by it; ``example``
holds the values the registry gate renders the clause at.
"""
from __future__ import annotations

from typing import Dict

from core.notes import NoteSpec

# Each recorded weighting's clause, guide section and the word a row or column label
# carries for it (`NoteSpec.label`).
_RECORDED_WEIGHTINGS = {
    'biganzoli-full': ('residual weight: Biganzoli–Ferri variance with its shot-noise terms',
                       '3.2.9', 'sourced variance'),
    'biganzoli-classical': ('residual weight: Biganzoli–Ferri variance, infinite-count-rate '
                            'limit', '3.2.9', 'sourced variance, classical'),
    'instrument': ("residual weight: the instrument's per-channel standard deviations",
                   '3.2.11', 'instrument variance'),
    'iso-a18': ("residual weight: w = y on ln(g2-1), the linear fit's form", '3.2.12',
                'linear-fit weight'),
    'uniform': ('residual weight: none, unweighted least squares', '3.2.11', 'unweighted'),
    'g1-squared': ('residual weight: g1², noise taken as constant across channels', '3.4.3',
                   'g₁² weight'),
}

# Why the Biganzoli-Ferri variance, Eq. (9), was not formed: its refusals (the set
# `tests/test_dls.py::test_every_raise_in_eq9_is_a_refusal_with_a_cause_key` pins), the causes
# both fallback rows name.
_EQ9_REFUSALS = (
    'decay per sample time ≤ 0 or not finite', 'a sample time contradicts its lag',
    'the run duration not past the last lag', 'a count rate unusable for shot noise',
    'a per-channel SD ≤ 0 or not finite')

REGISTRY: Dict[str, NoteSpec] = {
    # -- the scale gates: preconditions of Mw and absolute A2 (the blank, of every value)
    'uncalibrated': NoteSpec(
        'precondition', 'uncalibrated, arbitrary scale', '1.7.1',
        fails=('mw_reliable', 'a2_reliable')),
    'no_dn_dc': NoteSpec(
        'precondition', 'no usable dn/dc, arbitrary optical scale', '1.5.1',
        fails=('mw_reliable', 'a2_reliable')),
    'blank_unmeasured': NoteSpec(
        'precondition', 'no solvent blank measured', '1.7.2',
        fails=('mw_reliable', 'a2_reliable', 'rg_reliable', 'two_a2_mw_reliable',
               'axial_ratio_reliable')),
    # k_c applied while the standard's refractive index was blank, so the index correction
    # could not be formed and the scale is arbitrary all the same.
    'no_standard_index': NoteSpec(
        'precondition', 'no standard refractive index, arbitrary scale', '1.7.1',
        fails=('mw_reliable', 'a2_reliable')),
    # -- qualifiers, read off typed fields by core.notes.qualifiers
    'apparent': NoteSpec('qualifier', 'apparent: not a thermodynamic value', '4.3',
                         label='apparent'),
    'se_statistical': NoteSpec(
        'qualifier', '± statistical; excludes calibration & dn/dc', '6'),
    'se_regression': NoteSpec('qualifier', '± statistical only', '6'),
    'se_ols': NoteSpec('qualifier', '± from classical OLS, not HC3', '6.1.1', label='OLS'),
    'cumulant_average': NoteSpec(
        'qualifier', 'intensity-weighted harmonic mean (cumulant)', '3.2.2',
        label='cumulant'),
    'distribution_weighted': NoteSpec('qualifier', 'distribution-weighted', '3.4',
                                      label='distribution-weighted'),
    'se_excludes_inversion': NoteSpec(
        'qualifier', "± statistical; excludes the inversion's systematics", '4.2.2'),
    'mode_resolved': NoteSpec(
        'qualifier', "mode-resolved: one mode's share of the excess intensity", '4.2.2',
        label='mode-resolved'),
    'mode_family': NoteSpec(
        'qualifier', "mode-resolved: one mode family's decay rates across angles", '3.5.4',
        label='mode-resolved'),
    # A Berry Rg and Mw read past the limit a Zimm Rg and Mw are read at: what the numbers
    # are, not a failed precondition, since the program holds no limit of Berry's own.
    'berry_past_zimm_limit': NoteSpec(
        'qualifier', 'Berry Rg and Mw read at qRg(max) {qrg}, past the Zimm limit {limit}',
        '4.4',
        example={'qrg': 1.37, 'limit': 1.0}, label='past the Zimm limit'),
    'rg_below_resolution': NoteSpec(
        'qualifier', 'Rg {rg} nm < {floor} nm, the q-range resolution floor', '4.2.1',
        example={'rg': 8.2, 'floor': 14.0}),
    'degenerate': NoteSpec(
        'qualifier', '{n_other} other solution(s) fit equally; {shown} shown', '4.9.1',
        example={'n_other': 1, 'shown': 'prolate'},
        tokens={'shown': ('prolate', 'oblate', 'one')}),
    'model_dependent': NoteSpec(
        'qualifier', 'model-dependent: assumes the {model} shape', '4.8',
        example={'model': 'sphere'}, label='model-dependent'),
    'degenerate_pick': NoteSpec(
        'qualifier', 'user-selected among degenerate shape solutions', '4.9.1',
        label='degenerate'),
    'includes_model_dependent': NoteSpec(
        'qualifier', 'includes a model-dependent (form-factor) value', '4.8',
        label='includes model-dependent values'),
    'includes_degenerate_pick': NoteSpec(
        'qualifier', 'includes a user-selected degenerate shape solution', '4.9.1',
        label='includes a degenerate pick'),
    'branching_model_dependent': NoteSpec(
        'qualifier', 'model-dependent: n read under the chosen branching model', '5.3.1',
        label='model-dependent'),
    'model_weighted_view': NoteSpec(
        'qualifier', '{weighting}-weighted: a model-dependent view ({model})', '3.7',
        example={'weighting': 'number', 'model': 'RDG sphere'}),
    'vh_family': NoteSpec(
        'qualifier', 'VH family: the slope carries rotation, not a translational D', '3.6.1',
        label='VH family'),
    **{f'weighting.{value}': NoteSpec('qualifier', clause, section,
                                      example={'weighting': value}, label=label)
       for value, (clause, section, label) in _RECORDED_WEIGHTINGS.items()},
    # -- DLS: the size and its viscosity (every arm)
    'no_viscosity': NoteSpec(
        'precondition', 'Rh not computed: no viscosity on the measurement', '1.3',
        fails=('rh_reliable',)),
    'rh_above_ceiling': NoteSpec(
        'precondition', 'Rh {rh} nm > {limit} nm, the DLS upper limit', '3.1.5',
        fails=('rh_reliable',), example={'rh': 6200.0, 'limit': 5000.0}),
    'rh_below_floor': NoteSpec(
        'precondition', 'Rh {rh} nm < {limit} nm, the hydrodynamic size floor', '3.1.5',
        fails=('rh_reliable',), example={'rh': 0.12, 'limit': 0.2}),
    'decay_incomplete': NoteSpec(
        'precondition', 'decay incomplete in the delay window: {quantity} {value} {op} {limit}',
        '3.1.4', fails=('rh_reliable',),
        example={'quantity': 'Γ·Δτ', 'value': 0.8, 'op': '<', 'limit': 1.96},
        tokens={'quantity': ('Γ·Δτ', 'lowest g2-1 ratio'), 'op': ('<', '>')}),
    'dead_channels': NoteSpec(
        'record', '{n_dead} of {n} channels collected no coincidence, dropped', '3.1.1',
        example={'n_dead': 31, 'n': 180}),
    # `channels` says which channels were counted: all those in the window, or only the
    # live ones where dead channels sit inside it.
    'window_too_few_channels': NoteSpec(
        'precondition', '{n} {channels} in the delay window, fewer than the {need} the fit needs',
        '3.1.1', fails=('success', 'rh_reliable', 'pdi_valid'),
        example={'n': 3, 'channels': 'live channel(s)', 'need': 4},
        tokens={'channels': ('channel(s)', 'live channel(s)')}),
    # -- DLS: the cumulant fits
    'pdi_above_limit': NoteSpec(
        'precondition', 'PDI {pdi} > {limit}, the cumulant limit', '3.2.1',
        fails=('rh_reliable', 'pdi_valid'), example={'pdi': 0.914, 'limit': 0.3}),
    'gamma_nonpositive': NoteSpec(
        'precondition', 'decay rate Γ {gamma} 1/s ≤ {limit}', '3.2.3',
        fails=('rh_reliable', 'pdi_valid'), example={'gamma': -12.5, 'limit': 0.0}),
    'retained_range_too_short': NoteSpec(
        'precondition', '{n} point(s) above the {cutoff} cutoff, fewer than the {need} '
        'order {order} needs', '3.2.3', fails=('success', 'rh_reliable', 'pdi_valid'),
        example={'n': 3, 'cutoff': 0.1, 'need': 4, 'order': 3}),
    'too_few_live_channels': NoteSpec(
        'precondition', '{n_live} of {n} channels live, fewer than the {need} order {order} '
        'needs', '3.2.11', fails=('success', 'rh_reliable', 'pdi_valid'),
        example={'n_live': 5, 'n': 180, 'need': 6, 'order': 3}),
    'fit_not_converged': NoteSpec(
        'precondition', 'fit did not converge', '3.2.5',
        fails=('success', 'rh_reliable', 'pdi_valid')),
    # The nonlinear arm's rejection of a fit that did converge, below the floor whose Rh
    # it withholds (CLAUDE.md, invariant 4's owner-ruled NaN exception).
    'solution_below_floor': NoteSpec(
        'precondition', 'fit converged to Rh {rh} nm < {limit} nm, the hydrodynamic size '
        'floor', '3.1.5', fails=('success', 'rh_reliable', 'pdi_valid'),
        example={'rh': 0.15, 'limit': 0.2}),
    # The nonlinear arm's other withheld returns, each naming its own cause. A converged
    # solution with a non-physical parameter is the form-factor fits' cause too, so its
    # `fails` spans both families and each class judges the flags it carries.
    'solution_nonphysical': NoteSpec(
        'precondition', 'fit converged to a non-physical {quantity}', '6.6',
        fails=('converged', 'mw_reliable', 'a2_reliable', 'rg_reliable', 'success',
               'rh_reliable', 'pdi_valid'),
        example={'quantity': 'β'},
        tokens={'quantity': ('Γ', 'β', 'µ₂', 'µ₃', 'µ₄', '1/Mw', 'Mw', 'Rg', 'R_core')}),
    'baseline_bound_unformable': NoteSpec(
        'precondition', 'no finite positive bound for the floating baseline; fit withheld',
        '3.2.5', fails=('success', 'rh_reliable', 'pdi_valid')),
    'first_pass_failed': NoteSpec(
        'precondition', "the sourced weight's first pass did not fit; fit withheld", '3.2.11',
        fails=('success', 'rh_reliable', 'pdi_valid')),
    # The linear arm's closed-form regression, which has no optimiser that could fail.
    'coefficients_not_finite': NoteSpec(
        'precondition', 'regression coefficients not finite; fit withheld', '3.2.3',
        fails=('success', 'rh_reliable', 'pdi_valid')),
    # The instrument's own residual weight refused, the fit with it: one token per cause on
    # which the caller's per-channel standard deviations are refused. A sourced weight that
    # is not formed never refuses the fit; it falls back (the two fallback rows below).
    'weight_inputs_refused': NoteSpec(
        'precondition', 'residual weight not formed: {cause}', '3.2.11',
        fails=('success', 'rh_reliable', 'pdi_valid'),
        example={'cause': 'no instrument standard deviations'},
        tokens={'cause': (
            'no instrument standard deviations', 'instrument SDs not one per channel',
            'an instrument SD ≤ 0 or not finite')}),
    # The default sourced weight not formed, so the fit ran unweighted: a record of how the
    # fit was obtained and never a failed precondition, so no verdict reads it (a
    # recommendation the owner accepted, 2026-10-16). `cause` is why Eq. (9) refused.
    'weight_fallback_uniform': NoteSpec(
        'record', 'unweighted fallback: {cause}', '3.2.11',
        example={'cause': 'a sample time contradicts its lag'},
        tokens={'cause': _EQ9_REFUSALS}),
    'moment_on_bound': NoteSpec(
        'record', '{moment} at its lower bound {bound}, so the value shown is that bound',
        '3.2.7', example={'moment': 'µ₂', 'bound': 0.0}, tokens={'moment': ('µ₂', 'µ₄')}),
    'baseline_on_bound': NoteSpec(
        'record', 'floating baseline at its bound, {d_b} from the tail mean {b0}', '3.2.5',
        example={'d_b': 0.0021, 'b0': 1.0003}),
    'sample_time_from_axis': NoteSpec(
        'record', 'per-channel sample time derived from the delay axis', '3.2.9'),
    # -- DLS: the distribution fits and the CONTIN alpha selection
    # The distribution arms' fallback to the g1² weight, a record like the cumulant's;
    # `cause` is why the sourced weight was not formed, Eq. (9)'s refusal or its own.
    'weight_fallback_g1': NoteSpec(
        'record', 'g1² weight fallback: {cause}', '3.4.3',
        example={'cause': 'a first-pass spectrum of zeros'},
        tokens={'cause': _EQ9_REFUSALS + (
            'the first pass did not fit', 'a weight entry < 0 or not finite',
            'a first-pass spectrum of zeros', 'a scale anchor outside the float range')}),
    'alpha_at_floor': NoteSpec(
        'precondition', 'CONTIN α at the sweep floor {alpha}', '3.4.5',
        fails=('alpha_reliable',), example={'alpha': 1e-08}),
    'alpha_at_ceiling': NoteSpec(
        'precondition', 'CONTIN α at the sweep ceiling {alpha}', '3.4.5',
        fails=('alpha_reliable',), example={'alpha': 30.0}),
    'alpha_no_corner': NoteSpec(
        'precondition', 'L-curve with no corner; least-regularized α {alpha} reported',
        '3.4.8', fails=('alpha_reliable',), example={'alpha': 1e-08}),
    'alpha_one_point': NoteSpec(
        'precondition', 'one-point α sweep: no interior to select from', '3.4.5',
        fails=('alpha_reliable',)),
    'ftest_reference': NoteSpec(
        'precondition', 'F-test reference penalty/residual {ratio} > {limit}', '3.4.9',
        fails=('alpha_reliable',), example={'ratio': 0.004, 'limit': 0.001}),
    'ftest_band': NoteSpec(
        'precondition', 'F-test probability to reject outside {lo}–{hi}', '3.4.9',
        fails=('alpha_reliable',), example={'lo': 0.05, 'hi': 0.95}),
    'alpha_selector': NoteSpec(
        'record', 'α chosen by {selector}', '3.4.5', example={'selector': 'L-curve corner'}),
    'alpha_sweep': NoteSpec(
        'record', 'α swept from {lo} to {hi}', '3.4.5', example={'lo': 1e-08, 'hi': 30.0},
        precision={'lo': 6, 'hi': 6}),
    'ftest_level': NoteSpec(
        'record', 'F-test probability to reject {p}', '3.4.9', example={'p': 0.5},
        precision={'p': 6}),
    'grid_malformed': NoteSpec(
        'precondition', 'decay-rate grid length differs from the Rh grid', '3.4.1',
        fails=('success',)),
    # A peak whose lobe maximum is the grid's first or last node, so the grid does not
    # contain the population the lobe stands for; its values the end and the node's Rh.
    'grid_end': NoteSpec(
        'precondition', 'peak maximum on the {end} grid point, {rh} nm', '3.4.1',
        fails=('rh_reliable',), example={'end': 'last', 'rh': 1000.0},
        tokens={'end': ('first', 'last')}),
    # A degenerate inversion's withheld summary: a precondition of the summary values that
    # fails no flag.
    'summary_withheld': NoteSpec(
        'precondition', 'solution weights sum ≤ 0 or not finite; summary withheld', '3.4.1'),
    'baseline_above_data': NoteSpec(
        'precondition', 'baseline {baseline} ≥ {peak}, the peak of the windowed correlogram',
        '3.1.3', fails=('success', 'rh_reliable', 'pdi_valid'),
        example={'baseline': 0.0213, 'peak': 0.0198}),
    # -- DLS: the pooled lines (Γ vs q², D vs c) and their inputs
    'inputs_flagged': NoteSpec(
        'precondition', '{k} of {n} {inputs} carry a flagged fit', '3.5.3',
        fails=('rh_reliable',), example={'k': 2, 'n': 8, 'inputs': 'angles'},
        tokens={'inputs': ('angles', 'concentrations')}),
    # `fields` names the identity fields the check compares, not the ones that differ.
    'mixed_sample_identity': NoteSpec(
        'record', 'inputs differ in identity ({fields})', '3.5',
        example={'fields': ('polymer', 'solvent', 'concentration', 'temperature')},
        tokens={'fields': ('polymer', 'solvent', 'concentration', 'temperature')}),
    'unusable_rate_dropped': NoteSpec(
        'record', '{n_dropped} of {n} {points} dropped: no usable {quantity}', '3.5.3',
        example={'n_dropped': 1, 'n': 8, 'points': 'angles', 'quantity': 'Γ'},
        tokens={'points': ('angles', 'concentrations', 'modes', 'runs'),
                'quantity': ('Γ', 'D', 'VV Γ')}),
    'too_few_points': NoteSpec(
        'precondition', '{n} usable {points}, fewer than the {need} the line needs', '3.5.2',
        fails=('rh_reliable', 'is_diffusive', 'kd_reliable', 'well_conditioned'),
        example={'n': 1, 'points': 'angles', 'need': 2},
        tokens={'points': ('angles', 'concentrations')}),
    'd0_not_finite': NoteSpec(
        'precondition', 'D0 not finite: no infinite-dilution intercept', '3.5.2',
        fails=('kd_reliable', 'rh_reliable')),
    'd0_nonpositive': NoteSpec(
        'precondition', 'D0 {d0} m²/s ≤ {limit}', '3.5.2',
        fails=('kd_reliable', 'rh_reliable'), example={'d0': -2.1e-12, 'limit': 0.0}),
    'kd_not_finite': NoteSpec(
        'precondition', 'kD not finite; value withheld', '3.5.2', fails=('kd_reliable',)),
    # The inputs whose own flag is down for a cause in DECAY_BEARING_CODES (below): a
    # flagged decay rate fails the line's kD as well as its Rh, and DDLS's D_t, the
    # through-origin line on its VV rates.
    'inputs_flagged_decay': NoteSpec(
        'precondition', '{k} of {n} {inputs} carry a flagged decay rate', '3.5.3',
        fails=('rh_reliable', 'kd_reliable', 'd_t_reliable'),
        example={'k': 2, 'n': 8, 'inputs': 'angles'},
        tokens={'inputs': ('angles', 'concentrations', 'mode points', 'replicates')}),
    # `cause` names the refusal by the code of its first note, one token per code a
    # refusal carries (`app.result_notes._LEFT_OUT_CAUSES`); a run left out for a failure
    # that names none carries no cause.
    'run_left_out': NoteSpec(
        'record', '{run} left out, inputs refused: {cause}', '3.5.3',
        example={'run': 'PVP-1 90°', 'cause': 'too few channels in the window'},
        tokens={'cause': ('too few channels in the window', 'baseline ≥ the windowed peak',
                          'residual weight not formed', 'no viscosity')}),
    'region': NoteSpec(
        'record', 'Correlogram-tab region (s): window {window}; baseline {baseline}', '3.5.3',
        example={'window': (1e-06, 0.1), 'baseline': (0.5, 1.0)},
        precision={'window': 6, 'baseline': 6},
        tokens={'window': ('not set', '—'), 'baseline': ('not placed',)}),
    # A precondition of the ± (independent points) that fails no flag: the fit runs on the
    # user's choice and its value stands; the ± treats an average and its sources as
    # independent, which they are not.
    'average_beside_sources': NoteSpec(
        'precondition', '{average} pooled beside its sources {sources}, not independent',
        '6', example={'average': 'PVP-1 avg', 'sources': ('PVP-1a', 'PVP-1b')}),
    'average_replaces_sources': NoteSpec(
        'record', '{average} stands in for {sources}, left out', '3.5.3',
        example={'average': 'PVP-1 avg', 'sources': ('PVP-1a', 'PVP-1b')}),
    'not_diffusive': NoteSpec(
        'precondition', 'Γ not linear through the origin in q² (R² {r2}, intercept {rel})',
        '3.5.1', fails=('is_diffusive',), example={'r2': 0.95, 'rel': 0.15}),
    # -- DLS: the mode map, mode families and the amplitude report
    'q_unusable': NoteSpec(
        'precondition', 'scattering vector q not usable (q² = {q2} m⁻²)', '1.2',
        fails=('success',), example={'q2': 0.0}),
    'fit_refused_inputs': NoteSpec(
        'precondition', 'the {fit} fit refused its inputs', '3.5.4',
        fails=('success',), example={'fit': 'CONTIN'},
        tokens={'fit': ('double-exponential', 'CONTIN', 'NNLS')}),
    'no_modes': NoteSpec(
        'precondition', 'no mode with a usable decay rate', '3.5.4', fails=('success',)),
    'empty_manual_interval': NoteSpec(
        'record', '{n} manual interval(s) captured no weight, no mode', '3.5.4',
        example={'n': 1}),
    'manual_bounds': NoteSpec(
        'record', 'manual integration bounds set the mode amplitudes', '4.2'),
    'mixed_extraction_methods': NoteSpec(
        'record', 'modes picked from {methods}', '3.5.4',
        example={'methods': ('CONTIN', 'double-exponential')},
        tokens={'methods': ('double-exponential', 'CONTIN', 'NNLS')}),
    'se_withheld_same_run': NoteSpec(
        'record', 'points share a run; regression SE withheld', '4.2.2'),
    'low_amplitude_points': NoteSpec(
        'record', '{n} of {total} mode points below the {floor} amplitude floor', '3.5.4',
        example={'n': 2, 'total': 7, 'floor': 0.02}),
    # Failed preconditions of the family's D and Rh (a point undecayed by its rate's Γ·Δτ
    # or, from an inversion, by its correlogram's own decay: `ModePoint.decay_incomplete`),
    # each failing `rh_reliable` on a class that carries it (`NoteSpec.fails`). The
    # sub-floor count beside them stays a record (guide §3.5.4).
    'undecayed_points': NoteSpec(
        'precondition', '{n} of {total} mode points undecayed in their window', '3.1.4',
        fails=('rh_reliable',), example={'n': 2, 'total': 7}),
    'grid_end_points': NoteSpec(
        'precondition', "{n} of {total} mode points at the Rh grid's end", '3.4.1',
        fails=('rh_reliable',), example={'n': 2, 'total': 7}),
    'q4_not_fitted': NoteSpec(
        'precondition', '{n} distinct angles, fewer than the {need} the q⁴ form needs',
        '3.5.4', fails=('q4_well_conditioned',), example={'n': 2, 'need': 3}),
    'picks_unmatched': NoteSpec(
        'record', 'no extracted mode for {n} pick(s)', '3.5.4', example={'n': 1}),
    'family_not_recorded': NoteSpec(
        'record', 'family {label} matched no map point, not recorded', '3.5.4',
        example={'label': 'slow'}),
    'removed_run_modes_dropped': NoteSpec(
        'record', '{n} mode(s) of removed measurement(s) dropped from the map', '3.5.4',
        example={'n': 2}),
    'multiple_points_at_angle': NoteSpec(
        'record', '{family} holds several points at {angles} deg, skipped there', '3.5.4',
        example={'family': 'slow', 'angles': (60.0, 90.0)},
        tokens={'family': ('fast', 'slow')}),
    'angles_in_one_family': NoteSpec(
        'record', '{n} angle(s) in one family only, skipped', '3.5.4', example={'n': 2}),
    'mixed_runs_at_angle': NoteSpec(
        'record', 'at {angles} deg the modes come from different runs, not combined',
        '3.5.4', example={'angles': (45.0,)}),
    # The flatness check is a diagnostic that gates a note, never a verdict a result
    # depends on (guide §3.5.4): records, and `fast_flat` a label.
    'too_few_shared_angles': NoteSpec(
        'record', '{n} shared angle(s), fewer than the {need} flatness needs', '3.5.4',
        example={'n': 1, 'need': 2}),
    'ratio_withheld': NoteSpec(
        'record', '{n} angle(s) with no usable slow amplitude; A_f/A_s withheld', '3.5.4',
        example={'n': 1}),
    'fast_amplitude_not_flat': NoteSpec(
        'record', 'A_f spans {span} of its mean across angles > {limit}', '3.5.4',
        example={'span': 0.35, 'limit': 0.2}),
    # -- DLS: replicate averaging
    'temperature_span': NoteSpec(
        'record', 'temperatures span {span} K', '6.4',
        example={'lo': 298.12, 'hi': 298.31, 'span': 0.19}),
    'averaged_from': NoteSpec(
        'record', 'averaged from {n} measurements: {sources}', '6.4',
        example={'n': 3, 'sources': ('PVP-1a', 'PVP-1b', 'PVP-1c')}),
    'peak_count_varies': NoteSpec(
        'precondition', 'replicates resolved {lo} to {hi} peaks, aligned by position', '6.4',
        example={'lo': 1, 'hi': 2}),
    # A written average's flagged fits, counted out of those averaged; its Table-2 row
    # carries their causes after it (`app.result_notes._replicate_verdict_notes`).
    'replicates_flagged': NoteSpec(
        'precondition', '{k} of {n} averaged fits flagged', '6.4', fails=('rh_reliable',),
        example={'k': 2, 'n': 3}),
    # `counts` holds the number of fits under each of `forms`, in the same order.
    'mixed_weightings': NoteSpec(
        'record', 'replicate fits ran under {forms} ({counts} fits)', '3.4.3',
        example={'forms': ('biganzoli-full', 'g1-squared'), 'counts': (2, 1)},
        tokens={'forms': ('biganzoli-full', 'biganzoli-classical', 'g1-squared')}),
    # -- DDLS and the shape models
    'dr_nonpositive': NoteSpec(
        'precondition', 'Γ_VH ≤ Γ_VV at {n} angle(s): D_r ≤ {limit}', '3.6.3',
        fails=('d_r_reliable',), example={'n': 1, 'limit': 0.0}),
    'vh_rate_missing': NoteSpec(
        'precondition', '{n_missing} of {n} angles without a VH decay rate', '3.6.1',
        fails=('d_r_reliable',), example={'n_missing': 1, 'n': 4}),
    # D_r's inputs, `inputs_flagged_decay`'s rule over both of its rates: the VV and VH
    # replicates whose own fit carries a precondition bearing on its decay rate
    # (`analysis.depolarization.analyze_ddls`).
    'dr_inputs_flagged_decay': NoteSpec(
        'precondition', '{k} of {n} VV and VH replicates carry a flagged decay rate',
        '3.6.2', fails=('d_r_reliable',), example={'k': 1, 'n': 12}),
    'ql_above_limit': NoteSpec(
        'precondition', 'qL {ql} ≥ {limit}, the single-exponential limit', '3.6.2',
        fails=('single_exponential_valid',), example={'ql': 3.4, 'limit': 3.0}),
    'no_shape_solution': NoteSpec(
        'precondition', 'no {model} reproduces both D_t and D_r', '3.6.7',
        fails=('converged', 'in_valid_range'), example={'model': 'rigid rod'},
        tokens={'model': ('rigid rod', 'ellipsoid of revolution')}),
    'aspect_ratio_out_of_range': NoteSpec(
        'precondition', 'p {p} outside {lo}–{hi}, the {model} range', '3.6.7',
        fails=('in_valid_range',),
        example={'p': 41.0, 'lo': 2.0, 'hi': 30.0, 'model': 'rod'},
        tokens={'model': ('rod', 'ellipsoid')}),
    'mc_census': NoteSpec(
        'record', '{used} of {n} Monte-Carlo draws solvable, seed {seed}', '3.6.7',
        example={'used': 1987, 'n': 2000, 'seed': 12345}),
    # -- depolarization ratio
    'rho_v_out_of_range': NoteSpec(
        'precondition', 'ρv {rho_v} outside {lo} ≤ ρv < {hi}', '1.6.1',
        fails=('physically_valid',), example={'rho_v': 0.81, 'lo': 0.0, 'hi': 0.75}),
    # -- weighted size distributions
    'bins_outside_model': NoteSpec(
        'precondition', '{n_bad} of {n} size bins outside the {model} range', '3.7',
        fails=('all_valid',), example={'n_bad': 12, 'n': 200, 'model': 'RDG sphere'},
        tokens={'model': ('Rayleigh point', 'RDG sphere', 'Mie sphere')}),
    'mie_low_contrast': NoteSpec(
        'record', '|m−1| {m1} ≤ {limit}: RDG and Mie agree here', '4.10.1',
        example={'m1': 0.04, 'limit': 0.1}),
    # -- SLS: the line fits (Debye, Guinier, Zimm/Berry, the ladder, calibration-free A2)
    'ill_conditioned': NoteSpec(
        'precondition', 'design condition number {cond} > {limit}', '6.6',
        fails=('well_conditioned', 'q4_well_conditioned', 'mw_reliable', 'a2_reliable',
               'two_a2_mw_reliable', 'kd_reliable', 'rg_reliable'),
        example={'cond': 3.2e12, 'limit': 1e12}),
    'subnormal_divisor': NoteSpec(
        'precondition', 'Kc/ΔR {value} below the smallest normal float {limit}', '6.6',
        fails=('well_conditioned', 'mw_reliable'),
        example={'value': 1e-310, 'limit': 2.2e-308}),
    # One row per withheld quantity, each failing its own flag only, since one quantity
    # can overflow while another of the same fit stays finite.
    'mw_not_finite': NoteSpec(
        'precondition', 'Mw not finite; value withheld', '6.6', fails=('mw_reliable',)),
    'a2_not_finite': NoteSpec(
        'precondition', 'A₂ not finite; value withheld', '6.6', fails=('a2_reliable',)),
    'two_a2_mw_not_finite': NoteSpec(
        'precondition', '2·A₂·Mw not finite; value withheld', '6.6',
        fails=('two_a2_mw_reliable',)),
    'kc_dr_nonpositive': NoteSpec(
        'precondition', 'Kc/ΔR {value} ≤ {limit} at the one angle', '4.3',
        fails=('well_conditioned', 'mw_reliable'), example={'value': -2.4e-07, 'limit': 0.0}),
    'intercept_nonpositive': NoteSpec(
        'precondition', '{quantity} {value} ≤ {limit}', '4.3.1',
        fails=('mw_reliable', 'a2_reliable', 'two_a2_mw_reliable'),
        example={'quantity': 'intercept', 'value': -3.1e-07, 'limit': 0.0},
        tokens={'quantity': ('intercept',)}),
    'no_rg': NoteSpec(
        'precondition', 'no Rg: {quantity} {value} {unit}, not {required}', '4.6',
        fails=('rg_reliable', 'guinier_valid', 'qrg_regime_valid'),
        example={'quantity': 'slope', 'value': 12.4, 'unit': 'nm²', 'required': 'negative'},
        tokens={'quantity': ('slope', 'slope/intercept', 'intercept', '1/A(0)'),
                'unit': ('nm²', 'mol/g', '(mol/g)^½', '(1/amplitude)'),
                'required': ('negative', 'positive')}),
    'qrg_above_limit': NoteSpec(
        'precondition', 'qRg(max) {qrg} > {limit}, the {regime} limit', '4.6',
        fails=('guinier_valid', 'qrg_regime_valid', 'rg_reliable', 'mw_reliable'),
        example={'qrg': 1.37, 'limit': 1.0, 'regime': 'Debye'},
        tokens={'regime': ('Debye', 'Zimm', 'amplitude-Guinier')}),
    'qrg_above_guinier_limit': NoteSpec(
        'precondition', 'qRg(max) {qrg} > {limit}, the Guinier limit', '4.6',
        fails=('guinier_valid', 'rg_reliable', 'mw_reliable'),
        example={'qrg': 1.62, 'limit': 1.3}),
    'berry_cross_term_dropped': NoteSpec(
        'record', 'Berry q²·c cross term unresolved ({reason}); fitted without it', '4.4',
        example={'reason': 'rank-deficient'},
        tokens={'reason': ('rank-deficient', 'ill-conditioned')}),
    'grid_level_single_angle': NoteSpec(
        'record', 'c = {levels} g/mL: one usable angle, its ordinate the q→0 intercept',
        '4.4', example={'levels': (0.001,)}),
    'grid_level_single_concentration': NoteSpec(
        'record', 'q² = {levels} nm⁻²: one usable c, its ordinate the c→0 intercept',
        '4.4', example={'levels': (0.000412,)}),
    # A fit may count its non-finite points with the non-positive ones or apart from them,
    # so the clause says "non-positive" only, and a fit that counts both says so in its
    # ordinate token ('or non-finite amplitude').
    'nonpositive_points_dropped': NoteSpec(
        'record', '{n} point(s) with non-positive {ordinate} dropped', '4',
        example={'n': 2, 'ordinate': 'Kc/ΔR'},
        tokens={'ordinate': ('Kc/ΔR', 'excess Rayleigh ratio', 'or non-finite amplitude')}),
    'runs_averaged': NoteSpec(
        'record', 'one point averaged from {n} runs at {angle} deg', '4.1',
        example={'n': 3, 'angle': 90.0}),
    'averaged_point_sem': NoteSpec(
        'record', "SEM {sem} cps at {angle} deg, from the repeats' spread", '4.1',
        example={'sem': 1520.0, 'angle': 90.0}),
    # A flagless precondition (ui_style_guide §5): repeats at one angle that disagree by more
    # than the tolerance are averaged as they are; the spread is shown on the Derived
    # Intensity tab and kept in the derived point's record, which the export Comments carry.
    'rate_spread': NoteSpec(
        'precondition', '{runs} at {angle} deg: count-rate spread {spread} > {limit}', '4.1',
        example={'runs': ('PVP-1a', 'PVP-1b', 'PVP-1c'), 'angle': 90.0, 'spread': 0.12,
                 'limit': 0.05}),
    'derived_cps_unit': NoteSpec(
        'record', 'derived intensities in cps; k_c cancels the unit only if all are cps',
        '4.1'),
    'runs_differ': NoteSpec(
        'record', "the runs differ in {field}; the first run's {value} {unit} is used", '4.1',
        example={'field': 'temperature', 'value': 298.15, 'unit': 'K'},
        precision={'value': 6},
        tokens={'field': ('polymer name', 'solvent name', 'temperature', 'viscosity',
                          'wavelength', 'solvent refractive index',
                          'molecular-weight fraction', 'analyzer geometry'),
                'unit': ('K', 'Pa·s', 'nm')}),
    # -- SLS: the Sedlák partition and the per-mode analyses
    'modes_not_separated': NoteSpec(
        'precondition', 'adjacent mode rates a factor {ratio} apart < {limit}', '4.2',
        fails=('well_separated',), example={'ratio': 3.1, 'limit': 5.0}),
    'closure_deficit': NoteSpec(
        'record', 'up to {deficit} of the excess intensity in unrecorded modes', '4.2',
        example={'deficit': 0.12}),
    'closure_excess': NoteSpec(
        'record', 'the picked fractions sum above {limit} by up to {excess}', '4.2',
        example={'limit': 1.0, 'excess': 0.08}),
    'mode_amplitude_missing': NoteSpec(
        'record', 'mode {label}: {n} of {total} angles without an amplitude, excluded', '4.2',
        example={'label': 'slow', 'n': 1, 'total': 7}),
    # Failed preconditions of one mode's partitioned intensities (Sedlák's amplitude floor,
    # the Γ·Δτ test), counted per mode over the points its share was built on and minted
    # by the fits on that mode's series, about its values, its label a names value.
    'points_low_amplitude_used': NoteSpec(
        'precondition', 'mode {label}: {n} of {total} points below their amplitude floor',
        '3.5.4', fails=('mw_reliable', 'rg_reliable', 'a2_reliable'),
        example={'label': 'slow', 'n': 2, 'total': 7}),
    'points_undecayed_used': NoteSpec(
        'precondition', 'mode {label}: {n} of {total} points undecayed in their window',
        '3.1.4', fails=('mw_reliable', 'rg_reliable', 'a2_reliable'),
        example={'label': 'slow', 'n': 2, 'total': 7}),
    # A decayed mode whose points share their decomposition with an undecayed mode: its
    # own points decayed, so the per-mode clause above would be false of it. A point is
    # an angle at one concentration, since a fit over several sums the counts over them.
    'partition_angle_spoiled': NoteSpec(
        'precondition', '{n} of {total} points share a decomposition with an undecayed mode',
        '4.2', fails=('mw_reliable', 'rg_reliable', 'a2_reliable'),
        example={'n': 2, 'total': 7}),
    # A mode whose lobe sits on the Rh grid's end: the grid does not contain its population,
    # so its amplitude, and every fraction sharing its decomposition, is not what the
    # partition fits; the per-angle rule the undecayed points follow (the owner's ruling,
    # 2026-10-17; the rule's home is `analysis/sls.py::ModePointCounts`).
    'points_grid_end_used': NoteSpec(
        'precondition', "mode {label}: {n} of {total} points at the Rh grid's end", '3.4.1',
        fails=('mw_reliable', 'rg_reliable', 'a2_reliable'),
        example={'label': 'slow', 'n': 2, 'total': 7}),
    'partition_angle_grid_end': NoteSpec(
        'precondition',
        "{n} of {total} points share a decomposition with a mode at the Rh grid's end",
        '4.2', fails=('mw_reliable', 'rg_reliable', 'a2_reliable'),
        example={'n': 2, 'total': 7}),
    'points_without_concentration': NoteSpec(
        'record', '{n} recorded point(s) without a committed concentration', '4.2',
        example={'n': 2}),
    'repeat_series_not_partitioned': NoteSpec(
        'record', 'a repeat SLS series at this concentration is not partitioned', '4.2'),
    # -- SLS: the form-factor fits
    'ff_not_converged': NoteSpec(
        'precondition', 'form-factor fit did not converge', '4.8',
        fails=('converged', 'mw_reliable', 'a2_reliable', 'rg_reliable')),
    'jacobian_ill_conditioned': NoteSpec(
        'precondition', 'fit Jacobian condition number {cond} > {limit}', '6.6',
        fails=('well_conditioned', 'mw_reliable', 'a2_reliable'),
        example={'cond': 3.2e12, 'limit': 1e12}),
    'sub_molecular_size': NoteSpec(
        'precondition', '{quantity} {size} nm < {floor} nm, the size floor', '4.8.2',
        fails=('form_factor_valid', 'mw_reliable', 'a2_reliable', 'rg_reliable'),
        example={'quantity': 'Rg', 'size': 0.09, 'floor': 0.155},
        tokens={'quantity': ('Rg', 'outer radius')}),
    'past_first_minimum': NoteSpec(
        'precondition', 'qRg(max) {qrg} > {limit}, past the first form-factor minimum',
        '4.8.1', fails=('form_factor_valid', 'mw_reliable', 'a2_reliable'),
        example={'qrg': 3.9, 'limit': 3.48}),
    'deep_minimum': NoteSpec(
        'precondition', 'min P(q) over the angles < {limit}', '4.10.1',
        fails=('form_factor_valid', 'mw_reliable', 'a2_reliable'), example={'limit': 0.05}),
    'no_ff_solution': NoteSpec(
        'precondition', 'no {model} reproduces the angular P(q)', '4.9.1',
        fails=('converged', 'mw_reliable', 'a2_reliable', 'rg_reliable',
               'axial_ratio_reliable', 'well_conditioned'),
        example={'model': 'ellipsoid of revolution'},
        tokens={'model': ('ellipsoid of revolution', 'core-shell sphere')}),
    'axial_ratio_unresolved': NoteSpec(
        'precondition', 'axial ratio unresolved at qRg(max) {qrg}', '4.9.3',
        fails=('axial_ratio_reliable',), example={'qrg': 0.6}),
    # Resolved but imprecise: the axial ratio is identified, its relative SE past the limit.
    'axial_ratio_imprecise': NoteSpec(
        'precondition', 'axial ratio relative SE {rel_se} > {limit}, the precision limit',
        '4.9.3', fails=('axial_ratio_reliable',), example={'rel_se': 0.62, 'limit': 0.5}),
    'rg_undefined': NoteSpec(
        'precondition', 'contrast-weighted Rg undefined for this contrast profile', '4.11.2',
        fails=('rg_reliable',)),
    # -- cross-sample
    'rho_sample_mismatch': NoteSpec(
        'record', 'samples differ: Rg {rg_key} K, Rh {rh_key} K', '5.1',
        example={'rg_key': ('PVP', 'water', 298.15), 'rh_key': ('PVP', 'water', 308.15)},
        precision={'rg_key': 6, 'rh_key': 6}),
    'linear_reference_law': NoteSpec(
        'record', 'fitted {quantity}–Mw law over {n} linear points, Mw {lo} to {hi}', '5.3',
        example={'quantity': 'Rg', 'n': 5, 'lo': 1e5, 'hi': 2e6},
        tokens={'quantity': ('Rg', 'Rh')}),
    # -- count-rate trace diagnostics
    'null_not_white': NoteSpec(
        'precondition', 'trace correlated; the false-alarm null assumes white noise', '2.1.8'),
    # -- parser load-time notices. Their facts are file-format facts, so each points into the
    #    User-Manual's format section (guide='manual'; doc_style_guide §8.4 keeps such facts
    #    out of the Theory guide; the owner's word, 2026-10-16): §3 for the box every loader
    #    shows, §4.2 and §5.x for one format's own notice, §5 for a notice shared by the
    #    instrument formats.
    'parser.rows_skipped': NoteSpec(
        'record', '{n_skipped} of {n} data row(s) unreadable, skipped; {n_loaded} kept',
        '3', example={'n_skipped': 2, 'n': 182, 'n_loaded': 180}, guide='manual'),
    'parser.block_truncated': NoteSpec(
        'record', '{block} block ends at line {line}; {n_read} row(s) read', '5.7',
        example={'block': 'Correlation', 'line': 214, 'n_read': 180}, guide='manual',
        tokens={'block': ('Correlation', 'Count Rate')}),
    'parser.empty_columns': NoteSpec(
        'record', '{n_empty} of {n} record column(s) with no number, not loaded: '
        '{columns}', '5.8', example={'n_empty': 1, 'n': 6, 'columns': ('Record 2',)},
        guide='manual'),
    'parser.negative_intensity': NoteSpec(
        'record', '{n} negative intensity value(s), stored as read', '4.2',
        example={'n': 2}, guide='manual'),
    'parser.duration_withheld': NoteSpec(
        'record', 'header run duration {duration} s ≤ last live lag {last_lag} s, not carried',
        '5', example={'duration': 30.0, 'last_lag': 41.9}, guide='manual'),
    'parser.sample_times_unread': NoteSpec(
        'record', 'per-channel sample times not read from the header', '5.2', guide='manual'),
    # -- which premise of the correlator-section header failed (beside the record above)
    'parser.section_item_invalid': NoteSpec(
        'record', 'header item {item}, a section {quantity}, has no usable value', '5.2',
        example={'item': 29, 'quantity': 'channel count'}, guide='manual',
        tokens={'quantity': ('channel count', 'sampling time')}),
    'parser.sections_unsummed': NoteSpec(
        'record', 'correlator sections sum to {n} channels, the data hold {n_data}',
        '5.2', example={'n': 251, 'n_data': 250}, guide='manual'),
    'parser.hs_delay_unknown': NoteSpec(
        'record', 'high-speed channel {channel}: its delay is no fixed high-speed delay',
        '5.2', example={'channel': 1}, guide='manual'),
    'parser.delay_not_multiple': NoteSpec(
        'record', 'channel {channel}: delay not a multiple of its {dt} µs sample time',
        '5.2', example={'channel': 196, 'dt': 40000.0}, guide='manual'),
    'parser.repeated_lags_dropped': NoteSpec(
        'record', '{n_dropped} of {n} lags repeat, dropped; first copies kept, {n_kept} left',
        '5.6', example={'n_dropped': 7, 'n': 199, 'n_kept': 192},
        guide='manual'),
    'parser.no_metadata': NoteSpec(
        'record', 'the file carries no measurement parameters', '5.6', guide='manual'),
    # -- a sentence from a session file older than the codes
    'legacy': NoteSpec('legacy', '{text}', example={'text': 'a note saved by an older build'}),
}

# The per-point preconditions bearing on a decay rate (this comment their normative home):
# all but `no_viscosity`, as Γ and D need none, and no record, so not the fallback weight
# (recommendations the owner accepted, 2026-10-16). The size range is in though its test
# reads a viscosity: past the ceiling a particle sediments (`cumulants.CUMULANT_MAX_RH_NM`),
# below the floor the fit chased an artifact into Γ, so Γ is in doubt; a point with no
# viscosity is not judged by it (a recommendation the owner accepted, 2026-10-17). A pooled
# line's input flagged by one is `inputs_flagged_decay`, which fails its kD as well as its
# Rh.
DECAY_BEARING_CODES: frozenset[str] = frozenset({
    'decay_incomplete', 'pdi_above_limit', 'rh_above_ceiling', 'rh_below_floor',
    'gamma_nonpositive', 'retained_range_too_short',
    'too_few_live_channels', 'fit_not_converged', 'solution_below_floor',
    'grid_end', 'solution_nonphysical', 'coefficients_not_finite',
    'baseline_bound_unformable', 'first_pass_failed', 'weight_inputs_refused',
    'window_too_few_channels', 'baseline_above_data'})
