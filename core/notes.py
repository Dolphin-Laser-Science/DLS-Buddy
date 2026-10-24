"""The notes data model: what a note is, as data. The normative home of the model.

A **note** is one structured fact about one result, stated by whoever measured it: a
registry ``code``, the result attributes it is ``about`` (``()`` for the whole fit) and the
``values`` its clause prints. The engine mints ``Note(code, about, values)`` and never a
sentence, so no display text lives in ``analysis/`` (architecture invariant 1); the words
are the registry's, one row per code in ``core/note_registry.py``, and the remedies the
GUI's, one row per code in ``gui/note_text.py``. Where a note goes and how its clause is
worded is the policy's, ui_style_guide §5; this module is the shape.

Four kinds, each a :class:`NoteSpec` ``kind``:

* ``precondition`` -- a method's precondition failed on this run (invariant 4). The spec's
  ``fails`` names the flag FIELDS that failure sets False, several where one failure fails
  several. The relation is one way: a precondition note implies each flag it names is False
  on any class that carries it; a flag can be False with no note (the silent conditions,
  pinned by the family gate and allowed only to shrink), so a flag is never derived from
  its notes.
* ``record`` -- how the fit was obtained (a fallback weight, a dead channel, a dropped
  point). It reaches the export and the Summary tooltip, never the analysis tab.
* ``qualifier`` -- what the number is (apparent, a statistical ±, an estimator, a
  weighting). Never minted: :func:`qualifiers` reads it off the result's typed fields.
* ``legacy`` -- a sentence from a session file older than the codes, carried as
  ``Note('legacy', (), {'text': ...})`` so it is shown rather than dropped.

``values`` holds JSON scalars (float, int, str, bool, None) or tuples of them, because a
note travels into session files; a numpy scalar is stored as its Python value. The
mapping is read-only and hashable, so a result that carries notes hashes as it did when
its notes were strings.

A result holds every note it has, of every kind, in one ``notes`` tuple of :class:`Note`;
:func:`has_note` and :func:`find_notes` read it by code and attribute, never by wording.
"""
from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, FrozenSet, Iterator, Optional, Tuple, Union

import numpy as np

KINDS: Tuple[str, ...] = ('precondition', 'record', 'qualifier', 'legacy')
#: The guide a registry row's section points into: a 'theory' row prints as a bare "(§N)", a
#: 'manual' row as "(User-Manual §N)" (the load notices, whose facts are file formats).
GUIDES: Tuple[str, ...] = ('theory', 'manual')
# The value keys `render` reserves for a relayed note's origin (its docstring).
WHERE, WHERE_UNIT = 'where', 'where_unit'
# The units a relayed note's coordinates are written in: a word `render` prints beside the
# `where` numbers, so it is spelled here as a clause's tokens are spelled in its row.
WHERE_UNITS: FrozenSet[str] = frozenset({'deg', 'g/mL'})
# The words a relayed note's `where` may carry besides measured names: which part of a fit
# the note is about (a double exponential's modes on the Summary row).
WHERE_WORDS: FrozenSet[str] = frozenset({'fast mode', 'slow mode', 'both modes'})
# A line over more sources than `NAME_MAX` counts them instead of naming them ("12 of 40
# runs"): its one template, and the nouns a surface may count.
WHERE_COUNT = '{n} of {total} {what}'
WHERE_NOUNS: FrozenSet[str] = frozenset({'runs', 'fits', 'points', 'angles', 'models',
                                         'samples', 'solutions', 'concentrations'})
# The value keys that carry the names of measured things (a run, a sample, a file, a mode
# family's label, a pasted record's column), which are data and need no token in their row
# (ui_style_guide §5, the `values` rule); a word the program itself writes under one, as under
# every other key, is a token its row spells (`WHERE_WORDS` for `where`).
NAMES_KEYS: FrozenSet[str] = frozenset({'run', 'runs', 'sources', 'average', 'label', 'family',
                                        'file', 'columns', WHERE})
# The names keys a mint site fills with a tuple, and the noun each counts: where the GUI draws
# a note (`render(..., counted=True)`) a tuple of more than `NAME_MAX` names prints inside its
# clause as "6 runs"; a replicate average's `sources` are its DLS runs.
NAMES_NOUNS: Mapping[str, str] = {'runs': 'runs', 'sources': 'runs', 'columns': 'columns'}
# The value keys a stored note can lack, because its row gained the placeholder after the
# note was written: a clause ends `: {key}`, and `render` drops that ending from the clause
# of a note without the key (the registry gate holds every such row to that shape).
OPTIONAL_KEYS: FrozenSet[str] = frozenset({'cause'})

# What `qualifiers` can mint: these codes, plus WEIGHTING_PREFIX + a recorded weighting.
QUALIFIER_CODES: FrozenSet[str] = frozenset({
    'apparent', 'mode_resolved', 'mode_family', 'cumulant_average', 'distribution_weighted',
    'model_weighted_view', 'se_statistical', 'se_regression', 'se_ols',
    'se_excludes_inversion', 'rg_below_resolution', 'degenerate', 'model_dependent',
    'degenerate_pick', 'includes_model_dependent', 'includes_degenerate_pick',
    'branching_model_dependent', 'berry_past_zimm_limit'})
# The scale gates (ui_style_guide §5): preconditions of Mw and absolute A2 that the GUI and
# the export compose from the typed fields, so a relay of a fit's preconditions leaves
# them out.
SCALE_CODES: FrozenSet[str] = frozenset({'uncalibrated', 'no_dn_dc', 'blank_unmeasured',
                                         'no_standard_index'})
WEIGHTING_PREFIX = 'weighting.'
WEIGHTING_FIELDS: Tuple[str, ...] = ('weighting', 'cumulant_weighting')
# The average a DLS result reports (invariant 7), keyed by the class's qualified name so
# this module imports nothing from analysis/; the registry gate resolves every name.
CUMULANT_AVERAGE_CLASSES: FrozenSet[str] = frozenset({
    'analysis.dls.cumulants.CumulantResult'})
DISTRIBUTION_WEIGHTED_CLASSES: FrozenSet[str] = frozenset({
    'analysis.dls.distributions.DistributionResult',
    'analysis.dls.distributions.ContinResult',
    'analysis.optical_model.WeightedDistributionResult'})
# A DLS mode family's value is one mode's, picked across angles (invariant 7); not the
# Sedlak partition's `mode_resolved`, which is one mode's share of the intensity.
MODE_FAMILY_CLASSES: FrozenSet[str] = frozenset({'analysis.dls.modes.ModeFamilyFitResult'})
# A stored sample result says per quantity what its value is: `<q>_apparent`,
# `<q>_model_dependent` and `<q>_degenerate`, each about the value field of its quantity
# prefix. The registry gate finds every such boolean and asks its prefix to be here.
VALUE_FIELDS: Mapping[str, str] = {'rh': 'rh_nm', 'mw': 'mw_g_per_mol', 'rg': 'rg_nm',
                                   'a2': 'a2_mol_mL_per_g2'}
PER_QUANTITY_FLAGS: Tuple[str, ...] = ('_apparent', '_model_dependent', '_degenerate')

# The wording rule's mechanical half (ui_style_guide.md §5), read by the registry gate.
CLAUSE_MAX = 80   # characters of a clause rendered for the screen, its section included
REMEDY_MAX = 90   # characters of a remedy in gui/note_text.py
BANNED_WORDS: Tuple[str, ...] = ('may', 'might', 'typically', 'generally', 'consider',
                                 'note that', 'please', 'it is possible', 'in principle')
# The kind is the verdict, so a clause never says it.
VERDICT_WORDS: Tuple[str, ...] = ('unreliable', 'reliable', 'invalid', 'warning', 'caution',
                                  'suspect', 'untrustworthy')
# A clause states a fact, never what to do: none of these as the first word of the clause
# or of a part after ';', ':', ' -- ' or ' — '.
REMEDY_VERBS: Tuple[str, ...] = ('measure', 'widen', 'prefer', 'refit', 'enter', 'check', 'try',
                                 'use', 'drop', 'select', 'pick', 'lower', 'raise', 'increase',
                                 'decrease', 'remove', 'rerun', 'set', 'supply', 'load', 'run')


def _plain(value: Any) -> Any:
    """A note value as a JSON-ready scalar: a numpy scalar becomes its Python value, a list
    or tuple a tuple of plain values."""
    if isinstance(value, (list, tuple)):
        return tuple(_plain(v) for v in value)
    item = getattr(value, 'item', None)
    if callable(item) and not isinstance(value, (str, bytes)):
        try:
            return item()
        except (TypeError, ValueError):
            return value
    return value


class NoteValues(Mapping):
    """A read-only, hashable mapping of a note's values (``Note.values``)."""

    __slots__ = ('_d',)

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        self._d = {str(k): _plain(v) for k, v in dict(values or {}).items()}

    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._d.items(), key=lambda kv: kv[0])))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and self._d == dict(other)

    def __repr__(self) -> str:
        return repr(self._d)


@dataclass(frozen=True)
class Note:
    """One fact about one result: a registry code, what it is about, and its values."""

    code: str
    about: Tuple[str, ...] = ()
    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        about = (self.about,) if isinstance(self.about, str) else tuple(self.about)
        object.__setattr__(self, 'about', about)
        object.__setattr__(self, 'values', NoteValues(self.values))


@dataclass(frozen=True)
class NoteSpec:
    """One registry row: the code's kind, its one-clause template, its guide section, the
    flag fields a precondition fails, and representative values the gate renders it at.

    ``label`` is a qualifier's word on a row or column label (``vocabulary``), '' where
    the clause itself is the word. ``precision`` gives a placeholder its own significant
    figures where the display's would print two different values alike: a recorded input,
    a temperature, a key that names a sample."""

    kind: str
    clause: str
    section: str = ''
    fails: Tuple[str, ...] = ()
    example: Mapping[str, Any] = field(default_factory=dict)
    label: str = ''
    precision: Mapping[str, int] = field(default_factory=dict)
    #: Which guide ``section`` points into: the Theory guide for every analysis note, the
    #: User-Manual for the parsers' load-time notices, whose facts are file-format facts the
    #: Manual's format sections hold and the Theory guide never does (doc_style_guide §8.4;
    #: the owner's word, 2026-10-16).
    guide: str = 'theory'
    #: The string values a placeholder may take, by key: the only words a mint site may write
    #: into ``values`` besides numbers and a names key's measured names (ui_style_guide §5,
    #: the ``values`` rule), so every word a clause prints is spelled in its row.
    tokens: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f'NoteSpec kind {self.kind!r} is not one of {KINDS}')
        if self.guide not in GUIDES:
            raise ValueError(f'NoteSpec guide {self.guide!r} is not one of {GUIDES}')
        object.__setattr__(self, 'fails', tuple(self.fails))
        object.__setattr__(self, 'example', NoteValues(self.example))
        object.__setattr__(self, 'precision', NoteValues(self.precision))
        object.__setattr__(self, 'tokens', NoteValues(self.tokens))
        if self.fails and self.kind != 'precondition':
            raise ValueError(f'only a precondition fails a flag; a {self.kind} names {self.fails}')
        if self.label and self.kind != 'qualifier':
            raise ValueError(f'only a qualifier has a label; a {self.kind} names {self.label!r}')


def _shown(value: Any, sig: int) -> str:
    if isinstance(value, float):
        return format(value, f'.{sig}g')
    if isinstance(value, tuple):
        return ', '.join(_shown(v, sig) for v in value)
    return str(value)


def _registry(registry: Optional[Mapping[str, NoteSpec]]) -> Mapping[str, NoteSpec]:
    if registry is not None:
        return registry
    from core.note_registry import REGISTRY
    return REGISTRY


def _counted(key: str, value: Any) -> Any:
    """A names value as the GUI draws it: a tuple of more than `NAME_MAX` names under a key
    of `NAMES_NOUNS` as its count and noun, any other value as it is."""
    noun = NAMES_NOUNS.get(key)
    if noun and isinstance(value, tuple) and len(value) > NAME_MAX:
        return f'{len(value)} {noun}'
    return value


def _filled(template: str, note: 'Note', spec: NoteSpec, sig: int, *,
            counted: bool = False) -> str:
    """`template` with the note's values, each at its placeholder's precision; a key of
    `OPTIONAL_KEYS` the note lacks takes its `: {key}` ending off the template first.
    ``counted`` prints a long list of names as its count (`_counted`)."""
    for key in OPTIONAL_KEYS - set(note.values):
        template = template.replace(': {' + key + '}', '')
    text = template.format(**{k: _shown(_counted(k, v) if counted else v,
                                        spec.precision.get(k, sig))
                              for k, v in note.values.items()})
    return re.sub(r' {2,}', ' ', text)


def render(note: Note, *, sig: int = 3, surface: str = 'screen', counted: bool = False,
           registry: Optional[Mapping[str, NoteSpec]] = None) -> str:
    """The note's clause with its values at ``sig`` significant figures; ``'screen'``
    appends the section pointer, `` (§<section>)`` for a Theory-guide row and
    `` (User-Manual §<section>)`` for a Manual row; ``'export'`` does not.

    ``counted`` is for the GUI, whose every call passes it: on ``'screen'`` a tuple of more
    than ``NAME_MAX`` names under a key of ``NAMES_NOUNS`` prints inside its clause as its
    count and noun, ``6 runs at 90 deg: ...``, so a drawn line keeps the wording rule's length
    at any number of runs. Without it (a stored record) and on ``'export'`` every name
    prints, since the note is the session's and the file's record of what it holds.

    A float prints at ``sig`` significant figures, an int, a string or a bool as itself,
    a tuple as its members joined by ', '. ``registry`` defaults to the program's; the
    registry gate passes its own rows through the same code.

    A placeholder the spec's ``precision`` names prints at its own significant figures
    instead. A placeholder that renders empty (an optional unit) leaves no doubled space.
    A note lacking a key of ``OPTIONAL_KEYS`` (stored before its row gained the
    placeholder) prints its clause without the ``: {key}`` ending.

    Two value keys are reserved for a note relayed from the runs or points it was found
    on, and no clause names them: ``where`` (a label or a tuple of them, the runs' names or
    the points' coordinates) and ``where_unit`` (the coordinates' unit). A non-empty
    ``where`` prints in front of the clause, ``0.0002, 0.0006 g/mL: <clause>``, its
    numbers at six significant figures whatever ``sig`` is."""
    if surface not in ('screen', 'export'):
        raise ValueError(f"surface must be 'screen' or 'export', not {surface!r}")
    spec = _registry(registry)[note.code]
    text = _filled(spec.clause, note, spec, sig, counted=counted and surface == 'screen')
    where = note.values.get(WHERE)
    if where not in (None, '', ()):
        unit = note.values.get(WHERE_UNIT) or ''
        # Coordinates name the runs or points, so they print as recorded (six figures),
        # never rounded to the clause's precision, where two of them could read alike.
        text = f"{_shown(where, 6)}{' ' + unit if unit else ''}: {text}"
    if surface == 'screen' and spec.section:
        # A bare § is the Theory guide, which the click popup names in full (gui/notes_block.py);
        # a pointer into the Manual names the book on the line too, or the reader opens the
        # wrong one.
        text += (f' (§{spec.section})' if spec.guide == 'theory'
                 else f' (User-Manual §{spec.section})')
    return text


class RefusedInputs(ValueError):
    """A fit refused its inputs on a precondition it can name: ``notes`` are the
    precondition notes, each a registry code, so a caller relays the cause rather than a
    proxy ("refused its inputs"). A ``ValueError``, so a caller that catches one still
    catches it; its message is the notes rendered for an export, or ``message``."""

    def __init__(self, notes: Iterable[Note], message: str = '') -> None:
        self.notes: Tuple[Note, ...] = tuple(notes)
        super().__init__(message or '; '.join(render(n, surface='export') for n in self.notes))


def vocabulary(note: Note, *, sig: int = 3,
               registry: Optional[Mapping[str, NoteSpec]] = None) -> str:
    """A qualifier's word on a row or column label: its spec's ``label`` with its values,
    else its clause, never with a section (the label is vocabulary, the clause its
    tooltip)."""
    spec = _registry(registry)[note.code]
    return _filled(spec.label or spec.clause, note, spec, sig)


def _field_names(result: Any) -> Tuple[str, ...]:
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return tuple(f.name for f in dataclasses.fields(result))
    return tuple(getattr(result, '__dict__', {}))


def _is_true(value: Any) -> bool:
    """True for a Python or NumPy boolean True, never for a truthy non-boolean."""
    return isinstance(value, (bool, np.bool_)) and bool(value)


def qualifiers(result: Any, *,
               registry: Optional[Mapping[str, NoteSpec]] = None) -> Tuple[Note, ...]:
    """The qualifier notes read off a result's typed fields; no other code mints one.

    One rule per code, each reading only what the result carries:

    * ``apparent`` -- ``is_apparent`` is True, or ``mode_resolved`` is True, since every
      value on a mode's share is apparent whatever its extrapolation (guide §4.2.2); and,
      on a stored sample result, one about each value whose ``<q>_apparent`` flag is True
      (invariant 7);
    * ``model_dependent`` and ``degenerate_pick`` -- on a stored sample result, one about
      each value whose ``<q>_model_dependent`` or ``<q>_degenerate`` flag is True: a value
      that assumes a shape, and one resting on a user-selected solution among degenerate
      shape fits (R5.6); the first names ``<q>_model`` where the result carries it;
    * ``includes_model_dependent`` and ``includes_degenerate_pick`` -- ``any_model_dependent``
      or ``any_degenerate`` is True: a plot over several samples includes such a value;
    * ``branching_model_dependent`` -- ``branching_index`` is a finite float, which only a
      random-branching model reads from g: another model reads another index (guide §5.3.1);
    * ``mode_resolved`` -- ``mode_resolved`` is True: the value is one mode's share of the
      excess intensity, the Sedlák partition's (invariant 7);
    * ``mode_family`` -- the class is named in ``MODE_FAMILY_CLASSES``: one DLS mode's
      value, picked across angles (invariant 7);
    * ``cumulant_average`` or ``distribution_weighted`` -- the result's class is named in
      ``CUMULANT_AVERAGE_CLASSES`` or ``DISTRIBUTION_WEIGHTED_CLASSES``, or, for the
      cumulant average, its ``gamma_source`` is ``'cumulant'``: a pooled line on the
      cumulant decay rates (invariant 7);
    * ``model_weighted_view`` -- a non-empty ``optical_model_label`` beside a non-empty
      ``weighting``: a size distribution re-expressed under a sphere model, its label
      the one the result carries (this module imports nothing from ``analysis/``);
    * ``weighting.<value>`` -- a non-empty ``weighting`` or ``cumulant_weighting`` string
      (on the result, or on the ``distribution`` a CONTIN result wraps) whose code is a
      registry row; a value with no row (a request token, an optical-model
      weighting) is not a recorded residual weighting and mints nothing;
    * ``se_statistical`` -- the class carries ``calibrated`` and at least one ``*_se``
      field whose ± calibration and dn/dc can enter: a scale-bearing SLS result, whose ±
      excludes them (invariant 8). Not ``rh_se`` (a DLS size's, which a sample's stored
      result carries beside its SLS quantities) nor ``two_a2_mw_se`` (the calibration-free
      2·A2·Mw product's, on the ladder and the calibration-free fit alike); the
      calibration-free fit's ``a2_se`` is named, since its A2 divides the product by an Mw
      that carries both;
    * ``se_regression`` -- a reported ``*_se`` no calibration enters, which is statistical
      only (invariant 8): on a class with no ``calibrated`` field, every reported one when
      the class carries ``se_estimator``; on a scale-bearing class, the reported ones
      ``se_statistical`` leaves out (the calibration-free 2·A2·Mw product's, a DLS size's).
      Never on a mode-resolved result, whose ± takes ``se_excludes_inversion``;
    * ``se_ols`` -- ``se_estimator`` is ``'ols'``, the opt-in estimator (invariant 8); never
      about a ``<q>_se`` whose ``<q>_se_kind`` is ``'replicate_sem'``, a spread of
      replicates no regression estimator produced;
    * ``se_excludes_inversion`` -- ``mode_resolved`` is True and at least one ``*_se``
      value is reported: that ± excludes the DLS inversion's own systematics (invariant 8);
    * ``rg_below_resolution`` -- ``below_resolution_floor`` is True: the apparent Rg
      ``rg_app_nm`` sits below the q-window's ``rg_resolution_floor_nm``;
    * ``berry_past_zimm_limit`` -- ``method`` is ``'berry'`` and a finite ``qrg_max`` exceeds
      ``qrg_max_valid``: a Berry Rg and Mw read past the limit a Zimm Rg and Mw are read at,
      which is what the numbers are rather than a failed precondition, since no limit of
      Berry's own is held (the owner's ruling, 2026-10-16). Minted about the whole fit, it
      says what the Rg and Mw are, so a screen puts the word on those two rows' labels
      (Claude's recommendation, accepted, 2026-10-18, once the Zimm Mw shared the limit)
      while the export keeps it on every cell of the fit;
    * ``degenerate`` -- ``degenerate`` is True and ``n_solutions`` is an int of 2 or more:
      a shape fit that other solutions reproduce equally (R5.6), the one shown named by
      ``is_prolate`` where the class carries it.

    ``se_statistical`` is about those ``*_se`` fields, ``se_ols`` about all the class's
    ``*_se`` fields but a replicate SEM (an export's SE column names its estimator whether
    or not it holds a value), ``se_excludes_inversion`` and ``se_regression`` about the reported ones, ``rg_below_resolution`` about ``rg_app_nm``, a per-quantity ``apparent`` about
    its value, the others about the whole fit. Order is fixed: apparent, mode-resolved,
    mode family, average, weighted view, weighting, the four SE qualifiers, resolution,
    the Berry limit, degenerate, then a stored value's model-dependence and degenerate
    pick."""
    names = _field_names(result)
    out = []
    mode_resolved = _is_true(getattr(result, 'mode_resolved', None))
    if _is_true(getattr(result, 'is_apparent', None)) or mode_resolved:
        out.append(Note('apparent'))
    def flagged(name: str) -> bool:
        return name in names and _is_true(getattr(result, name, None))

    shapes = []
    for prefix, value_field in VALUE_FIELDS.items():
        if flagged(prefix + '_apparent'):
            out.append(Note('apparent', (value_field,)))
        if flagged(prefix + '_model_dependent'):
            shapes.append(Note('model_dependent', (value_field,),
                               {'model': getattr(result, prefix + '_model', '') or ''}))
        if flagged(prefix + '_degenerate'):
            shapes.append(Note('degenerate_pick', (value_field,)))
    if mode_resolved:
        out.append(Note('mode_resolved'))
    qualname = f'{type(result).__module__}.{type(result).__qualname__}'
    if qualname in MODE_FAMILY_CLASSES:
        out.append(Note('mode_family'))
    if (qualname in CUMULANT_AVERAGE_CLASSES
            or getattr(result, 'gamma_source', None) == 'cumulant'):
        out.append(Note('cumulant_average'))
    elif qualname in DISTRIBUTION_WEIGHTED_CLASSES:
        out.append(Note('distribution_weighted'))
    model_label = getattr(result, 'optical_model_label', None)
    weighting = getattr(result, 'weighting', None)
    if (isinstance(model_label, str) and model_label
            and isinstance(weighting, str) and weighting):
        out.append(Note('model_weighted_view', (),
                        {'weighting': weighting, 'model': model_label}))
    rows = _registry(registry)
    # A CONTIN result records its weighting on the distribution it wraps.
    weighted = getattr(result, 'distribution', None) or result
    for name in WEIGHTING_FIELDS:
        value = getattr(result, name, None) or getattr(weighted, name, None)
        code = WEIGHTING_PREFIX + value if isinstance(value, str) and value else ''
        if code in rows and all(n.code != code for n in out):
            out.append(Note(code, (), {'weighting': value}))
    ses = tuple(sorted(n for n in names if n.endswith('_se')))
    scaled_ses = tuple(n for n in ses if n not in ('rh_se', 'two_a2_mw_se'))
    reported = tuple(n for n in ses if getattr(result, n, None) is not None)
    if 'calibrated' in names and scaled_ses:
        out.append(Note('se_statistical', scaled_ses))
    # A ± no calibration enters: every reported one on a class with no `calibrated` field
    # (that carries an estimator), else the reported ones the scale cancels in.
    regression = (tuple(n for n in reported if n not in scaled_ses) if 'calibrated' in names
                  else reported if 'se_estimator' in names else ())
    if regression and not mode_resolved:
        out.append(Note('se_regression', regression))
    if getattr(result, 'se_estimator', None) == 'ols':
        ols = tuple(n for n in ses if getattr(result, n + '_kind', None) != 'replicate_sem')
        if ols:
            out.append(Note('se_ols', ols))
    if mode_resolved and reported:
        out.append(Note('se_excludes_inversion', reported))
    if _is_true(getattr(result, 'below_resolution_floor', None)):
        out.append(Note('rg_below_resolution', ('rg_app_nm',),
                        {'rg': getattr(result, 'rg_app_nm', None),
                         'floor': getattr(result, 'rg_resolution_floor_nm', None)}))
    qrg_max, qrg_limit = getattr(result, 'qrg_max', None), getattr(result, 'qrg_max_valid', None)
    if (getattr(result, 'method', None) == 'berry'
            and all(isinstance(v, (float, np.floating)) and np.isfinite(v)
                    for v in (qrg_max, qrg_limit))
            and qrg_max > qrg_limit):
        out.append(Note('berry_past_zimm_limit', (),
                        {'qrg': float(qrg_max), 'limit': float(qrg_limit)}))
    n_solutions = getattr(result, 'n_solutions', None)
    if (_is_true(getattr(result, 'degenerate', None)) and isinstance(n_solutions, int)
            and not isinstance(n_solutions, bool) and n_solutions >= 2):
        prolate = getattr(result, 'is_prolate', None)
        shown = (('prolate' if prolate else 'oblate')
                 if isinstance(prolate, (bool, np.bool_)) else 'one')
        out.append(Note('degenerate', (), {'n_other': n_solutions - 1, 'shown': shown}))
    out.extend(shapes)
    # A plot over several samples says what its points include (R5.6), off its own flags.
    if _is_true(getattr(result, 'any_model_dependent', None)):
        out.append(Note('includes_model_dependent'))
    if _is_true(getattr(result, 'any_degenerate', None)):
        out.append(Note('includes_degenerate_pick'))
    branching_index = getattr(result, 'branching_index', None)
    if (isinstance(branching_index, float) and not isinstance(branching_index, bool)
            and np.isfinite(branching_index)):
        out.append(Note('branching_model_dependent', ('branching_index',)))
    return tuple(out)


# ------------------------------------------------------------------------------------
# Reading a result's notes by code, never by wording.
# ------------------------------------------------------------------------------------

NoteLike = Union[Note, str]


def _is_note_sequence(source: Any) -> bool:
    return isinstance(source, Iterable) and not isinstance(source, (str, bytes, Mapping))


def notes_in(source: Any) -> Tuple[NoteLike, ...]:
    """The notes of ``source``: a tuple or list of notes as given, else the ``notes``
    attribute of a result (``()`` when it has none). A lone sentence (a notes field still
    holding one string) or a lone :class:`Note` is a one-element tuple; an empty sentence
    is none."""
    if isinstance(source, (Note, str)):
        return (source,) if isinstance(source, Note) or source.strip() else ()
    notes = source if _is_note_sequence(source) else getattr(source, 'notes', ())
    if notes is None:
        return ()
    if isinstance(notes, Note):
        return (notes,)
    if isinstance(notes, str):
        return (notes,) if notes.strip() else ()
    return tuple(notes)


def find_notes(source: Any, code: Optional[str] = None, *, kind: Optional[str] = None,
               about: Optional[str] = None,
               registry: Optional[Mapping[str, NoteSpec]] = None) -> Tuple[Note, ...]:
    """The :class:`Note` items of ``source`` (a result or an iterable of notes) of ``code``
    and/or ``kind``, and, when ``about`` is given, about that attribute; a sentence not yet
    converted never matches."""
    rows = _registry(registry) if kind is not None else {}
    out = []
    for n in notes_in(source):
        if not isinstance(n, Note):
            continue
        if code is not None and n.code != code:
            continue
        if kind is not None and (n.code not in rows or rows[n.code].kind != kind):
            continue
        if about is not None and about not in n.about:
            continue
        out.append(n)
    return tuple(out)


def has_note(source: Any, code: str, *, about: Optional[str] = None) -> bool:
    """True when ``source`` (a result or an iterable of notes) holds a :class:`Note` of
    ``code``, about ``about`` when that is given."""
    return bool(find_notes(source, code, about=about))


def _kind(note: NoteLike, registry: Mapping[str, NoteSpec]) -> str:
    """A note's kind; a sentence, or a code with no registry row, is read as a record."""
    if isinstance(note, Note):
        spec = registry.get(note.code)
        return spec.kind if spec is not None else 'record'
    return 'record'


# ------------------------------------------------------------------------------------
# One cause per line: several sources' notes grouped by code, for one surface.
# ------------------------------------------------------------------------------------

# A surface names the runs a cause holds up to this many, and counts them past it.
NAME_MAX = 3


def value_range(values) -> Any:
    """One value standing for several copies of a note: the value itself when every copy
    agrees, else the span of the numbers, rendered by the registry's own formatter, else
    the distinct values in order."""
    distinct = list(dict.fromkeys(values))
    if len(distinct) == 1:
        return distinct[0]
    numbers = [v for v in distinct if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(numbers) == len(distinct):
        return f'{_shown(min(numbers), 3)}–{_shown(max(numbers), 3)}'
    return tuple(distinct)


def merged(copies: Iterable[Note]) -> Note:
    """Several copies of one code as one note: each value its ``value_range`` across the
    copies that carry it, the ``about`` their union in first-seen order.

    The merged note is one line stating its cause once (the owner's message rule,
    ``grouped``), so its ``where`` names every copy's origin once. A unit is the line's only
    when every copy that names an origin shares it; otherwise each origin keeps its own unit
    beside it, as ``_origin`` writes one over several sources, and the note keeps none: a
    Γ-vs-q² copy's angles and a D-vs-c copy's concentrations read ``45 deg, 0.001 g/mL:``,
    and a replicate row's count beside an angle ``45 deg, 3 of 3 fits:``."""
    copies = tuple(copies)
    first = copies[0]
    if len(copies) == 1:
        return first
    keys = list(dict.fromkeys(k for n in copies for k in n.values))
    about = tuple(dict.fromkeys(a for n in copies for a in n.about))
    values = {k: value_range([n.values[k] for n in copies if k in n.values]) for k in keys}
    if WHERE in values:
        located = [n for n in copies if n.values.get(WHERE) not in (None, '', ())]
        # One unit for the line only when every origin is in it, the unit the line prints,
        # else each keeps its own (the docstring's rule).
        own_units = len({n.values.get(WHERE_UNIT) or '' for n in located}
                        | {values.get(WHERE_UNIT) or ''}) > 1
        # The copies' origins, each once: a relayed note names its runs or points, so the
        # merged one names all of them rather than a span or a tuple of tuples.
        origins: list[Union[str, float]] = []
        for n in located:
            where = n.values[WHERE]
            if own_units:
                origins.append(_origin('', n))
            else:
                origins.extend(where if isinstance(where, tuple) else (where,))
        values[WHERE] = tuple(dict.fromkeys(origins))
        if own_units:
            values.pop(WHERE_UNIT, None)
    return dataclasses.replace(first, about=about, values=values)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _mergeable(a: Note, b: Note) -> bool:
    """Two copies of one code read as one line: their values differ in at most one key,
    and that a number (the measured value, spanned), never in a model or a word, nor in
    two values -- a rod's p outside 2–30 and an ellipsoid's outside 0.1–10 are two facts.
    The one differing number is not told apart from a limit, so copies whose measured
    values agree exactly and whose limits differ also merge; a limit read from a module
    constant cannot differ."""
    keys_a = {k for k in a.values if k not in (WHERE, WHERE_UNIT)}
    keys_b = {k for k in b.values if k not in (WHERE, WHERE_UNIT)}
    if keys_a != keys_b:
        return False
    differ = [k for k in keys_a if a.values[k] != b.values[k]]
    return not differ or (len(differ) == 1 and _number(a.values[differ[0]])
                          and _number(b.values[differ[0]]))


def groups(sources: Iterable[Tuple[str, Any]], *, kinds: Tuple[str, ...] = ('precondition',),
           what: str = 'runs', n_total: Optional[int] = None, name_max: int = NAME_MAX,
           registry: Optional[Mapping[str, NoteSpec]] = None
           ) -> Tuple[Tuple[Note, Tuple[str, ...], Tuple[Note, ...]], ...]:
    """``grouped``'s lines with the copies each one stands for: ``(note, names, members)``,
    so a surface can find the line a given note is on."""
    if what not in WHERE_NOUNS:
        raise ValueError(f'{what!r} is not a noun of WHERE_NOUNS')
    rows = _registry(registry)
    sources = [(name, notes_in(src)) for name, src in sources]
    clusters: list = []                      # [code, [copies], [names]]
    for name, notes in sources:
        for n in notes:
            if not (isinstance(n, Note) and _kind(n, rows) in kinds):
                continue
            entry = next((c for c in clusters if c[0] == n.code and _mergeable(c[1][0], n)),
                         None)
            if entry is None:
                entry = [n.code, [], [], []]
                clusters.append(entry)
            entry[1].append(n)
            if name not in entry[2]:
                entry[2].append(name)
            label = _origin(name, n)
            if label not in entry[3]:
                entry[3].append(label)
    out = []
    total = n_total if n_total is not None else len(sources)
    for _code, copies, names, labels in clusters:
        note = merged(copies)
        if total > 1:
            where = (tuple(labels) if len(names) <= name_max
                     else WHERE_COUNT.format(n=len(names), total=total, what=what))
            # The labels carry each copy's own unit, so the merged note keeps none.
            values = {k: v for k, v in note.values.items() if k != WHERE_UNIT}
            note = dataclasses.replace(note, values={**values, WHERE: where})
        out.append((note, tuple(names), tuple(copies)))
    return tuple(out)


def _origin(name: str, note: Note) -> str:
    """A source's name for a line over several sources, with the copy's own origin beside
    it when it names one ('run A (slow mode)', 'fit 2 (30 deg)'): the source says which run,
    the copy which part of it. With no name, the origin alone, as ``merged`` writes each one
    when the copies' units differ ('45 deg')."""
    own = note.values.get(WHERE)
    if own in (None, '', ()):
        return name
    unit = note.values.get(WHERE_UNIT) or ''
    text = f"{_shown(own, 6)}{' ' + unit if unit else ''}"
    return f'{name} ({text})' if name else text


def grouped(sources: Iterable[Tuple[str, Any]], *, kinds: Tuple[str, ...] = ('precondition',),
            what: str = 'runs', n_total: Optional[int] = None, name_max: int = NAME_MAX,
            registry: Optional[Mapping[str, NoteSpec]] = None
            ) -> Tuple[Tuple[Note, Tuple[str, ...]], ...]:
    """One line per distinct cause of ``kinds`` over ``sources``, with the sources that
    carry it: the owner's message rule for a verdict over several results, each cause once
    with its count and its value range.

    ``sources`` is ``[(name, result or notes)]``. A cause is a code, and copies of one code
    are one cause when they differ in at most one numeric value (``_mergeable``); copies
    that differ in two values or in a word are separate lines. Over one source a lone copy
    is returned as it was minted; otherwise a cause's copies are ``merged`` and, over
    several sources, its ``where`` names the sources holding it, each with the copy's own
    origin beside it where the copy names one (``_origin``), up to ``name_max``, else
    counts them ("12 of 40 runs", out of ``n_total`` or the number of sources, the noun
    ``what``). First-seen order."""
    return tuple((note, names) for note, names, _members in
                 groups(sources, kinds=kinds, what=what, n_total=n_total,
                        name_max=name_max, registry=registry))
