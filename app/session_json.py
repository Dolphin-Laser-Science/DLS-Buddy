"""
app/session_json.py
===================

How a session file carries a non-finite float: `_json_sanitize` writes NaN and +/-Inf
as a one-key marker object, so the file stays strict JSON, and `_json_restore` turns
the marker back into its float on load. The controller's session save and load call
them. Qt-free (invariant 5).
"""

from __future__ import annotations

import math


# The one-key object a non-finite float is written as in a session file, and the three
# tokens it may carry. An OBJECT, not a bare string: no persisted dataclass field holds a
# dict with this key, so the reader can never mistake a user's own text (a polymer
# literally named "NaN") for a number, while strict JSON stays strict.
_NONFINITE_KEY = '__nonfinite__'
_NONFINITE_TOKENS = {'nan': float('nan'), 'inf': float('inf'), '-inf': float('-inf')}


def _json_sanitize(obj):
    """Recursively replace non-finite floats (NaN / ±Inf) with the ``_NONFINITE_KEY``
    sentinel object so a session serializes to strict, portable JSON
    (``allow_nan=False``) rather than the non-standard ``NaN``/``Infinity`` tokens
    strict/external readers reject. ``_json_restore`` is the exact inverse on load.

    Why a sentinel and not ``None``: ``None`` in a result field
    means "never computed", a NaN means "computed, no finite answer" -- ``run_zimm``
    stores a NaN Rg from a non-positive angular slope beside a live ``rg_source`` and
    label. Writing both as ``null`` made the same regroup gesture store a different Rg
    before and after a save/reload, because the merge branches on presence. A stored
    value must come back as what it was. Sessions written before the sentinel hold
    ``null`` there and load as ``None`` (readers treat the two alike -- the merge now
    included). ``np.float64`` is a ``float`` subclass, so the ``float`` check covers it."""
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        return {_NONFINITE_KEY: ('nan' if math.isnan(obj)
                                 else 'inf' if obj > 0 else '-inf')}
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return obj


def _json_restore(obj):
    """The inverse of ``_json_sanitize``: turn every ``_NONFINITE_KEY`` sentinel object
    back into its float, recursively, before the payload reaches any ``from_dict``.
    Only an object holding exactly that one key with one of the three tokens is
    restored; anything else passes through untouched, so a pre-sentinel session
    (``null`` for NaN) loads exactly as it did."""
    if isinstance(obj, dict):
        if (len(obj) == 1 and _NONFINITE_KEY in obj
                and obj[_NONFINITE_KEY] in _NONFINITE_TOKENS):
            return _NONFINITE_TOKENS[obj[_NONFINITE_KEY]]
        return {k: _json_restore(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_restore(v) for v in obj]
    return obj
