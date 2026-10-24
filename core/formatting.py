"""
core/formatting.py
==================

The house display formatters and the decimal-place rule the solvent autofill rounds at.

Two kinds of code share this module:

- **Text formatting.** `format_pm`, `_format_sig`, `format_fixed_sig`,
  `format_value_at_uncertainty` and `pdg_erases_value` turn a value and its ± into display
  and record text: the GUI's cells and labels, and the record lines `core.naming` builds,
  which reach an export's Comments notes.
- **The decimal-place rule.** `_pdg_decimals` chooses the place an uncertainty supports, and
  `round_to_uncertainty` rounds a value at it. The controller's solvent autofill calls the
  latter to round the library's proposed refractive index and viscosity before they are
  written into the measurement parameters the DLS and SLS fits read, so a defect here moves
  q and Rh, not only a displayed digit.

Both kinds read the one place-chooser, `_pdg_decimals`, so the displayed precision and the
autofill rounding cannot drift apart. Pure functions: no Qt, no I/O, and nothing imported
from `analysis`, so `core/` stays below the engine.
"""

from __future__ import annotations

import math
from typing import Optional


# ---------------------------------------------------------------------------
# Display: value +/- uncertainty with significant figures driven by the SE
# ---------------------------------------------------------------------------

def _pdg_decimals(sigma: float) -> int:
    """The decimal place (as ``round()``'s ``ndigits``) the PDG-style convention
    assigns to an uncertainty: round sigma to 1 significant figure — 2 if its
    leading digit is 1 — and report the value at that same place. The single
    place-chooser shared by ``format_pm`` and ``round_to_uncertainty`` so the
    display convention and the autofill rounding can never drift apart."""
    # BOTH the exponent and the leading digit come from ONE read of the value -- the point
    # of this line, not an incidental economy. Sourcing them separately (`floor(log10)` for
    # the exponent, the rendering for the digit) makes two decade decisions that disagree
    # wherever the double sits a hair below a power of ten: `log10` rounds up while the
    # rendering shows `9.99...e(exp-1)`. The width matters too: `.17e` is the exact
    # round-trip width of a double, so the mantissa is never rounded across a decade, where
    # the default six-decimal `%e` renders 9.9999999e-5 as '1.000000e-04' -- a leading digit
    # of 1 where the true one is 9.
    mantissa, _, exponent = f'{sigma:.17e}'.partition('e')
    exp = int(exponent)
    lead = int(mantissa[0])                    # first significant digit of sigma
    sig = 2 if lead == 1 else 1
    return -(exp - (sig - 1))                  # decimals for round()


def round_to_uncertainty(value: Optional[float],
                         sigma: Optional[float]) -> Optional[float]:
    """Round ``value`` at the last digit ``sigma`` can stand behind (the PDG-style
    place of ``format_pm``) — e.g. value 1.3325541, sigma 6e-4 -> 1.3326.

    Used by the solvent-library autofill to strip false precision from proposed
    values before they are written (display-class perturbation, <= half a unit in
    the last confident digit; the sigma chooses the decimal place and travels no
    further — invariant #8). Idempotent. A missing/zero/non-finite sigma means
    there is nothing to stand behind, so the value is returned unrounded.
    """
    has_val = value is not None and math.isfinite(value)
    has_sig = sigma is not None and math.isfinite(sigma) and sigma > 0
    if not has_val or not has_sig:
        return value
    return round(value, _pdg_decimals(sigma))


def _format_sig(value: float, sig: int) -> str:
    """Render a finite ``value`` at ``sig`` significant figures in the house
    convention: scientific notation with a BARE integer exponent when
    ``|value| >= 1e5`` or ``0 < |value| < 1e-3`` — the SAME threshold and exponent
    style ``format_pm`` uses for a ± value — otherwise fixed-point. Trailing zeros
    are stripped: a value with no σ has nothing to make them confident (unlike
    ``format_value_at_uncertainty``, whose trailing zeros ARE σ-backed).

    This is the shared engine for every ±-less bare-value display, so a no-±
    result never renders '1.23e+06' next to a ± result's '(1.23 ± …)e6'.
    """
    if value == 0:
        return '0'
    # Round to `sig` significant figures FIRST (as format_pm rounds the value to the SE
    # place first), so a large value never shows MORE digits than `sig` — e.g. 3421.7 at
    # 3 sig -> 3420, not 3422 — then decide sci vs fixed on the ROUNDED magnitude (a value
    # that rounds up across 1e5, e.g. 99999 -> 1.00e5, then renders in scientific form).
    exp10 = math.floor(math.log10(abs(value)))
    ndig = sig - 1 - exp10                            # sig-fig rounding place (may be <= 0)
    r = round(value, ndig)
    if r == 0:
        return '0'
    mag = abs(r)
    if mag >= 1e5 or mag < 1e-3:
        # %e re-expresses the already-rounded value with correct carry (9.999e5 -> 1e6).
        mant, _, e = f'{r:.{sig - 1}e}'.partition('e')
        mant = mant.rstrip('0').rstrip('.')
        return f'{mant}e{int(e)}'                     # int() drops the '+' and zero-padding
    s = f'{r:.{max(0, ndig)}f}'                       # sig figs -> decimal places
    return s.rstrip('0').rstrip('.') if '.' in s else s


def format_pm(value: Optional[float], se: Optional[float], unit: str = '') -> str:
    """Format 'value +/- se' with the precision set by the uncertainty.

    Convention (PDG-style): round the SE to 1 significant figure, or 2 if its
    leading digit is 1; round the value to the same decimal place (the place
    comes from the shared ``_pdg_decimals``). Large/small magnitudes are shown
    with a shared power of ten, e.g. '(1.23 +/- 0.06)e6'. Falls back to a plain
    value when no usable SE is given.
    """
    has_val = value is not None and math.isfinite(value)
    has_se = se is not None and math.isfinite(se) and se > 0
    if not has_val:
        return 'n/a'
    if not has_se:
        s = _format_sig(value, 4)                   # ±-less fallback: house sig-fig render
        return f'{s} {unit}' if unit else s

    exp = math.floor(math.log10(se))            # SE's order (sci-notation fallback)
    dec = _pdg_decimals(se)
    last = -dec                                 # power-of-ten place of the last sig fig
    se_r = round(se, dec)
    val_r = round(value, dec)

    mag = max(abs(val_r), se_r)
    use_sci = mag != 0 and (mag >= 1e5 or mag < 1e-3)
    if use_sci:
        # isfinite: `!= 0` admits both inf and nan (nan != 0 is True), and
        # math.floor(math.log10(inf)) raises OverflowError. UNREACHABLE, though, and
        # recorded as such: format_pm returns 'n/a' at its `has_val` check above unless
        # `value` is finite, and rounding a finite value cannot produce a non-finite one.
        # Kept as defence in depth for a future caller that reaches this helper by another
        # route -- it was reported as a live crash on the display path, and it is not.
        E = (math.floor(math.log10(abs(val_r)))
             if (math.isfinite(val_r) and val_r != 0) else exp)
        mant_dec = max(0, E - last)
        v, s = val_r / 10 ** E, se_r / 10 ** E
        body = f'({v:.{mant_dec}f} ± {s:.{mant_dec}f})e{E}'
    else:
        d = max(0, dec)
        body = f'{val_r:.{d}f} ± {se_r:.{d}f}'
    return f'{body} {unit}' if unit else body


def pdg_erases_value(value: Optional[float], se: Optional[float]) -> bool:
    """True when the PDG place chosen by ``se`` rounds a NON-ZERO ``value`` to zero.

    The display convention sets the value's precision from its uncertainty, which is
    right until the uncertainty is so much larger that the value's own leading digit
    sits below the place sigma supports: ``30 +/- 900`` renders as ``0 +/- 900`` and the
    measurement is gone from the cell. The condition is not "sigma exceeds the value"
    (``30 +/- 90`` renders fine and is informative) and not any ratio -- pick a ratio and
    it is a magic number, since the boundary is a decade artifact of the rounding, not a
    property of the quantity. So this asks the actual question, through the same
    ``_pdg_decimals`` the renderer uses, and cannot drift from it.

    A caller that gets True should show the value at its own fixed precision and the
    uncertainty beside it, rather than through it (``format_fixed_sig`` on each) -- the
    number and its spread are both real and both belong on screen. Nothing here changes
    what an SE IS: the estimator's value is untouched, and exports, which write value and
    SE in separate full-precision columns, never reach this question.

    False whenever there is nothing to erase: no value, no usable SE, or a value that is
    itself zero (the render is then honest).
    """
    if value is None or se is None:
        return False
    if not (math.isfinite(value) and math.isfinite(se)) or se <= 0 or value == 0:
        return False
    return round(value, _pdg_decimals(se)) == 0


def format_fixed_sig(value: Optional[float], sig: int = 3) -> str:
    """Format ``value`` at a fixed number of significant figures — the display rule
    for **every result that has no honest uncertainty** (invariant #8).

    Used wherever an SE is genuinely unavailable, not merely absent from the display:
    a single-correlogram dynamic fit (correlated lag channels, Schaetzel 1990), an
    ill-posed inversion (NNLS/CONTIN), a single datum (single-angle Mw), and every
    other ±-less reported quantity (Γ, β, τ, PDI, ρ, δ², R², qRg, …). We must NOT
    fabricate a ± just to drive the σ-aware formatter (``format_value_at_uncertainty``);
    instead we show a modest, fixed precision that carries no confidence claim — the
    number's honesty is carried by its apparent/±-omitted qualifier (invariant #7),
    the digit count by this rule.

    Standing policy (the rounding sentence of the Theory-and-Equations-Guide, "Uncertainty" chapter opening): a value that HAS an uncertainty is
    set by that uncertainty and never comes here; a value WITHOUT one is set by ``sig``,
    which is user-configurable (Settings → the no-uncertainty precision knob, default 3).
    Because any sig-fig choice for an unquantified number is inherently arbitrary, that is
    no invariant-8 violation — nothing is shown as more certain than its ± supports.
    Rendering (scientific-notation threshold + bare exponent) is shared with ``format_pm``
    via ``_format_sig``. Returns 'n/a' for a missing/non-finite value.
    """
    if value is None or not (isinstance(value, (int, float)) and math.isfinite(value)):
        return 'n/a'
    return _format_sig(float(value), sig)


def format_value_at_uncertainty(value: Optional[float],
                                sigma: Optional[float]) -> str:
    """Format ``value`` at the decimal place ``sigma`` stands behind, KEEPING the
    confident trailing zeros that place implies (e.g. value 1.33, sigma 6e-4 ->
    '1.3300') — the bare-value companion to :func:`format_pm`, with no '+/- se'.

    Used for a display-only readout of an already-rounded, uncertainty-tagged value
    (the Data-tab solvent-library cells): the value is shown to exactly the precision
    the library's per-condition sigma supports, so a rounded 1.3300 does not collapse
    to '1.33'. The place comes from the shared :func:`_pdg_decimals`, so it can never
    drift from :func:`round_to_uncertainty` (the autofill rounding). Falls back to a
    clean general format when sigma is missing/zero/non-finite (nothing to stand
    behind). The sigma chooses the decimal place and travels no further (invariant #8).
    """
    if value is None or not math.isfinite(value):
        return ''
    has_sig = sigma is not None and math.isfinite(sigma) and sigma > 0
    if not has_sig:
        return f'{value:.10g}'
    # Round at the σ place FIRST (honoring a negative ndigits for a coarse σ, e.g.
    # σ≈60 -> the tens place), exactly as round_to_uncertainty/format_pm do, THEN
    # show that place — so the two can never drift. `max(0, dec)` only clips the
    # DISPLAY decimals (a place at/left of the units digit shows no fractional part).
    dec = _pdg_decimals(sigma)
    return f'{round(value, dec):.{max(0, dec)}f}'
