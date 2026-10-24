"""
parsers/autodetect.py
=====================

The one place that answers "which parser reads this file?".

**Why this is a module and not a method on the window.** The contract used to live in two
hand-copied loops (`MainWindow._autodetect`, shared by the DLS and SLS routes, and
`UtilitiesModule._autodetect_trace`) plus three generic fallbacks, and the five call sites
disagreed with each other about their sentinels. A remediation round taught one loop the
new rule and left the others as they were, so a trace or SLS file with a bad column still
reported as "unrecognised format" — the very defect the round existed to fix, standing one
file over. Hoisting it here is the rule-10 remedy: a shared mechanism instead of N
hand-added predicates.

It cannot live in `gui/main_window.py`, because `main_window` imports `utilities_module`
and a helper there would be unreachable from the trace route — which is precisely the seam
the previous attempt stopped at.

**Invariant 2 holds.** This module imports only `base_parser`; the parser list is an
argument, and no parser class is named here. `Detection` deliberately carries **no** parser
name: an earlier version had `parser_name` / `malformed_parser` fields that every path wrote
and nothing read, and a name in scope is a name someone eventually branches on. They are
gone until a message actually needs one; `test_loader_diagnostics.py` still pins that the
GUI does not branch on parser identity, so re-adding one cannot go unnoticed.

**A commit failure is RECORDED, not raised.** Re-raising inside the loop would make the
load hostage to parser ORDER: a foreign file that happens to trip an earlier parser's
post-commit raise would abort before its own parser is ever tried. Measured across
`test-data/` that has no live instance today, but it is a real hazard and it is cheaper to
remove structurally than to pin. So the loop keeps going, and a later winner discards the
recorded error — which means a mis-classified raise site cannot make one parser's failure
pre-empt another parser's success.

**What it does NOT buy, corrected after the standard pass over-stated it.** An earlier
version of this paragraph claimed a mis-classified site "can never lose a file". It can:
every loader skips the generic fallback when `malformed` is set (deliberately — see
`_finish_load`), so a commit point set one call too early turns "falls through and loads as
a plain two-column table" into "not loaded, with a misleading report". The classification is
therefore gated behaviourally, not merely argued. The gate is a SCAN, not a roster:
`tests/fixtures/real_corpus.py` walks every committed file under `test-data/Real/` through
the three autodetect lists (the generic fallbacks stay out by design), `test_parser_error_taxonomy.py` fails if any is diagnosed as malformed by a
parser that does not own it, `test_parsers.py` pins the accepted set by identity, and
`test_autodetect_orchestration.py` pins that a foreign plain file comes back with
`malformed is None`. (An earlier version of this paragraph claimed the sweep while the test
was a ten-file hand list -- the gap through which a whole refused sub-family shipped.)
Those are the reason a per-parser review is enough; this loop's deferral is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from parsers.base_parser import MalformedDataError, ParseError


@dataclass(frozen=True)
class Detection:
    """What `autodetect` found. One sentinel, not two.

    `previews` is always a list, never None: both old "nothing matched" sentinels (`None`
    from the DLS/SLS loop, `[]` from the trace loop) collapse to one falsy value, so every
    call site's `if not detection.previews:` reads identically. That frees `None` at the
    two functions that own a dialog to mean exactly one thing — "nothing to load and the
    user has already been dealt with".
    """

    previews: list = field(default_factory=list)
    malformed: Optional[MalformedDataError] = None

    def reason(self, basename: str) -> str:
        """The `'<file>: <why>'` line for the malformed-file report. '' when there is none.

        Formatted here so the two loaders that render it cannot drift apart in wording.
        """
        return f'{basename}: {self.malformed}' if self.malformed is not None else ''


def autodetect(path: str, parser_classes: Sequence[type], *,
               skip_missing_file: bool = False) -> Detection:
    """Try each parser in order; return the first one whose previews are non-empty.

    Parameters
    ----------
    path
        The file to read.
    parser_classes
        Instrument parsers, in priority order. Generic (lenient) parsers must NOT be in
        this list -- they match anything, so they are the explicit fallback each caller
        applies afterwards.
    skip_missing_file
        Whether a parser's `FileNotFoundError` is skipped like a refusal (True) or
        propagates (False, the default). The trace route passes True
        (`app/loading.py::detect_trace`); the DLS and SLS loaders keep the default. The
        flag's two arms are pinned by `tests/test_autodetect.py`; each route's choice is
        its own call.

    Returns
    -------
    Detection
        `previews` non-empty on a match. Otherwise `malformed` carries the FIRST
        post-commit failure seen, if any -- see the module docstring on why it is recorded
        rather than raised, and why "first" rather than "last".
    """
    first: Optional[MalformedDataError] = None
    for parser_cls in parser_classes:
        try:
            previews = parser_cls().parse(path)
        except MalformedDataError as exc:
            # MUST precede `except ParseError` -- MalformedDataError is a subclass, so the
            # other order silently swallows it. This is the single mechanical fact that
            # made the previous attempt's handler dead code.
            if first is None:
                first = exc
            continue
        except ParseError:
            continue                      # not this format; try the next
        except FileNotFoundError:
            if skip_missing_file:
                continue
            raise
        if previews:
            return Detection(previews)
    return Detection([], first)
