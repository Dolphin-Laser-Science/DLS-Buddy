"""
parsers/roster.py
=================

The instrument parsers the format auto-detection tries on load, one list per data kind.

**Why the lists live here and not in `parsers/autodetect.py`.** That module's docstring makes
naming no parser part of its invariant-2 contract: the detection loop takes the list as an
argument, so it cannot branch on which parser won. This module is the one place that names
them, so a new instrument format is a new parser file plus one entry here, and no widget
module changes.

The lenient generic parsers are NOT in these lists: they match almost any numeric table, so
each loader tries its kind's generic parser only as the explicit fallback after instrument
detection fails.
"""

from __future__ import annotations

from parsers.alv_asc import ALVCorrelatorParser, ALVTraceParser
from parsers.brookhaven_dat import BrookhavenDATParser
from parsers.brookhaven_dls import BrookhavenDLSParser, BrookhavenTraceParser
from parsers.brookhaven_sls import BrookhavenSLSParser, BrookhavenSLSScanParser
from parsers.panta_xlsx import PantaXlsxParser
from parsers.zetasizer_clipboard import ZetasizerClipboardParser
from parsers.zetasizer_export import ZetasizerExportParser

# Instrument-specific parsers tried, in order, by the format auto-detection on load.
# Each sniffs its own format and raises ParseError when the file is not its own, so
# the first that succeeds is the right one. The lenient *generic* parsers are NOT
# here (they would match almost anything); they are tried only as an explicit
# fallback after instrument detection fails -- the load button declares the data
# kind, and generic DLS prompts for its delay unit + data form.
DLS_PARSERS = [ALVCorrelatorParser, ZetasizerClipboardParser,
               ZetasizerExportParser, BrookhavenDLSParser, BrookhavenDATParser,
               PantaXlsxParser]
# The two Brookhaven SLS entries read the instrument's two .CSV layouts (concentration
# series / single angular scan). Their sniffs are mutually exclusive by construction, so
# the order between them carries no meaning.
SLS_PARSERS = [BrookhavenSLSParser, BrookhavenSLSScanParser]

# Instrument trace parsers tried (in order) by the auto-detection on load. Each
# sniffs its own format strictly (ALV .ASC structure; Brookhaven HH:MM:SS
# timestamps) and raises ParseError otherwise, so the first that succeeds is the
# right one. The lenient generic two-column parser is the explicit fallback (it
# matches almost anything, so it must not be in this list) and prompts for units.
TRACE_PARSERS = [ALVTraceParser, BrookhavenTraceParser]
