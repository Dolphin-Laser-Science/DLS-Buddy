"""Single source of truth for the DLS Buddy version.

Pre-1.0 the scheme is ``0.MINOR.PATCH`` (semantic-versioning style): bump MINOR
for new user-facing capability, PATCH for fixes/polish. Surfaced in the GUI
window title, the generated guide PDFs, and headed in ``PATCH_NOTES.md``.

Import it from the app package: ``from app.version import __version__``.
"""

from __future__ import annotations

__version__ = "0.16.0"
