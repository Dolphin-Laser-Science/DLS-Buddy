"""Single source of truth for the DLS Buddy version.

The scheme is ``MAJOR.MINOR.PATCH`` (semantic-versioning style): bump MINOR
for new user-facing capability, PATCH for fixes/polish. Surfaced in the GUI
window title, the generated guide PDFs, and headed in ``PATCH_NOTES.md``.

This value tracks the WORKING TREE's version (the unreleased top section of
``PATCH_NOTES.md``), not the last published release: bump it in the same
commit that opens or renumbers that section — ``publish_release.ps1``'s -Tag
guard cross-checks this value, so a lagging bump blocks the release.

Import it from the app package: ``from app.version import __version__``.
"""

from __future__ import annotations

__version__ = "1.21.0"
