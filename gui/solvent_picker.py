"""
gui/solvent_picker.py
=====================

The one way the program offers the solvent library in a dropdown.

Three widgets show this same roster — the Data tab's per-measurement solvent cell,
the Solvent Explorer's subject picker, and the Settings default-solvent field — and
before this module they showed it three different ways (10 names vs 19, suffixed vs
plain). A duplicated populate function is exactly the sibling hazard CLAUDE.md rule 10
is about, so there is one here and the three call it.

Two things it fixes beyond the duplication:

* **All 19 solvents, everywhere.** The Data tab used to offer only the primary tier.
  Estimate-tier solvents are weaker-data, not secret — the Explorer has always shown
  them, and hiding them from the Data tab only meant the user could look one up and
  then not pick it.
* **Split into labelled sections** ("Measured properties" / "Estimated properties")
  rather than tagging every second row `(estimate)`. Same information, read once at
  the top of a block instead of nine times down a column.

The canonical name rides on ``itemData``, so the visible label stays cosmetic and a
caller never parses it back.

**Closed by design.** These combos are non-editable. Free text is not forbidden — it
is moved to an explicit **Other…** entry (Data tab only), which is what makes it safe
to store solvent names verbatim: a picked value is canonical by construction, so
nothing downstream has to guess what the user meant (see core/data_models.py).
"""

from __future__ import annotations

from typing import Optional

from PySide6 import QtWidgets

import physics.solvents as solvents_lib
from core.data_models import casefold_identity_name
from gui.widgets import add_combo_section_header, first_selectable_index


# The trailing "type it yourself" entry. Its itemData is a sentinel rather than a
# name so it can never be confused with a solvent: casefold_identity_name would
# have to produce this exact string, and it cannot -- the library holds no name
# with a NUL in it, and neither does anything a user can type into the box.
OTHER_DATA = '\x00other'
OTHER_LABEL = 'Other…'

# Section titles. Wording tracks the library's own tier vocabulary and the
# lib_primary / lib_estimate provenance dots (ui_style_guide §"Colour tokens"),
# so a user meets the same distinction under the same words in both places.
_SECTIONS = (
    ('primary', 'Measured properties'),
    ('estimate', 'Estimated properties'),
)


def populate_solvent_combo(combo: QtWidgets.QComboBox, *,
                           include_blank: bool = False,
                           include_other: bool = False) -> None:
    """Fill `combo` with every library solvent, grouped into labelled sections.

    `include_blank` prepends an empty "not set yet" row (the Data tab needs one — a
    freshly loaded measurement has no solvent). `include_other` appends the
    :data:`OTHER_LABEL` row that unlocks free text.

    The combo is left non-editable and its current index is placed on the first
    row a user could actually choose, never on a section header.
    """
    combo.setEditable(False)
    combo.clear()

    if include_blank:
        combo.addItem('', None)

    for tier, title in _SECTIONS:
        names = solvents_lib.available_solvents(tier)
        if not names:
            continue
        if combo.count():
            combo.insertSeparator(combo.count())
        add_combo_section_header(combo, title)
        for name in names:
            combo.addItem(name, name)

    if include_other:
        if combo.count():
            combo.insertSeparator(combo.count())
        combo.addItem(OTHER_LABEL, OTHER_DATA)

    first = first_selectable_index(combo)
    if first >= 0:
        combo.setCurrentIndex(first)


def index_of_solvent(combo: QtWidgets.QComboBox, name: Optional[str]) -> int:
    """Index of the row for `name`, matched caselessly, or -1 if it is not in the list.

    Caselessly because the stored string is verbatim: a Zetasizer file supplies
    ``Water`` where the library key is ``water``, and both are the same solvent under
    the one comparison rule (:func:`core.data_models.casefold_identity_name`). A -1
    return is the signal that a name is free text — the caller shows it as *Other*
    rather than silently dropping it.
    """
    if not isinstance(name, str) or not name.strip():
        return -1
    wanted = casefold_identity_name(name)
    for i in range(combo.count()):
        data = combo.itemData(i)
        # BOTH sides are casefolded. The library's own rows already carry canonical
        # lowercase names, so this is a no-op for them -- but a caller may APPEND a row
        # for a name the library does not have (Settings does, to preserve a hand-edited
        # default), and that row's data is stored VERBATIM. Casefolding only the input
        # would then fail to match the row on the next lookup, and the caller would append
        # it again on every reload. Symmetry is what makes the lookup idempotent.
        if (isinstance(data, str) and data != OTHER_DATA
                and casefold_identity_name(data) == wanted):
            return i
    return -1


def solvent_names(combo: QtWidgets.QComboBox) -> list:
    """Every solvent name offered by `combo`, in display order.

    Excludes the blank row, the section headers, the separators and *Other*. The
    roster test uses this to assert the picker and the library agree, so a solvent
    added to physics/solvents.py cannot silently fail to appear.
    """
    return [combo.itemData(i) for i in range(combo.count())
            if isinstance(combo.itemData(i), str)
            and combo.itemData(i) != OTHER_DATA]
