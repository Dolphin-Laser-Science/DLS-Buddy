"""
app/settings.py
===============

Global user preferences (the Settings module's data), as a framework-agnostic
dataclass owned by the controller.

The platform's locked rule is **seed, never override**: a setting is the *initial*
value a module's per-run control starts at, and the per-run value the user actually
chooses is what is applied and recorded in the result. A setting never silently
multiplies or replaces a computed number. So every analysis-affecting field here is
consumed as a default the GUI control is seeded with (or that a controller method
falls back to when the caller passes None) — not as a hidden global. Appearance
fields (theme) are the one exception: they are purely global presentation.

Because the values live here, decoupled from where they are edited, moving a setting
between modules later is a localized GUI change (relocate the widget, bind it to the
same field) — no data migration.

Persistence: a JSON file at the repo root (resolved from this file's location, not
the CWD), gitignored. NOTE: if the program is ever packaged and installed to a
read-only location, switch `default_path()` to a per-user config directory.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

SETTINGS_FILENAME = 'settings.json'


@dataclass
class SettingsState:
    """Seed defaults + appearance. Add fields here as more settings are wired."""

    # --- DLS analysis defaults ---
    cumulant_method: str = 'nonlinear'      # 'nonlinear' (Frisken 2001) or 'linear' (Koppel 1972)
    cumulant_order: int = 2                 # cumulant expansion order (1-3)
    skip_initial_channels: int = 0          # leading lag channels to drop (afterpulsing); 0 = keep all
    rh_grid_min_nm: float = 1.0             # CONTIN/NNLS Rh grid lower bound
    rh_grid_max_nm: float = 1000.0          # CONTIN/NNLS Rh grid upper bound
    rh_grid_points: int = 100               # CONTIN/NNLS Rh grid resolution
    lcurve_alpha_min: float = 1.0e-6        # CONTIN L-curve alpha sweep lower bound
    lcurve_alpha_max: float = 1.0e2         # CONTIN L-curve alpha sweep upper bound
    contin_alpha_method: str = 'lcurve'     # CONTIN alpha selection: 'lcurve' (corner,
    #                                         default) or 'ftest' (Provencher F-test)
    contin_ftest_prob_reject: float = 0.5   # F-test probability-to-reject level
    #                                         (Provencher default 0.5); higher -> smoother

    # --- SLS defaults ---
    standard_geometry: str = 'VU'           # toluene Rayleigh geometry: VU / VV / VH
    guinier_qrg_max: float = 1.3            # Guinier validity limit q*Rg

    # --- solvent library ---
    default_solvent: str = 'water'          # preselects the Solvent Explorer combo
    #                                         (seed, not a physics override)

    # --- uncertainty (applies to every regression SE: SLS + DLS Gamma-q^2/kD) ---
    se_estimator: str = 'hc3'               # 'hc3' (robust, default; never under-reports)
    #                                         or 'ols' (classical s^2(X'X)^-1, opt-in for
    #                                         comparability; can under-report). See the
    #                                         Theory-and-Equations-Guide 15.1.

    # NOTE: the synthetic-generator (β/noise/points) and intensity-trace (outlier k,
    # baseline, block-variance, ADF) defaults used to live here. They were moved into
    # their own tabs as plain session-only fields (Synthetic generator / Traces),
    # seeded each launch from in-module code constants, and the controller's trace
    # methods fall back to the constants in app/controller.py. They are intentionally
    # no longer persisted settings.

    # --- result display (GUI-global; applies directly, not a seed) ---
    no_uncertainty_sig_figs: int = 3        # significant figures for a result that has
    #                                         NO honest ± (single correlogram, NNLS/CONTIN,
    #                                         single-angle Mw). NEVER governs a value that
    #                                         HAS a ± — that place is σ-driven.

    # --- appearance (GUI-global; may apply directly, not seed) ---
    theme: str = 'system'                   # 'system' / 'light' / 'dark'
    ui_density: str = 'comfortable'         # app-wide font size: 'compact' / 'comfortable'
    #                                         / 'large' (accessibility + screen real-estate)
    show_tooltips: bool = True              # passive hover tooltips on/off
    plot_palette: str = 'okabe_ito'         # 'okabe_ito' / 'tab10' / 'grayscale'
    plot_match_theme: bool = False          # theme the ON-SCREEN plot to the dark app
    #                                         theme (opt-in). Export ALWAYS stays white
    #                                         regardless (locked; style guide R10.3).
    # Plot-axis display units: {quantity: unit}. Empty = the human-scale defaults
    # from plotting.plots. Applied globally to the plots.
    plot_units: Dict[str, str] = field(default_factory=dict)

    # --- session (interface) ---
    reopen_last_session: bool = False       # auto-save the workspace on exit and reopen
    #                                         it on the next launch (off by default)
    last_session_path: str = ''             # app-managed autosave file (set by the GUI;
    #                                         resolved beside settings.json, never CWD)

    # ---- serialization ----
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SettingsState':
        """Tolerant: unknown keys are ignored, missing keys take the default, so a
        settings.json written by an older or newer build still loads."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    # ---- persistence ----
    @staticmethod
    def default_path() -> Path:
        # app/settings.py -> repo root is two levels up.
        return Path(__file__).resolve().parent.parent / SETTINGS_FILENAME

    @classmethod
    def load(cls, path: Optional[str] = None) -> 'SettingsState':
        """Load settings from JSON, or return defaults if the file is absent or
        unreadable (a corrupt file must never block startup)."""
        p = Path(path) if path else cls.default_path()
        try:
            with open(p, encoding='utf-8') as fh:
                return cls.from_dict(json.load(fh))
        except (OSError, ValueError):
            return cls()

    def save(self, path: Optional[str] = None) -> str:
        p = Path(path) if path else self.default_path()
        with open(p, 'w', encoding='utf-8') as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return str(p)
