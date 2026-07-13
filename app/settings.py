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
import math
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, get_type_hints

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
    contin_alpha_method: str = 'gcv'        # CONTIN alpha selection: 'gcv' (Golub 1979,
    #                                         default), 'lcurve' (Hansen max-curvature
    #                                         corner), or 'ftest' (Provencher F-test)
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
    #                                         Theory-and-Equations-Guide 6.1.

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
    def _defaults(cls) -> Dict[str, Any]:
        """Each field's default value (resolving default_factory), without building a full
        instance -- used to name the value a reverted field falls back to."""
        out: Dict[str, Any] = {}
        for f in fields(cls):
            if f.default is not MISSING:
                out[f.name] = f.default
            elif f.default_factory is not MISSING:   # type: ignore[misc]
                out[f.name] = f.default_factory()     # type: ignore[misc]
        return out

    @classmethod
    def _coerce_field(cls, name: str, value: Any, hint: Any,
                      default: Any) -> Tuple[Any, bool, Optional[str]]:
        """Validate/coerce one loaded field against its declared type.

        Returns ``(coerced_value, ok, problem)``. ``ok`` True means use ``coerced_value``
        (possibly coerced from a recoverable form -- "2"->2, 2.0->2, 2->2.0); ``ok`` False
        means the value is unrecoverable, so the caller reverts to the default and surfaces
        ``problem`` (a human-readable one-liner) to the user. bool is handled before int
        because ``bool`` is a subclass of ``int`` (a stray ``true`` must not pass as 1)."""
        def bad(expected: str) -> Tuple[Any, bool, str]:
            return None, False, (f"{name}: expected {expected}, got {value!r} — "
                                 f"using default ({default!r})")

        if hint is bool:
            if isinstance(value, bool):
                return value, True, None
            if isinstance(value, int) and value in (0, 1):   # recoverable 0/1
                return bool(value), True, None
            return bad("true or false")

        if hint is int:
            if isinstance(value, bool):
                return bad("a whole number")
            if isinstance(value, int):
                return value, True, None
            if isinstance(value, float) and math.isfinite(value) and value.is_integer():
                return int(value), True, None                # recoverable 2.0 -> 2
            if isinstance(value, str):
                try:
                    f = float(value)
                except ValueError:
                    return bad("a whole number")
                if math.isfinite(f) and f.is_integer():
                    return int(f), True, None                # recoverable "2" / "2.0"
            return bad("a whole number")

        if hint is float:
            if isinstance(value, bool):
                return bad("a number")
            if isinstance(value, (int, float)):
                f = float(value)                             # recoverable int -> float
            elif isinstance(value, str):
                try:
                    f = float(value)                         # recoverable "1.3" -> 1.3
                except ValueError:
                    return bad("a number")
            else:
                return bad("a number")
            if not math.isfinite(f):
                return bad("a finite number")
            return f, True, None

        if hint is str:
            # Every string setting (free-form paths/solvent names AND the method/appearance
            # enums) is accepted on a type check alone -- we deliberately do NOT validate an
            # enum value against a hard-coded allow-list. An unknown-but-string value (a typo,
            # or a choice from a newer build) passes through to its consumer, which is already
            # safe: the analysis SE falls back to HC3, an unknown Rayleigh geometry raises
            # loudly at calibration, and the method selectors fall back to their defaults -- so
            # a bad enum can never produce a silently-wrong result. This keeps forward-compat
            # ("a newer build's settings.json still loads") and avoids a drift-prone hand list
            # (owner decision, 2026-07-12, after the audit-04 review found both costs).
            if isinstance(value, str):
                return value, True, None
            return bad("text")

        if isinstance(default, dict):                        # plot_units (Dict[str, str])
            if isinstance(value, dict):
                return value, True, None
            return bad("an object")

        return value, True, None                             # unknown future type: accept

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SettingsState':
        """Tolerant: unknown keys are ignored, missing keys take the default, so a
        settings.json written by an older or newer build still loads. Invalid-typed values
        are coerced when recoverable and otherwise reverted to the default (see
        :meth:`from_dict_with_report` for the reversion report)."""
        return cls.from_dict_with_report(d)[0]

    @classmethod
    def from_dict_with_report(cls, d: Any) -> Tuple['SettingsState', List[str]]:
        """Like :meth:`from_dict`, but also returns a list of human-readable problems for
        every field that was reverted to its default because its value could not be honored.
        A non-dict top level (``42`` / ``"x"`` / ``[1]``) yields all-defaults plus one
        problem -- a corrupt file must never block startup, but the user is told."""
        if not isinstance(d, dict):
            return cls(), [f"settings file is not a settings object (got "
                           f"{type(d).__name__}); using all defaults"]
        known = {f.name for f in fields(cls)}
        hints = get_type_hints(cls)
        defaults = cls._defaults()
        accepted: Dict[str, Any] = {}
        problems: List[str] = []
        for k, v in d.items():
            if k not in known:
                continue                                     # unknown key ignored (tolerant)
            coerced, ok, problem = cls._coerce_field(k, v, hints.get(k), defaults.get(k))
            if ok:
                accepted[k] = coerced
            elif problem:
                problems.append(problem)                     # field reverts to its default
        return cls(**accepted), problems

    # ---- persistence ----
    @staticmethod
    def default_path() -> Path:
        # app/settings.py -> repo root is two levels up.
        return Path(__file__).resolve().parent.parent / SETTINGS_FILENAME

    @classmethod
    def load(cls, path: Optional[str] = None) -> 'SettingsState':
        """Load settings from JSON, or return defaults if the file is absent or
        unreadable (a corrupt file must never block startup). Thin wrapper over
        :meth:`load_with_report` for callers that don't surface the reversion report."""
        return cls.load_with_report(path)[0]

    @classmethod
    def load_with_report(cls, path: Optional[str] = None) -> Tuple['SettingsState', List[str]]:
        """Load settings and also return a list of human-readable problems (see
        :meth:`from_dict_with_report`) so the GUI can tell the user which saved settings were
        invalid and reverted to defaults. An ABSENT file is normal -> (defaults, no problems);
        a present-but-unreadable/corrupt file -> (defaults, one problem). A corrupt file must
        never block startup."""
        p = Path(path) if path else cls.default_path()
        if not p.exists():
            return cls(), []                       # absent is the ordinary first-run case
        try:
            with open(p, encoding='utf-8') as fh:
                raw = json.load(fh)
        except (OSError, ValueError) as exc:
            return cls(), [f"settings file at {p} could not be read "
                           f"({exc.__class__.__name__}); using all defaults"]
        return cls.from_dict_with_report(raw)

    def save(self, path: Optional[str] = None) -> str:
        p = Path(path) if path else self.default_path()
        with open(p, 'w', encoding='utf-8') as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return str(p)
