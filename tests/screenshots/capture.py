"""Real-platform screenshot harness for the DLS Buddy PySide6 shell.

This is a standalone visual-review aid, NOT a pytest test (it lives under
``tests/screenshots/`` which has no ``__init__.py``, and its filename does not
match pytest's ``test_*`` pattern, so pytest never collects it). It writes one
PNG per module tab so the maintainer can eyeball fonts, spacing, and theming.

IMPORTANT — run on the REAL Windows desktop, not offscreen. Qt's offscreen
platform ships no fonts, so grabbed images come out blank. Run with the project
venv activated (or point at its interpreter):

    python tests/screenshots/capture.py
    python tests/screenshots/capture.py --tab dls

Images are written to ``tests/screenshots/_out/`` (gitignored). Passing
``--tab <name>`` captures only that tab (case-insensitive match against the tab
title, e.g. ``data``, ``dls``, ``cross-sample``, ``settings``).

There are no assertions: this is a review aid, not a pass/fail gate. It can be
imported / run under ``QT_QPA_PLATFORM=offscreen`` purely as a smoke check that
it constructs and writes files (the PNGs will be blank in that case).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# --- make the repo root and the tests dir importable (this script is run as a
#     bare file, not via pytest, so pytest.ini's `pythonpath` does not apply). ---
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[2]          # tests/screenshots/capture.py -> repo root
_TESTS_DIR = _THIS.parents[1]          # .../tests  (for `import fixtures.*`)
for _p in (str(_REPO_ROOT), str(_TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_OUT_DIR = _THIS.parent / "_out"


def _redirect_settings_to_temp() -> None:
    """Point SettingsState.default_path at a throwaway temp file so running the
    harness never reads or clobbers the user's real settings.json."""
    from app.settings import SettingsState

    tmp = Path(tempfile.mkdtemp()) / "settings.json"
    SettingsState.default_path = staticmethod(lambda: tmp)


def _script_workspace(mw) -> None:
    """Inject one synthetic DLS correlogram so the tabs have something to draw.

    Uses the shared synthetic builder and the same load -> commit path the shell
    uses, then points the modules at the new measurement.
    """
    from fixtures.synthetic_dls import monomodal

    m = monomodal(30.0)
    raw = {
        "delay_times_s": [float(x) for x in m.delay_times_s],
        "correlogram": [float(x) for x in m.correlogram],
    }
    params = dict(
        polymer_name=m.polymer_name, solvent_name=m.solvent_name,
        temperature_K=m.temperature_K, angle_deg=m.angle_deg,
        wavelength_nm=m.wavelength_nm,
        solvent_refractive_index=m.solvent_refractive_index,
        viscosity_Pa_s=m.viscosity_Pa_s, concentration_g_per_mL=None,
    )
    iid = mw.controller.add_loaded("dls", raw, params)
    mw.controller.commit()
    # Refresh the navigator + point every module at the new measurement, and tick
    # it in the DLS checklist so the correlogram/distribution tabs render a curve.
    mw._refresh_sidebar()
    mw._set_current(iid)
    try:
        mw.dls_module.selection.set(iid, True)
        mw.dls_module.set_measurement(iid)
    except Exception:
        pass


def _grab(widget, path: Path) -> None:
    """Grab a widget to a PNG (best-effort; prints what it wrote)."""
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    widget.grab().save(str(path))
    print(f"  wrote {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tab", default=None,
        help="capture only this tab (case-insensitive title match, "
             "e.g. 'dls'); default captures every tab.")
    args = parser.parse_args(argv)

    _redirect_settings_to_temp()

    from PySide6 import QtWidgets
    from gui.main_window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    mw = MainWindow()
    mw.resize(1280, 760)
    _script_workspace(mw)
    # Let pending layout / draw events settle before grabbing.
    app.processEvents()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    want = args.tab.lower() if args.tab else None
    n_written = 0
    for i in range(mw.tabs.count()):
        title = mw.tabs.tabText(i)
        if want is not None and title.lower() != want:
            continue
        mw.tabs.setCurrentIndex(i)
        app.processEvents()
        safe = title.lower().replace(" ", "_").replace("/", "-")
        _grab(mw, _OUT_DIR / f"tab_{i}_{safe}.png")
        n_written += 1

    if want is not None and n_written == 0:
        print(f"No tab matched --tab {args.tab!r}. Available: "
              + ", ".join(mw.tabs.tabText(i) for i in range(mw.tabs.count())))
        mw.close()
        return 2

    print(f"Done: {n_written} screenshot(s) in {_OUT_DIR}")
    print("NOTE: run on the real Windows desktop for real fonts — offscreen "
          "produces blank images.")
    mw.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
