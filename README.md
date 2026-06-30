# DLS Buddy

A general-purpose, instrument-agnostic Python platform for analyzing **static and
dynamic light scattering (SLS/DLS)** data from polymer solutions. Correctness-first:
every physical parameter is user-supplied, results are validated against synthetic
ground truth and real datasets, and apparent vs. thermodynamic (and calibrated vs.
uncalibrated) results are always distinguished.

## Features

- **Instrument-agnostic loading** with auto-detection: Brookhaven, Malvern
  Zetasizer, ALV `.ASC` (multi-angle), and a plain-text fallback.
- **DLS**: cumulants, single/double/KWW exponentials, NNLS, CONTIN, lognormal;
  Γ–q² and concentration extrapolation; multi-measurement co-plotting; replicate
  averaging (ISO 22412).
- **SLS**: single-point calibration, Zimm/Berry/Debye/Guinier, single-angle,
  calibration-free A₂, data masking, manual-Mw override.
- **Cross-sample**: ρ = R_g/R_h and R_g–M_w / A₂–M_w scaling, with provenance-aware
  source pickers.
- **Depolarized scattering (DPLS/DDLS)**: depolarization ratio, Cabannes split,
  rotational diffusion.
- **Utilities, Settings, Origin-compatible CSV export**, and a matplotlib plotting
  layer with light/dark themes.

## Running it

Double-click the launcher for your OS in the project folder:

- **Windows** — **`Launch DLS Buddy (Windows).bat`**
- **macOS** — **`Launch DLS Buddy (MacOS).command`** (the first time, you may need to
  right-click → **Open** once to get past macOS Gatekeeper)

On the first run the launcher creates a virtual environment and installs dependencies
automatically; subsequent launches go straight to the app. **Requires
[Python 3.13](https://www.python.org/downloads/) to be installed.**

Alternatively, from a terminal in the project root:

```powershell
# Windows (PowerShell)
py -3.13 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m gui.main
```

```bash
# macOS
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m gui.main
```

## Documentation

- **Quickstart** and **Advanced Guide** (theory + equations) — PDFs in [`docs/`](docs/).
- [`code_map.md`](docs/code_map.md) — directory and per-file tour, for reading/forking the code.
- [`code_references.md`](docs/code_references.md) — every literature source mapped to where it is used.
- [`PATCH_NOTES.md`](PATCH_NOTES.md) — what changed per release, plus known issues.
- [`CLAUDE.md`](CLAUDE.md) — architecture invariants for contributors.

## License

Licensed under **GPLv3** — see [`LICENSE`](LICENSE). © 2026 Esam Alfalah.
The Qt for Python bindings (**PySide6 / shiboken6**) are dynamically linked under
**LGPL-3.0**, which is compatible with GPLv3.
