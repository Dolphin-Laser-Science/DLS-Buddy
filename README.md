# DLS Buddy

A general-purpose, instrument-agnostic Python platform for analyzing **static and
dynamic light scattering (SLS/DLS)** data.

## Features

- **Instrument-agnostic loading** with auto-detection: Brookhaven, Malvern
  Zetasizer, ALV, and a plain-text fallback.
- **DLS**: cumulants, single/double/KWW exponentials, NNLS, CONTIN, lognormal;
  Γ–q² and concentration extrapolation; multi-measurement co-plotting; replicate
  averaging.
- **SLS**: Zimm/Berry/Debye/Guinier, single-angle, calibration-free *A*₂, data masking.
- **Cross-sample**: ρ = *R*<sub>g</sub>/*R*<sub>h</sub> and *R*<sub>g</sub>–*M*<sub>w</sub> /
  *A*₂–*M*<sub>w</sub> scaling, with provenance-aware source pickers.
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

- [`1. Quickstart-Guide.pdf`](docs/1.%20Quickstart-Guide.pdf) — the common path: load → confirm
  parameters → run a DLS or SLS analysis → export.
- [`2. User-Manual.pdf`](docs/2.%20User-Manual.pdf) — the comprehensive reference: every module,
  file format, and parameter, with worked figures.
- [`3. Theory-and-Equations-Guide.pdf`](docs/3.%20Theory-and-Equations-Guide.pdf) — the physics,
  numbered equations, and literature behind each method.
- [`4. Code-Map.md`](docs/4.%20Code-Map.md) — directory and per-file tour, for reading/forking the code.
- [`5. Citation-Index.md`](docs/5.%20Citation-Index.md) — every literature source mapped to where it is used.
- [`6. Acknowledgements.md`](docs/6.%20Acknowledgements.md) — beta testers and test-data contributors.
- [`7. AI-Use-Statement.txt`](docs/7.%20AI-Use-Statement.txt) — how this project was built, in the author's words.
- [`PATCH_NOTES.md`](PATCH_NOTES.md) — what changed per release, plus known issues.
- [`CLAUDE.md`](CLAUDE.md) — architecture invariants for contributors.

## License

Licensed under **GPLv3** — see [`LICENSE`](LICENSE). © 2026 Esam Alfalah.
The Qt for Python bindings (**PySide6 / shiboken6**) are dynamically linked under
**LGPL-3.0**, which is compatible with GPLv3.
