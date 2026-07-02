"""Paths to committed ``test-data/`` files, for the real-data regression tests.

Every path here resolves inside the repository (or the active git worktree), so
the suite runs on any machine. The older standalone validators reached the SMALS
replicate set through a local-only synced folder; that set is committed under
``test-data/ALV/`` and is reached here via ``ALV_DIR`` instead.
"""
from __future__ import annotations

from pathlib import Path

# tests/fixtures/data_paths.py -> tests/fixtures -> tests -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA = REPO_ROOT / "test-data"

ALV_DIR = TEST_DATA / "ALV"
# Per-sample subfolders under ALV/ (all single unless noted).
ALV_LATEX_DIR = ALV_DIR / "Latex NPs (multi-angle)"   # ALV-7012 multi-angle (SMALS set)
ALV_PS_TOLUENE_DIR = ALV_DIR / "PS 290k in Toluene"   # ALV-7004 single-angle, PS/toluene
ALV_NAPSS_DIR = ALV_DIR / "NaPSS 40gL in Water"        # ALV-7004 single-angle, polyelectrolyte
BROOKHAVEN_DIR = TEST_DATA / "Brookhaven"
MALVERN_DIR = TEST_DATA / "Malvern"
SYNTH_CLEAN_DIR = TEST_DATA / "Synthetic Clean"
SYNTH_MESSY_DIR = TEST_DATA / "Synthetic Messy"
SYNTH_DPLS_DIR = TEST_DATA / "Synthetic DPLS"

# The 10 committed SMALS replicate .ASC files (Noisy050Latex0004_0001..0010),
# now under the per-sample Latex subfolder.
SMALS_FILES = [ALV_LATEX_DIR / f"Noisy050Latex0004_{i:04d}.ASC" for i in range(1, 11)]


def data_file(*parts: str) -> Path:
    """Absolute path to a file under ``test-data/`` (parts joined)."""
    return TEST_DATA.joinpath(*parts)


def require(path: Path) -> Path:
    """Return ``path`` or raise a clear error if the committed file is missing."""
    if not Path(path).exists():
        raise FileNotFoundError(f"expected committed test-data file is missing: {path}")
    return Path(path)
