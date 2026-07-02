"""Shared pytest fixtures for the DLS Buddy regression suite.

Centralises the setup that the old standalone validation scripts each
repeated by hand: the offscreen-Qt application, and a Controller whose
``settings.json`` is redirected to a temp dir so a test run never touches the
user's real settings file.

Path handling lives in ``pytest.ini`` (``pythonpath = . tests``): the repo root
is importable for ``analysis``/``physics``/``core``/``app``, and the ``tests``
dir is importable for ``from fixtures.<name> import ...``.
"""
from __future__ import annotations

import os
import warnings

import pytest

# The forward-model builders and real-data numerics emit expected ill-conditioned
# warnings (empty-bin chi^2, non-positive Berry points); pytest.ini already filters
# the common numeric ones. Keep the rest visible.
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero")


# --- Hypothesis (property-based testing) -------------------------------------
# Registered once here so every property test shares sane defaults. Numeric fits
# and Monte-Carlo builders routinely exceed Hypothesis's default 200 ms per-example
# deadline, so it is disabled (a slow example is not a failure here). Pick a deeper
# sweep with HYPOTHESIS_PROFILE=thorough. Hypothesis is a dev-only dependency, so
# its absence must never break collection.
try:
    from hypothesis import HealthCheck, settings

    settings.register_profile(
        "dev", max_examples=25, deadline=None,
        suppress_health_check=[HealthCheck.too_slow])
    settings.register_profile(
        "thorough", max_examples=300, deadline=None,
        suppress_health_check=[HealthCheck.too_slow])
    settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
except ImportError:
    pass


@pytest.fixture(scope="session")
def qt_app():
    """A single offscreen QApplication for the whole GUI-smoke tier.

    Skips the test if PySide6 is not installed. Sets the offscreen platform
    BEFORE importing PySide6, and auto-confirms the modal dialogs that would
    otherwise block headless (QMessageBox.question -> Yes) — the established
    headless convention.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # Auto-confirm the switch-guard / overwrite dialogs so headless never hangs.
    QtWidgets.QMessageBox.question = staticmethod(
        lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes)
    yield app


@pytest.fixture
def temp_settings_path(tmp_path, monkeypatch):
    """Redirect SettingsState.default_path to a per-test temp file.

    Returns the temp path. Any Controller / SettingsState built after this
    fixture runs persists there, never to the repo-root settings.json.
    """
    from app.settings import SettingsState

    path = tmp_path / "settings.json"
    monkeypatch.setattr(SettingsState, "default_path",
                        staticmethod(lambda: path))
    return path


@pytest.fixture
def controller(temp_settings_path):
    """A fresh Controller with settings redirected to a temp dir."""
    from app.controller import Controller

    return Controller()


@pytest.fixture(scope="session")
def smals():
    """Replicate-averaged SMALS correlograms (or None if data absent).

    Session-scoped: the 10-file load + per-angle averaging runs once.
    """
    from fixtures.synthetic_dls import smals_replicate_averaged

    return smals_replicate_averaged()
