"""Shared fixtures for the terminal-grid environment tests."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _hermetic_environment_runtime(tmp_path, monkeypatch):
    """Isolate every test from the live organism's environment learning state.

    ``EnvironmentKernel`` wires an ``AdvancedCognitionRuntime`` whose world
    model and zero-shot transfer memory persist under the *user-global* Aura
    data directory (``~/.aura/data/environment_runtime/<env>/learning``) —
    state shared with the live instance and every checkout. Without isolation,
    tests inherit whatever risk the organism has learned for a domain (a
    contaminated model was vetoing a benign move with risk 1.0 via the
    advanced-cognition gate) and, worse, test episodes are written back into
    the live learning state. ``AURA_ENV_RUNTIME_DIR`` is the sanctioned
    harness override; pointing it at ``tmp_path`` gives each test a fresh,
    disposable learning workspace.
    """
    monkeypatch.setenv("AURA_ENV_RUNTIME_DIR", str(tmp_path / "environment_runtime"))
