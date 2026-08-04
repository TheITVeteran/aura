"""Unverifiable permission is not permission.

CP126 eb01cf8b on core/brain/initiative_engine.py. ``_proactivity_
suppressed_now`` is the one control governing whether Aura starts a
conversation with a person unprompted. It returned False — "not
suppressed" — when the orchestrator was absent or the lookup raised.

So the failure mode was Aura speaking to someone precisely when the
runtime could not confirm she was allowed to.

The asymmetry is not close. Staying quiet when she could have spoken costs
a missed remark. Speaking during a quiet window is the thing the control
exists to prevent, and it reaches the person.
"""
from __future__ import annotations

import time

import pytest

from core.brain.initiative_engine import _proactivity_suppressed_now
from core.container import ServiceContainer


class _Orchestrator:
    def __init__(self, quiet_until: float = 0.0) -> None:
        self._suppress_unsolicited_proactivity_until = quiet_until


def test_an_absent_orchestrator_suppresses_rather_than_permits(monkeypatch):
    monkeypatch.setattr(ServiceContainer, "get", staticmethod(lambda *a, **k: None))
    assert _proactivity_suppressed_now() is True, (
        "with no quiet-window owner, permission to speak cannot be "
        "established — and Aura spoke anyway"
    )


def test_a_raising_lookup_suppresses_rather_than_permits(monkeypatch):
    def _explode(*args, **kwargs):
        raise RuntimeError("container is wedged")

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(_explode))
    assert _proactivity_suppressed_now() is True


def test_a_malformed_quiet_window_suppresses(monkeypatch):
    """A value that cannot be read is not a value that says "go ahead"."""

    class _Bad:
        _suppress_unsolicited_proactivity_until = "not a number"

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(lambda *a, **k: _Bad()))
    assert _proactivity_suppressed_now() is True


def test_an_active_quiet_window_suppresses(monkeypatch):
    orch = _Orchestrator(quiet_until=time.time() + 300)
    monkeypatch.setattr(ServiceContainer, "get", staticmethod(lambda *a, **k: orch))
    assert _proactivity_suppressed_now() is True


def test_an_expired_quiet_window_permits(monkeypatch):
    """The control: fail-closed must not mean permanently silent."""
    orch = _Orchestrator(quiet_until=time.time() - 300)
    monkeypatch.setattr(ServiceContainer, "get", staticmethod(lambda *a, **k: orch))
    assert _proactivity_suppressed_now() is False


def test_no_quiet_window_ever_set_permits(monkeypatch):
    """A healthy runtime that simply never asked for quiet is not suppressed."""
    orch = _Orchestrator(quiet_until=0.0)
    monkeypatch.setattr(ServiceContainer, "get", staticmethod(lambda *a, **k: orch))
    assert _proactivity_suppressed_now() is False


def test_the_boundary_is_evaluated_against_the_supplied_clock(monkeypatch):
    orch = _Orchestrator(quiet_until=1000.0)
    monkeypatch.setattr(ServiceContainer, "get", staticmethod(lambda *a, **k: orch))
    assert _proactivity_suppressed_now(now=999.0) is True
    assert _proactivity_suppressed_now(now=1001.0) is False
