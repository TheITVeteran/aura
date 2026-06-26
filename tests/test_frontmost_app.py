"""The frontmost-app fast path: in-process NSWorkspace, osascript only as fallback."""
from __future__ import annotations

import asyncio

import pytest

from core.perception import frontmost_app


def test_returns_none_cleanly_when_pyobjc_absent(monkeypatch):
    monkeypatch.setattr(frontmost_app, "_PYOBJC_OK", False)
    monkeypatch.setattr(frontmost_app, "_NSWorkspace", None)
    assert frontmost_app.frontmost_app_name_fast() is None


def test_uses_nsworkspace_when_present(monkeypatch):
    class _App:
        def localizedName(self):
            return "TestApp"

    class _WS:
        def frontmostApplication(self):
            return _App()

    class _NSWorkspace:
        @staticmethod
        def sharedWorkspace():
            return _WS()

    monkeypatch.setattr(frontmost_app, "_PYOBJC_OK", True)
    monkeypatch.setattr(frontmost_app, "_NSWorkspace", _NSWorkspace)
    assert frontmost_app.frontmost_app_name_fast() == "TestApp"


def test_app_focus_sensor_prefers_fast_path_without_subprocess(monkeypatch):
    """AppFocusSensor must NOT shell out when the fast path returns a name."""
    from core.body.app_focus_sensor import AppFocusSensor
    import core.body.app_focus_sensor as mod

    monkeypatch.setattr(mod, "frontmost_app_name_fast", lambda: "Safari")

    def _boom(*a, **k):
        error = AssertionError("subprocess gateway should not be called on the fast path")
        raise error

    monkeypatch.setattr(mod, "get_subprocess_gateway", _boom)

    result = asyncio.run(AppFocusSensor().read())
    assert result["active_app"] == "Safari"
    assert result["is_browser"] is True
    assert result["status"] == "healthy"
