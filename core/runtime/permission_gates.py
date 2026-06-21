"""In-app user-permission gates for sensors and the file workspace.

Thin, layering-clean wrappers over ``core.runtime.runtime_settings`` for the
permission toggles the settings UI exposes (these are the *in-app* gates, distinct
from and additional to macOS TCC). Each defaults to allowed so an unset or
unreadable setting never silently disables a sensor, and reflects user changes on
the next call (no restart). See docs/SETTINGS_WIRING_AUDIT.md.
"""
from __future__ import annotations

from core.runtime.runtime_settings import get_runtime_setting


def camera_allowed() -> bool:
    """Whether in-app camera capture is permitted (permissions.camera)."""
    return bool(get_runtime_setting("permissions.camera", True))


def screen_allowed() -> bool:
    """Whether in-app screen capture/perception is permitted (permissions.screen)."""
    return bool(get_runtime_setting("permissions.screen", True))


def workspace_files_allowed() -> bool:
    """Whether file access outside the workspace sandbox is permitted
    (permissions.files_workspace)."""
    return bool(get_runtime_setting("permissions.files_workspace", True))
