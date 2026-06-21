"""Layering-clean read access to user runtime settings for core subsystems.

The settings UI (``interface/routes/settings.py``) persists user toggles to
``~/.aura/data/settings/runtime.json`` (atomic write-tmp + rename). That file is
the contract boundary between the UI and the live runtime: core subsystems read
their gates from here WITHOUT importing the interface layer (which would invert
the dependency direction), so a toggle the user flips takes effect on the next
read. A small mtime cache keeps hot paths cheap and any read error falls back to
the caller's default, so a missing/corrupt file can never break a subsystem.

This is the seam that makes previously-dead settings load-bearing (see
``docs/SETTINGS_WIRING_AUDIT.md``). Wiring a dead setting becomes a one-line gate::

    from core.runtime.runtime_settings import get_runtime_setting

    if not get_runtime_setting("voice.output_enabled", True):
        return  # user disabled speech

``AURA_SETTINGS_PATH`` overrides the file location (used by tests).
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

_DEFAULT_SETTINGS_PATH = Path.home() / ".aura" / "data" / "settings" / "runtime.json"

# Reads must never raise into a subsystem gate — fall back to the default instead.
_RECOVERABLE = (OSError, ValueError, TypeError, json.JSONDecodeError)

_lock = threading.Lock()
_cache: dict[str, Any] = {}
_cache_key: tuple[str, float] | None = None


def _settings_path() -> Path:
    override = os.environ.get("AURA_SETTINGS_PATH")
    return Path(override) if override else _DEFAULT_SETTINGS_PATH


def _load_settings() -> dict[str, Any]:
    """Return the persisted settings dict, cached by (path, mtime).

    Re-reads only when the file changes, so a user toggling a setting is
    reflected on the next call without a restart.
    """
    global _cache, _cache_key
    path = _settings_path()
    try:
        mtime = path.stat().st_mtime
    except _RECOVERABLE:
        return {}
    key = (str(path), mtime)
    with _lock:
        if key != _cache_key:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                _cache = data if isinstance(data, dict) else {}
            except _RECOVERABLE:
                _cache = {}
            _cache_key = key
        return _cache


def get_runtime_setting(key: str, default: Any = None) -> Any:
    """Read a user runtime setting by dotted key, falling back to ``default``.

    Layering-clean (reads the persisted JSON the UI writes; never imports the
    interface layer). Reflects user changes on the next call. A missing key,
    missing file, or read error all yield ``default``.
    """
    value = _load_settings().get(key)
    return default if value is None else value


def clear_runtime_settings_cache() -> None:
    """Drop the in-memory cache (forces a re-read next call). For tests."""
    global _cache, _cache_key
    with _lock:
        _cache = {}
        _cache_key = None
