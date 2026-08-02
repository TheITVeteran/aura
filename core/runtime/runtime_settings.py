"""Layering-clean read access to the versioned runtime settings control plane.

The authenticated settings API commits a strict, revisioned envelope at
``~/.aura/data/settings/runtime.json``. Core subsystems read that contract here
without importing the interface layer. Nanosecond mtime and size caching keeps
hot paths cheap while still reflecting a committed change on the next read.

A never-created file represents first boot and uses each caller's documented
default. Corruption, incompatible state, permission loss, or deletion after a
valid read instead activates conservative governance overrides, so losing the
settings plane cannot silently relax containment or external-access policy.

See ``docs/SETTINGS_WIRING_AUDIT.md`` for the complete owner/evidence matrix::

    from core.runtime.runtime_settings import get_runtime_setting

    if not get_runtime_setting("voice.output_enabled", True):
        return  # user disabled speech

``AURA_SETTINGS_PATH`` overrides the file location (used by tests).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from core.runtime.settings_schema import (
    SCHEMA,
    SETTINGS_SCHEMA_NAME,
    SETTINGS_SCHEMA_VERSION,
    migrated_settings_snapshot,
)
from core.runtime.state_ownership import state_root

_DEFAULT_SETTINGS_PATH = state_root() / "data" / "settings" / "runtime.json"
logger = logging.getLogger("Aura.RuntimeSettings")

# Reads must never raise into a subsystem gate — fall back to the default instead.
_RECOVERABLE = (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError)

_lock = threading.Lock()
_cache: dict[str, Any] = {}
_cache_key: tuple[str, int, int] | None = None
_cache_error_key: tuple[str, str] | None = None

_PROTECTED_DEFAULTS = {
    definition.key: definition.default
    for definition in SCHEMA
    if not definition.mutable
}

_FAIL_CLOSED_OVERRIDES: dict[str, Any] = {
    "autonomy.self_modification": "blocked",
    "governance.approval_mode": "all",
    "model.cloud_fallback_enabled": False,
    "permissions.camera": False,
    "permissions.files_workspace": False,
    "permissions.screen": False,
    "privacy.mode": "isolated",
    "safety.safe_mode": True,
}

_DESTRUCTIVE_EFFECT_SCOPES = frozenset(
    {
        "desktop_file_io",
        "external_io",
        "foreground_browser_dialogue",
        "foreground_desktop_control",
        "model_weight_mutation",
        "privileged_mutation",
        "read_write_artifacts",
        "state_mutation",
        "subprocess",
        "unknown",
        "workspace_file_io",
    }
)


def _settings_path() -> Path:
    override = os.environ.get("AURA_SETTINGS_PATH")
    return Path(override) if override else _DEFAULT_SETTINGS_PATH


def _failed_settings_snapshot(path: Path, error: BaseException) -> dict[str, Any]:
    global _cache_error_key
    error_key = (str(path), f"{type(error).__name__}:{error}")
    with _lock:
        base = dict(_cache) if _cache_key and _cache_key[0] == str(path) else {}
        base.update(_FAIL_CLOSED_OVERRIDES)
        if _cache_error_key != error_key:
            logger.error(
                "Runtime settings unavailable; conservative overrides active: %s",
                error_key[1],
            )
            _cache_error_key = error_key
        return base


def _load_settings() -> dict[str, Any]:
    """Return the persisted settings dict, cached by (path, mtime).

    Re-reads only when the file changes, so a user toggling a setting is
    reflected on the next call without a restart.
    """
    global _cache, _cache_error_key, _cache_key
    path = _settings_path()
    try:
        stat_result = path.stat()
    except FileNotFoundError as exc:
        if _cache_key and _cache_key[0] == str(path):
            return _failed_settings_snapshot(path, exc)
        return {}
    except _RECOVERABLE as exc:
        return _failed_settings_snapshot(path, exc)
    key = (
        str(path),
        int(
            getattr(
                stat_result,
                "st_mtime_ns",
                int(stat_result.st_mtime * 1_000_000_000),
            )
        ),
        int(stat_result.st_size),
    )
    with _lock:
        if key != _cache_key:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise TypeError("settings state must be a JSON object")
                if "schema" in data or "schema_version" in data:
                    if data.get("schema") != SETTINGS_SCHEMA_NAME:
                        raise ValueError("settings schema is incompatible")
                    version = data.get("schema_version")
                    if (
                        isinstance(version, bool)
                        or not isinstance(version, int)
                        or version > SETTINGS_SCHEMA_VERSION
                    ):
                        raise ValueError("settings schema version is incompatible")
                    if version == SETTINGS_SCHEMA_VERSION:
                        from core.runtime.settings_control_plane import (
                            RuntimeSettingsStore,
                        )

                        verified = RuntimeSettingsStore(path).snapshot(refresh=True)
                        _cache = dict(verified.values)
                    else:
                        _cache, _unknown = migrated_settings_snapshot(
                            data.get("payload")
                        )
                else:
                    # Legacy flat-map compatibility. The control plane migrates
                    # it into the versioned envelope on the first mutation.
                    _cache, _unknown = migrated_settings_snapshot(data)
                _cache_error_key = None
            except (*_RECOVERABLE, KeyError) as exc:
                base = dict(_cache) if _cache_key and _cache_key[0] == str(path) else {}
                base.update(_FAIL_CLOSED_OVERRIDES)
                _cache = base
                error_key = (str(path), f"{type(exc).__name__}:{exc}")
                if _cache_error_key != error_key:
                    logger.error(
                        "Runtime settings invalid; conservative overrides active: %s",
                        error_key[1],
                    )
                    _cache_error_key = error_key
            _cache_key = key
        return _cache


def get_runtime_setting(key: str, default: Any = None) -> Any:
    """Read a user runtime setting by dotted key, falling back to ``default``.

    Layering-clean (reads the persisted JSON the UI writes; never imports the
    interface layer). Reflects user changes on the next call. A missing key,
    missing file, or read error all yield ``default``.
    """
    settings = _load_settings()
    if key in _PROTECTED_DEFAULTS:
        return _PROTECTED_DEFAULTS[key]
    value = settings.get(key)
    return default if value is None else value


def autonomous_actions_admitted(
    source: Any,
    context: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Preserve Aura's agency invariant across every normal action source.

    Consequential actions remain governed by safe mode, Constitution, Will,
    standing authority, capability tokens, Conscience, and effect receipts.
    A persisted preference is not allowed to silently turn cognition into a
    non-agentic runtime.
    """

    del source, context
    return True, "autonomous_agency_invariant"


def runtime_approval_mode() -> str:
    mode = str(
        get_runtime_setting("governance.approval_mode", "destructive")
        or "destructive"
    ).strip().lower()
    return mode if mode in {"all", "destructive", "none"} else "destructive"


def additional_confirmation_required(
    *,
    risk_level: Any,
    effect_scope: Any,
) -> tuple[bool, str]:
    """Return only the user-selected confirmation overlay.

    This never weakens Constitution, Will, standing authority, capability
    tokens, or Conscience. ``none`` removes only this additional prompt layer.
    """

    mode = runtime_approval_mode()
    if mode == "none":
        return False, "approval_mode_none"
    if mode == "all":
        return True, "approval_mode_all"
    risk = str(risk_level or "").strip().lower()
    scope = str(effect_scope or "").strip().lower()
    required = scope in _DESTRUCTIVE_EFFECT_SCOPES or risk == "critical"
    return required, (
        "approval_mode_destructive"
        if required
        else "approval_mode_destructive_non_destructive_action"
    )


def clear_runtime_settings_cache() -> None:
    """Drop the in-memory cache (forces a re-read next call). For tests."""
    global _cache, _cache_error_key, _cache_key
    with _lock:
        _cache = {}
        _cache_key = None
        _cache_error_key = None
