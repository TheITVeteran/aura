"""Shared helpers for safer macOS desktop boot behavior."""

from __future__ import annotations

import importlib
import os
import platform
import threading
from collections.abc import Mapping
from typing import Any

_GIB = 1024**3
SAFE_BOOT_MLX_MEMORY_CAP_GB = 34.0
SAFE_BOOT_PROCESS_RSS_CAP_GB = 40.0
_INPROCESS_MLX_LOCK = threading.Lock()
_INPROCESS_MLX_STATE: dict[str, Any] = {
    "configured": False,
    "device": "unknown",
    "reason": "uninitialized",
}


def env_flag_enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    try:
        return float(env.get(name, str(default)))
    except (TypeError, ValueError, OverflowError):
        return default


def _unsafe_memory_limits_allowed(env: Mapping[str, str]) -> bool:
    return env_flag_enabled(env.get("AURA_ALLOW_UNSAFE_MEMORY_LIMITS"))


def desktop_safe_boot_enabled(env: Mapping[str, str] | None = None) -> bool:
    env = env or os.environ
    explicit = str(env.get("AURA_SAFE_BOOT_DESKTOP", "")).strip().lower()
    if explicit in {"1", "true", "yes", "on"}:
        return True
    if explicit in {"0", "false", "no", "off"}:
        return False
    return env_flag_enabled(env.get("AURA_LAUNCHED_FROM_APP"))


def compute_mlx_cache_limit(total_ram_bytes: int, env: Mapping[str, str] | None = None) -> int:
    env = env or os.environ
    total_ram_bytes = max(int(total_ram_bytes), 8 * _GIB)

    if desktop_safe_boot_enabled(env):
        ratio = _env_float(env, "AURA_SAFE_BOOT_METAL_CACHE_RATIO", 0.16)
        hard_cap_gb = _env_float(env, "AURA_SAFE_BOOT_METAL_CACHE_CAP_GB", 10.0)
        floor_gb = _env_float(env, "AURA_SAFE_BOOT_METAL_CACHE_FLOOR_GB", 4.0)
        limit = int(total_ram_bytes * ratio)
        limit = min(limit, int(hard_cap_gb * _GIB))
        limit = max(int(floor_gb * _GIB), limit)
        if not _unsafe_memory_limits_allowed(env):
            limit = min(limit, 10 * _GIB)
        return limit

    ratio = _env_float(env, "AURA_METAL_CACHE_RATIO", 0.75)
    limit = int(total_ram_bytes * ratio)
    hard_cap_gb = _env_float(env, "AURA_METAL_CACHE_CAP_GB", 0.0)
    if hard_cap_gb > 0:
        limit = min(limit, int(hard_cap_gb * _GIB))
    return max(8 * _GIB, limit)


def compute_mlx_memory_limit(total_ram_bytes: int, env: Mapping[str, str] | None = None) -> int:
    """Return the active MLX memory ceiling for model/KV allocations."""

    env = env or os.environ
    total_ram_bytes = max(int(total_ram_bytes), 8 * _GIB)
    safe_boot = desktop_safe_boot_enabled(env)
    unsafe_allowed = _unsafe_memory_limits_allowed(env)
    configured = str(env.get("AURA_MLX_MEMORY_LIMIT_GB", "") or "").strip()
    if configured:
        try:
            configured_gb = float(configured)
        except (TypeError, ValueError, OverflowError):
            configured_gb = 0.0
        if configured_gb > 0.0:
            configured_limit = int(configured_gb * _GIB)
            if safe_boot and not unsafe_allowed:
                safe_cap_gb = min(
                    _env_float(env, "AURA_SAFE_BOOT_MLX_MEMORY_CAP_GB", SAFE_BOOT_MLX_MEMORY_CAP_GB),
                    SAFE_BOOT_MLX_MEMORY_CAP_GB,
                )
                return min(configured_limit, int(safe_cap_gb * _GIB))
            return configured_limit

    if safe_boot:
        ratio = _env_float(env, "AURA_SAFE_BOOT_MLX_MEMORY_RATIO", 0.54)
        hard_cap_gb = _env_float(env, "AURA_SAFE_BOOT_MLX_MEMORY_CAP_GB", SAFE_BOOT_MLX_MEMORY_CAP_GB)
        floor_gb = _env_float(env, "AURA_SAFE_BOOT_MLX_MEMORY_FLOOR_GB", 18.0)
        limit = min(int(total_ram_bytes * ratio), int(hard_cap_gb * _GIB))
        limit = max(int(floor_gb * _GIB), limit)
        if not unsafe_allowed:
            limit = min(limit, int(SAFE_BOOT_MLX_MEMORY_CAP_GB * _GIB))
        return limit

    ratio = _env_float(env, "AURA_MLX_MEMORY_RATIO", 0.72)
    limit = int(total_ram_bytes * ratio)
    hard_cap_gb = _env_float(env, "AURA_MLX_MEMORY_CAP_GB", 0.0)
    if hard_cap_gb > 0:
        limit = min(limit, int(hard_cap_gb * _GIB))
    return max(8 * _GIB, limit)


def compute_process_rss_limit(total_ram_bytes: int, env: Mapping[str, str] | None = None) -> int:
    """Return the process-tree RSS guard used by desktop safe boot.

    This is intentionally lower than the external sentinel kill ceiling. The
    in-process guard should refuse/recycle before the out-of-process sentinel
    has to SIGKILL Aura to protect the host.
    """

    env = env or os.environ
    total_ram_bytes = max(int(total_ram_bytes), 8 * _GIB)
    safe_boot = desktop_safe_boot_enabled(env)
    unsafe_allowed = _unsafe_memory_limits_allowed(env)
    configured = str(env.get("AURA_PROCESS_RSS_LIMIT_GB", "") or "").strip()
    if configured:
        try:
            configured_gb = float(configured)
        except (TypeError, ValueError, OverflowError):
            configured_gb = 0.0
        if configured_gb > 0.0:
            configured_limit = int(configured_gb * _GIB)
            if safe_boot and not unsafe_allowed:
                safe_cap_gb = min(
                    _env_float(env, "AURA_SAFE_BOOT_PROCESS_RSS_CAP_GB", SAFE_BOOT_PROCESS_RSS_CAP_GB),
                    SAFE_BOOT_PROCESS_RSS_CAP_GB,
                )
                return min(configured_limit, int(safe_cap_gb * _GIB))
            return configured_limit

    if safe_boot:
        ratio = _env_float(env, "AURA_SAFE_BOOT_PROCESS_RSS_RATIO", 0.62)
        hard_cap_gb = _env_float(env, "AURA_SAFE_BOOT_PROCESS_RSS_CAP_GB", SAFE_BOOT_PROCESS_RSS_CAP_GB)
        floor_gb = _env_float(env, "AURA_SAFE_BOOT_PROCESS_RSS_FLOOR_GB", 24.0)
        limit = min(int(total_ram_bytes * ratio), int(hard_cap_gb * _GIB))
        limit = max(int(floor_gb * _GIB), limit)
        if not unsafe_allowed:
            limit = min(limit, int(SAFE_BOOT_PROCESS_RSS_CAP_GB * _GIB))
        return limit

    ratio = _env_float(env, "AURA_PROCESS_RSS_RATIO", 0.56)
    hard_cap_gb = _env_float(env, "AURA_PROCESS_RSS_CAP_GB", 38.0)
    floor_gb = _env_float(env, "AURA_PROCESS_RSS_FLOOR_GB", 30.0)
    limit = min(int(total_ram_bytes * ratio), int(hard_cap_gb * _GIB))
    return max(int(floor_gb * _GIB), limit)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _macos_major_version(version: str | None = None) -> int:
    release = str(version or platform.mac_ver()[0] or "").strip()
    if not release:
        return 0
    head = release.split(".", 1)[0].strip()
    try:
        return int(head)
    except ValueError:
        return 0


def inprocess_mlx_metal_enabled(
    env: Mapping[str, str] | None = None,
    *,
    platform_name: str | None = None,
    mac_version: str | None = None,
) -> tuple[bool, str]:
    env = env or os.environ
    platform_name = str(platform_name or os.sys.platform).lower()

    if _truthy(env.get("AURA_FORCE_INPROCESS_MLX_METAL")) or _truthy(
        env.get("AURA_ALLOW_UNSAFE_INPROCESS_MLX_METAL")
    ):
        return True, "forced"

    if _truthy(env.get("AURA_DISABLE_INPROCESS_MLX_METAL")):
        return False, "env_disabled"

    if desktop_safe_boot_enabled(env):
        return False, "desktop_safe_boot"

    if platform_name == "darwin" and _macos_major_version(mac_version) >= 26:
        return False, "macos26_guard"

    return True, "enabled"


def configure_inprocess_mlx_runtime(
    env: Mapping[str, str] | None = None,
    *,
    platform_name: str | None = None,
    mac_version: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    enabled, reason = inprocess_mlx_metal_enabled(
        env,
        platform_name=platform_name,
        mac_version=mac_version,
    )

    desired_device = "metal" if enabled else "cpu"
    with _INPROCESS_MLX_LOCK:
        if (
            not force
            and _INPROCESS_MLX_STATE["configured"]
            and _INPROCESS_MLX_STATE["device"] == desired_device
            and _INPROCESS_MLX_STATE["reason"] == reason
        ):
            return dict(_INPROCESS_MLX_STATE)

        if not enabled:
            _INPROCESS_MLX_STATE.update(
                {
                    "configured": True,
                    "device": "cpu",
                    "reason": reason,
                }
            )
            return dict(_INPROCESS_MLX_STATE)

        try:
            importlib.import_module("mlx.core")
        except (ImportError, AttributeError, RuntimeError):
            _INPROCESS_MLX_STATE.update(
                {
                    "configured": True,
                    "device": "unavailable",
                    "reason": f"{reason}:mlx_unavailable",
                }
            )
            return dict(_INPROCESS_MLX_STATE)

        _INPROCESS_MLX_STATE.update(
            {
                "configured": True,
                "device": "metal",
                "reason": reason,
            }
        )
        return dict(_INPROCESS_MLX_STATE)
