"""Writable runtime workspace helpers for live environment adapters.

Environment adapters often need small sidecar files such as rc/config files,
trace handles, or driver state. Those files must not assume direct access to a
user home directory: headless runs, app sandboxes, and CI harnesses may only
permit writes inside Aura's runtime workspace or a temporary directory.

Test isolation ratchet (2026-07-23): the learning sidecars stored here are
the live organism's memory — a world model that tests were both reading
(inheriting learned risk and failing on it) and writing (test episodes
becoming real memories). Under pytest, without an explicit
``AURA_ENV_RUNTIME_DIR``, resolution therefore redirects to a per-process
temporary workspace instead of the live data directory. A test that truly
needs the live store must say so by setting ``AURA_ENV_RUNTIME_DIR`` to it
explicitly.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import tempfile
from pathlib import Path

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Environment.RuntimeWorkspace")

_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")

_pytest_isolation_root: Path | None = None
_pytest_isolation_logged = False


def _test_isolation_root() -> Path | None:
    """Per-process temp root when running under pytest without an override.

    Detection is ``"pytest" in sys.modules`` so the guard also covers calls
    made at import/collection time, before any fixture has run. The live
    runtime never imports pytest, so production resolution is unaffected.
    """
    global _pytest_isolation_root, _pytest_isolation_logged
    if "pytest" not in sys.modules:
        return None
    if _pytest_isolation_root is None:
        _pytest_isolation_root = Path(
            tempfile.mkdtemp(prefix=f"aura-test-env-runtime-{os.getpid()}-")
        )
    if not _pytest_isolation_logged:
        _pytest_isolation_logged = True
        logger.info(
            "[EnvRuntime] pytest detected without AURA_ENV_RUNTIME_DIR; "
            "isolating environment runtime state under %s (the live learning "
            "store is never read or written from tests)",
            _pytest_isolation_root,
        )
    return _pytest_isolation_root


def _safe_component(value: str, *, fallback: str) -> str:
    cleaned = _SAFE_COMPONENT.sub("_", value.strip()).strip("._-")
    return cleaned or fallback


def environment_runtime_dir(environment_id: str, *, purpose: str = "runtime") -> Path:
    """Return a writable directory for one adapter's runtime sidecar files.

    ``AURA_ENV_RUNTIME_DIR`` can override the root for harnesses. Otherwise we
    use the canonical Aura data directory, which already falls back to
    ``.aura_runtime`` inside the project when the configured home is not
    writable. If even that fails, use the system temp directory and surface a
    degradation receipt instead of failing silently.
    """
    env_name = _safe_component(environment_id, fallback="environment")
    purpose_name = _safe_component(purpose, fallback="runtime")
    override = os.environ.get("AURA_ENV_RUNTIME_DIR")

    try:
        if override:
            root = Path(override).expanduser().resolve()
        else:
            isolated = _test_isolation_root()
            if isolated is not None:
                root = isolated
            else:
                from core.config import config

                root = config.paths.data_dir / "environment_runtime"
        target = root / env_name / purpose_name
        target.mkdir(parents=True, exist_ok=True)
        return target
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation("environment_runtime_workspace", exc)
        fallback = Path(tempfile.gettempdir()) / "aura_environment_runtime" / env_name / purpose_name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def environment_runtime_file(environment_id: str, filename: str, *, purpose: str = "runtime") -> Path:
    """Return a writable sidecar file path without allowing path traversal."""
    safe_filename = _safe_component(Path(filename).name, fallback="sidecar")
    return environment_runtime_dir(environment_id, purpose=purpose) / safe_filename


__all__ = ["environment_runtime_dir", "environment_runtime_file"]
