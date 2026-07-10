"""Bounded process boundary for skill import and construction validation."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from core.runtime.subprocess_gateway import get_subprocess_gateway

_SECRET_ENV_MARKERS = ("API_KEY", "AUTH_TOKEN", "PASSWORD", "PRIVATE_KEY", "SECRET", "TOKEN")


def _probe_environment(project_root: Path, sandbox_root: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in _SECRET_ENV_MARKERS)
    }
    source_root = Path(__file__).resolve().parents[2]
    python_paths = [str(project_root), str(source_root)]
    python_paths.extend(
        path for path in env.get("PYTHONPATH", "").split(os.pathsep) if path
    )
    env.update(
        {
            "AURA_ROOT": str(sandbox_root),
            "AURA_SKILL_CATALOG_PROBE": "1",
            "AURA_TEST_MODE": "1",
            "HOME": str(sandbox_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.pathsep.join(dict.fromkeys(python_paths)),
            "TMPDIR": str(sandbox_root),
            "XDG_CACHE_HOME": str(sandbox_root),
            "XDG_CONFIG_HOME": str(sandbox_root),
            "XDG_DATA_HOME": str(sandbox_root),
        }
    )
    return env


def run_skill_catalog_probe(
    payload: dict[str, Any],
    *,
    project_root: Path,
    timeout_s: float = 45.0,
) -> dict[str, Any]:
    """Validate a catalog in an isolated child with no inherited credentials."""

    bounded_timeout = max(5.0, min(float(timeout_s), 120.0))
    root = Path(project_root).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="aura-skill-catalog-") as temporary:
        sandbox_root = Path(temporary)
        completed = get_subprocess_gateway().run(
            [sys.executable, "-m", "core.skills.catalog_probe_worker"],
            cwd=sandbox_root,
            env=_probe_environment(root, sandbox_root),
            timeout=bounded_timeout,
            read_only=True,
            capture_output=True,
            input=json.dumps(payload, sort_keys=True, separators=(",", ":")),
            source="capability_engine.skill_catalog_probe",
        )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "probe exited without diagnostics").strip()
        raise RuntimeError(f"skill catalog probe failed ({completed.returncode}): {detail[:1200]}")
    try:
        result = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"skill catalog probe emitted invalid JSON: {exc}") from exc
    if not isinstance(result, dict) or result.get("catalog_digest") != payload.get("catalog_digest"):
        raise RuntimeError("skill catalog probe response did not match the requested catalog digest")
    return result
