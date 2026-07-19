#!/usr/bin/env python3
"""Run recurrence training under an explicit, receipt-bound MLX envelope."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import runpy
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

GIB = 1024**3
SCHEMA = "aura.recurrence_training_resource_envelope.v1"


def _positive_gib(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure_mlx(mx: Any, *, memory_gb: float, cache_gb: float) -> dict[str, Any]:
    memory_bytes = int(memory_gb * GIB)
    cache_bytes = int(cache_gb * GIB)
    if cache_bytes >= memory_bytes:
        raise ValueError("MLX cache limit must be below the active memory limit")
    mx.set_memory_limit(memory_bytes)
    mx.set_cache_limit(cache_bytes)
    mx.clear_cache()
    return {
        "memory_limit_bytes": memory_bytes,
        "cache_limit_bytes": cache_bytes,
        "cache_cleared_before_model_load": True,
    }


def _write_envelope(path: Path, payload: dict[str, Any]) -> None:
    from core.runtime.atomic_writer import atomic_write_bytes, ensure_private_directory

    ensure_private_directory(path.parent)
    rendered = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    if path.exists():
        if path.is_symlink() or path.read_bytes() != rendered:
            raise RuntimeError("existing MLX resource envelope differs")
        return
    atomic_write_bytes(path, rendered)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-limit-gb", type=_positive_gib, required=True)
    parser.add_argument("--cache-limit-gb", type=_positive_gib, required=True)
    parser.add_argument("--envelope-out", required=True)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    trainer = Path(args.trainer).expanduser().resolve(strict=True)
    expected = (REPO_ROOT / "tools/recurrence_native_train_v2.py").resolve(strict=True)
    if trainer != expected or not trainer.is_file():
        raise RuntimeError("resource envelope only admits the v2 recurrence trainer")
    trainer_args = list(args.trainer_args)
    if trainer_args and trainer_args[0] == "--":
        trainer_args = trainer_args[1:]
    if not trainer_args:
        raise RuntimeError("recurrence trainer arguments are required")

    import mlx.core as mx

    limits = _configure_mlx(
        mx,
        memory_gb=args.memory_limit_gb,
        cache_gb=args.cache_limit_gb,
    )
    device_info = dict(mx.device_info())
    envelope = {
        "schema": SCHEMA,
        **limits,
        "device": {
            "architecture": device_info.get("architecture"),
            "memory_size": device_info.get("memory_size"),
            "max_recommended_working_set_size": device_info.get(
                "max_recommended_working_set_size"
            ),
        },
        "mlx_version": importlib.metadata.version("mlx"),
        "wrapper_sha256": _sha256_file(Path(__file__).resolve(strict=True)),
        "trainer_sha256": _sha256_file(trainer),
    }
    _write_envelope(Path(args.envelope_out).expanduser(), envelope)
    print(
        "MLX recurrence envelope: "
        f"active={limits['memory_limit_bytes'] // GIB}GiB "
        f"cache={limits['cache_limit_bytes'] // GIB}GiB",
        flush=True,
    )
    sys.argv = [str(trainer), *trainer_args]
    runpy.run_path(str(trainer), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
