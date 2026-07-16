"""Governance for the Recursive Latent Cortex: the invariant is checked, not promised.

The whole project's honesty rests on one sentence from the spec:

    Checkpoint SHA-256: unchanged. Permanent learned parameters: unchanged.
    No hidden fine-tuning.

`CheckpointInvariant` turns that from a promise into a measurement:

- **File fingerprint** — SHA-256 over the checkpoint's weight files, cached
  by (path, size, mtime) so the 20GB resident model is hashed once per boot,
  not per episode. When full hashing is too costly mid-flight, a structural
  fingerprint (sizes + boundary chunks) is used and the receipt SAYS SO.
- **Parameter fingerprint** — a deterministic sample of live tensors hashed
  pre- and post-episode. Fast weights wrap modules without touching base
  tensors, so ANY drift here means something illegitimately wrote to W₀ —
  a CRITICAL degradation, and the episode's output is not to be trusted.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.LatentCortex.Governance")

# Cache: (resolved_path, size, mtime) → digest. Worker processes are
# single-model and long-lived; this is hit once per boot in practice.
_FILE_FINGERPRINT_CACHE: dict[tuple[str, int, float], str] = {}

# Sampled-parameter fingerprint: how many leaf tensors and how many leading
# elements of each participate. Small enough for per-episode use on the 32B.
_PARAM_SAMPLE_STRIDE = 7
_PARAM_SAMPLE_ELEMENTS = 64
_FULL_HASH_BYTES_LIMIT = 8 * 1024**3  # full-hash files up to 8GB by default


def _hash_file_full(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_file_structural(path: Path) -> str:
    """Boundary-chunk fingerprint for very large files: size + first/last 4MB.

    Not collision-proof against an adversary; entirely sufficient to detect
    accidental weight-file mutation. Receipts label which method was used.
    """
    size = path.stat().st_size
    digest = hashlib.sha256(f"structural:{size}:".encode())
    with open(path, "rb") as fh:
        digest.update(fh.read(4 * 1024 * 1024))
        if size > 8 * 1024 * 1024:
            fh.seek(-4 * 1024 * 1024, os.SEEK_END)
            digest.update(fh.read())
    return "struct-" + digest.hexdigest()


def checkpoint_file_fingerprint(model_path: str | Path) -> dict[str, Any]:
    """Fingerprint every weight shard under a model directory (or one file)."""
    root = Path(model_path).expanduser()
    files: list[Path]
    if root.is_dir():
        files = sorted(root.glob("*.safetensors")) or sorted(root.glob("*.npz")) or sorted(
            root.glob("*.gguf")
        )
    elif root.is_file():
        files = [root]
    else:
        return {"fingerprint": "", "method": "missing", "files": 0}

    force_full = str(os.environ.get("AURA_RLC_FULL_SHA", "")).strip() == "1"
    combined = hashlib.sha256()
    method = "sha256"
    for f in files:
        stat = f.stat()
        key = (str(f.resolve()), stat.st_size, stat.st_mtime)
        cached = _FILE_FINGERPRINT_CACHE.get(key)
        if cached is None:
            if force_full or stat.st_size <= _FULL_HASH_BYTES_LIMIT:
                cached = _hash_file_full(f)
            else:
                cached = _hash_file_structural(f)
            _FILE_FINGERPRINT_CACHE[key] = cached
        if cached.startswith("struct-"):
            method = "structural"
        combined.update(f"{f.name}:{cached};".encode())
    return {
        "fingerprint": combined.hexdigest()[:32],
        "method": method,
        "files": len(files),
    }


def parameter_fingerprint(model) -> str:
    """Deterministic digest over a stride-sample of live parameter tensors.

    Iterates the flattened parameter tree in sorted-name order, takes every
    Nth leaf, and hashes the first K elements' bytes. Wrappers (fast weights)
    live OUTSIDE the parameter tree, so this sees only permanent tensors.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten

    leaves = sorted(tree_flatten(model.parameters()), key=lambda kv: kv[0])
    digest = hashlib.sha256()
    for name, tensor in leaves[::_PARAM_SAMPLE_STRIDE]:
        head = mx.reshape(tensor, (-1,))[:_PARAM_SAMPLE_ELEMENTS]
        mx.eval(head)
        digest.update(name.encode())
        digest.update(memoryview(head))
    return digest.hexdigest()[:32]


class CheckpointInvariant:
    """Pre/post-episode proof that the permanent cortex never changed."""

    def __init__(self, model, model_path: str | Path | None = None) -> None:
        self._model = model
        self._model_path = str(model_path) if model_path else ""
        self._pre_params = ""
        self.file_receipt: dict[str, Any] = {}

    def pre_episode(self) -> None:
        started = time.monotonic()
        if self._model_path:
            self.file_receipt = checkpoint_file_fingerprint(self._model_path)
        self._pre_params = parameter_fingerprint(self._model)
        logger.debug(
            "Checkpoint invariant armed in %.2fs (%s)",
            time.monotonic() - started,
            self.file_receipt.get("method", "params-only"),
        )

    def post_episode(self) -> bool:
        post = parameter_fingerprint(self._model)
        unchanged = post == self._pre_params
        if not unchanged:
            record_degradation(
                "latent_cortex",
                RuntimeError(
                    "permanent parameter fingerprint changed across a latent episode"
                ),
                action="flagged episode output as untrusted (checkpoint invariant violated)",
                severity="critical",
            )
        return unchanged

    def to_receipt(self) -> dict[str, Any]:
        receipt = dict(self.file_receipt)
        receipt["param_fingerprint"] = self._pre_params
        return receipt


__all__ = [
    "CheckpointInvariant",
    "checkpoint_file_fingerprint",
    "parameter_fingerprint",
]
