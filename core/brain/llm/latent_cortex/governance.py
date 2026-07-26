"""Governance for the Recursive Latent Cortex: the invariant is checked, not promised.

The whole project's honesty rests on one sentence from the spec:

    Checkpoint SHA-256: unchanged. Permanent learned parameters: unchanged.
    No hidden fine-tuning.

`CheckpointInvariant` turns that from a promise into a measurement:

- **File fingerprint** — SHA-256 over the checkpoint's weight files, cached
  by (path, size, mtime) so the 20GB resident model is hashed once per boot,
  not per episode. When full hashing is too costly mid-flight, a structural
  fingerprint (sizes + boundary chunks) is used and the receipt SAYS SO.
- **Parameter canary** — a fixed leading/middle/trailing sample of a declared
  stride over the live parameter tree, measured pre/post.
- **Adapted-layer identity** — every permanent parameter byte in every layer
  temporary fast weights can touch, measured pre/post.
- **Serving-stack identity** — tokenizer artifacts/runtime tokenizer,
  adapter order and bytes, and quantization configuration, measured pre/post.
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
        "fingerprint": combined.hexdigest(),
        "method": method,
        "files": len(files),
    }


def parameter_fingerprint(model) -> str:
    """Deterministic digest over a stride-sample of live parameter tensors.

    Iterates the flattened parameter tree in sorted-name order, takes every
    Nth leaf, and hashes fixed leading/middle/trailing elements. Wrappers
    (fast weights) live outside the permanent parameter tree.
    """
    from core.brain.llm.latent_cortex.runtime_integrity import (
        parameter_canary_fingerprint,
    )

    return parameter_canary_fingerprint(
        model,
        stride=_PARAM_SAMPLE_STRIDE,
        elements_per_tensor=_PARAM_SAMPLE_ELEMENTS,
    )["sha256"]


class CheckpointInvariant:
    """Pre/post-episode proof that the permanent cortex never changed."""

    def __init__(
        self,
        model,
        model_path: str | Path | None = None,
        *,
        tokenizer: Any = None,
        adapted_layer_indices: tuple[int, ...] = (),
        adapted_target: str = "o_proj",
    ) -> None:
        self._model = model
        self._model_path = str(model_path) if model_path else ""
        self._tokenizer = tokenizer
        self._adapted_layer_indices = adapted_layer_indices
        self._adapted_target = adapted_target
        self._pre_params = ""
        self._pre_parameter_measurement: dict[str, Any] = {}
        self._post_parameter_measurement: dict[str, Any] = {}
        self._pre_adapted_layers: dict[str, Any] = {}
        self._post_adapted_layers: dict[str, Any] = {}
        self._pre_serving_stack: dict[str, Any] = {}
        self._post_serving_stack: dict[str, Any] = {}
        self.file_receipt: dict[str, Any] = {}
        self.runtime_receipt: dict[str, Any] = {}

    def pre_episode(self) -> None:
        started = time.monotonic()
        if self._model_path:
            self.file_receipt = checkpoint_file_fingerprint(self._model_path)
        from core.brain.llm.latent_cortex.runtime_integrity import (
            adapted_layer_fingerprint,
            parameter_canary_fingerprint,
            serving_stack_measurement,
        )

        self._pre_parameter_measurement = parameter_canary_fingerprint(
            self._model,
            stride=_PARAM_SAMPLE_STRIDE,
            elements_per_tensor=_PARAM_SAMPLE_ELEMENTS,
        )
        self._pre_params = self._pre_parameter_measurement["sha256"]
        self._pre_adapted_layers = adapted_layer_fingerprint(
            self._model,
            layer_indices=self._adapted_layer_indices,
            target=self._adapted_target,
        )
        self._pre_serving_stack = serving_stack_measurement(
            self._model,
            self._tokenizer,
            self._model_path,
        )
        logger.debug(
            "Checkpoint invariant armed in %.2fs (%s)",
            time.monotonic() - started,
            self.file_receipt.get("method", "params-only"),
        )

    def post_episode(self, receipt: Any | None = None) -> bool:
        from core.brain.llm.latent_cortex.runtime_integrity import (
            adapted_layer_fingerprint,
            build_engine_runtime_integrity,
            parameter_canary_fingerprint,
            serving_stack_measurement,
        )

        self._post_parameter_measurement = parameter_canary_fingerprint(
            self._model,
            stride=_PARAM_SAMPLE_STRIDE,
            elements_per_tensor=_PARAM_SAMPLE_ELEMENTS,
        )
        post = self._post_parameter_measurement["sha256"]
        self._post_adapted_layers = adapted_layer_fingerprint(
            self._model,
            layer_indices=self._adapted_layer_indices,
            target=self._adapted_target,
        )
        self._post_serving_stack = serving_stack_measurement(
            self._model,
            self._tokenizer,
            self._model_path,
        )
        unchanged = post == self._pre_params
        if receipt is not None:
            self.runtime_receipt = build_engine_runtime_integrity(
                episode_id=str(getattr(receipt, "episode_id", "") or ""),
                input_tokens_sha256=str(
                    getattr(receipt, "input_tokens_sha256", "") or ""
                ),
                checkpoint={
                    **self.file_receipt,
                    "required": bool(self._model_path),
                },
                parameters_before=self._pre_parameter_measurement,
                parameters_after=self._post_parameter_measurement,
                adapted_layers_before=self._pre_adapted_layers,
                adapted_layers_after=self._post_adapted_layers,
                serving_stack_before=self._pre_serving_stack,
                serving_stack_after=self._post_serving_stack,
                fast_weights_applied=(
                    getattr(receipt, "fast_weights_applied", False) is True
                ),
                fast_weight_learning=getattr(
                    receipt,
                    "fast_weight_learning",
                    {},
                ),
                fast_weight_cleanup=getattr(
                    receipt,
                    "fast_weight_cleanup",
                    {},
                ),
                probe_cache=getattr(receipt, "probe_cache", {}),
            )
            receipt.runtime_integrity = dict(self.runtime_receipt)
            from core.brain.llm.latent_cortex.types import (
                WeightIntegrityProof,
            )

            erase = self.runtime_receipt["fast_weight_erase"]
            canary_before = (
                erase["pre_probe_sha256"]
                if erase["required"]
                else self._pre_adapted_layers["sha256"]
            )
            canary_after = (
                erase["post_probe_sha256"]
                if erase["required"]
                else self._post_adapted_layers["sha256"]
            )
            receipt.weight_integrity = WeightIntegrityProof(
                algorithm="sha256",
                version=2,
                params_before=self._pre_params,
                params_after=post,
                canary_before=canary_before,
                canary_after=canary_after,
                erased_layer_ids=list(erase["layer_ids"]),
                unavailable_reason=(
                    ""
                    if self.runtime_receipt["verdict"][
                        "engine_measurements_complete"
                    ]
                    else ",".join(
                        self.runtime_receipt["verdict"]["reasons"]
                    )
                ),
            )
            # Direct engine callers consume this compatibility return. It
            # must reflect the complete measured engine state, not only the
            # sampled parameter canary.
            unchanged = bool(
                self.runtime_receipt["verdict"][
                    "engine_measurements_complete"
                ]
            )
        if not unchanged:
            reasons = self.runtime_receipt.get("verdict", {}).get(
                "reasons",
                [],
            )
            record_degradation(
                "latent_cortex",
                RuntimeError(
                    "runtime integrity changed across a latent episode"
                    + (f":{','.join(reasons)}" if reasons else "")
                ),
                action=(
                    "refused output because checkpoint, serving stack, cache, "
                    "or temporary-weight cleanup was not intact"
                ),
                severity="critical",
            )
        return unchanged

    def to_receipt(self) -> dict[str, Any]:
        receipt = dict(self.file_receipt)
        receipt["param_fingerprint"] = self._pre_params
        receipt["runtime_integrity"] = dict(self.runtime_receipt)
        return receipt


__all__ = [
    "CheckpointInvariant",
    "checkpoint_file_fingerprint",
    "parameter_fingerprint",
]
