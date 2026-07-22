"""Measured model-artifact identity for admission, QoS, and proof decisions.

CP126 semantic review flagged a whole class of defects rooted in one habit:
model footprint, minimum-headroom, deadline, cache-residency, and identity
decisions were derived from SPOOFABLE PATH SUBSTRINGS ("72b", "cortex",
"zenith"). A renamed heavy checkpoint inherited light-model budgets; an
unrelated path containing "32b" inherited a 20GB reservation.

Every sharded MLX artifact already carries machine-readable evidence:

- ``model.safetensors.index.json`` → ``metadata.total_parameters`` and
  ``metadata.total_size`` (exact weight bytes),
- ``config.json`` → architecture shape (a parameter count can be estimated
  when the index metadata is absent),
- the safetensors file listing itself (names + sizes).

This module turns that evidence into a cached :class:`ModelArtifactProfile`.
Classification prefers measured evidence and only falls back to declared
path naming when the artifact is absent (tests and pre-download paths use
fake names) — and the profile SAYS which evidence produced it, so receipts
can distinguish measured truth from naming convention.

The fingerprint is a cheap identity binding (config bytes + index metadata
+ weight-file listing), NOT a weight hash — hashing 20GB per admission
check is not viable. It changes whenever the artifact's declared shape,
quantization, shard layout, or file sizes change, which is the tamper
surface the admission and proof lanes actually consult.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Parameter-count boundaries for the runtime's weight classes. The classes
# mirror the lanes the runtime actually provisions for (solver/cortex/
# brainstem/reflex); boundaries sit between real model families rather than
# on top of them.
_CLASS_BOUNDARIES: tuple[tuple[float, str], ...] = (
    (55e9, "72b"),
    (20e9, "32b"),
    (10e9, "14b"),
    (4e9, "7b"),
    (0.0, "small"),
)

_HEAVY_CLASSES = frozenset({"72b", "32b"})

_72B_PATH_TOKENS = ("72b", "solver")
_32B_PATH_TOKENS = ("32b", "cortex", "zenith")
_14B_PATH_TOKENS = ("14b", "24b", "40b")
_7B_PATH_TOKENS = ("7b", "brainstem")


@dataclass(frozen=True)
class ModelArtifactProfile:
    path: str
    exists: bool
    weight_bytes: int
    total_parameters: int
    quantization_bits: int
    size_class: str  # "72b" | "32b" | "14b" | "7b" | "small" | "unknown"
    evidence: str  # "index_metadata" | "config_estimate" | "file_sizes" | "path_tokens" | "absent"
    fingerprint: str

    @property
    def weight_gb(self) -> float:
        return float(self.weight_bytes) / float(1024**3)

    @property
    def is_heavy(self) -> bool:
        return self.size_class in _HEAVY_CLASSES

    @property
    def measured(self) -> bool:
        """True when the class came from artifact evidence, not naming."""
        return self.evidence in {"index_metadata", "config_estimate", "file_sizes"}


_PROFILE_CACHE: dict[str, tuple[tuple[float, float], ModelArtifactProfile]] = {}
_PROFILE_CACHE_LOCK = threading.Lock()
_PROFILE_CACHE_MAX = 32


def _class_for_parameters(total_parameters: float) -> str:
    for boundary, name in _CLASS_BOUNDARIES:
        if total_parameters >= boundary and total_parameters > 0:
            return name
    return "unknown"


def _class_from_path_tokens(model_path: str) -> str:
    lowered = str(model_path or "").lower()
    if any(token in lowered for token in _72B_PATH_TOKENS):
        return "72b"
    if any(token in lowered for token in _32B_PATH_TOKENS):
        return "32b"
    if any(token in lowered for token in _14B_PATH_TOKENS):
        return "14b"
    if any(token in lowered for token in _7B_PATH_TOKENS):
        return "7b"
    return "small"


def _estimate_parameters_from_config(config: dict) -> int:
    """Coarse transformer parameter estimate from architecture shape.

    Good to well within one class boundary for the dense decoder families
    this runtime serves (embedding + per-layer attention/MLP terms).
    """
    try:
        hidden = float(config.get("hidden_size") or 0)
        layers = float(config.get("num_hidden_layers") or 0)
        inter = float(config.get("intermediate_size") or 0)
        vocab = float(config.get("vocab_size") or 0)
        heads = float(config.get("num_attention_heads") or 0)
        kv_heads = float(config.get("num_key_value_heads") or heads or 1)
    except (TypeError, ValueError):
        return 0
    if hidden <= 0 or layers <= 0:
        return 0
    if inter <= 0:
        inter = hidden * 4
    head_dim = hidden / heads if heads > 0 else hidden
    # Attention: Q + output are hidden×hidden; K/V shrink under GQA.
    attn = 2.0 * hidden * hidden + 2.0 * hidden * (kv_heads * head_dim)
    mlp = 3.0 * hidden * inter  # gate/up/down
    embed = 2.0 * vocab * hidden  # embed + lm_head (upper bound if tied)
    estimate = embed + layers * (attn + mlp)
    if not math.isfinite(estimate) or estimate <= 0:
        return 0
    return int(estimate)


def _quantization_bits(config: dict) -> int:
    quant = config.get("quantization")
    if isinstance(quant, dict):
        try:
            bits = int(quant.get("bits") or 0)
        except (TypeError, ValueError):
            return 0
        return bits if 0 < bits <= 32 else 0
    return 0


def _cache_key_stamp(config_path: Path, index_path: Path) -> tuple[float, float]:
    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    return (_mtime(config_path), _mtime(index_path))


def get_model_artifact_profile(model_path: str) -> ModelArtifactProfile:
    """Return the (cached) measured profile for a model artifact path."""
    resolved = str(model_path or "")
    try:
        root = Path(resolved).expanduser()
        real = root.resolve() if root.exists() else root
    except OSError:
        root = Path(resolved)
        real = root
    cache_id = str(real)
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    stamp = _cache_key_stamp(config_path, index_path)
    with _PROFILE_CACHE_LOCK:
        cached = _PROFILE_CACHE.get(cache_id)
        if cached is not None and cached[0] == stamp:
            return cached[1]

    profile = _build_profile(resolved, root, config_path, index_path)
    with _PROFILE_CACHE_LOCK:
        if len(_PROFILE_CACHE) >= _PROFILE_CACHE_MAX:
            _PROFILE_CACHE.pop(next(iter(_PROFILE_CACHE)), None)
        _PROFILE_CACHE[cache_id] = (stamp, profile)
    return profile


def _build_profile(
    resolved: str,
    root: Path,
    config_path: Path,
    index_path: Path,
) -> ModelArtifactProfile:
    exists = False
    try:
        exists = root.exists()
    except OSError:
        exists = False
    if not exists or not root.is_dir():
        # Single-file or absent artifacts: declared naming is the only
        # evidence available. Say so.
        size_class = _class_from_path_tokens(resolved)
        weight_bytes = 0
        if exists:
            try:
                weight_bytes = int(root.stat().st_size)
            except OSError:
                weight_bytes = 0
        return ModelArtifactProfile(
            path=resolved,
            exists=exists,
            weight_bytes=weight_bytes,
            total_parameters=0,
            quantization_bits=0,
            size_class=size_class,
            evidence="path_tokens" if exists else "absent",
            fingerprint="",
        )

    config: dict = {}
    config_bytes = b""
    try:
        config_bytes = config_path.read_bytes()
        parsed = json.loads(config_bytes)
        if isinstance(parsed, dict):
            config = parsed
    except (OSError, ValueError):
        config = {}

    index_metadata: dict = {}
    try:
        index = json.loads(index_path.read_text())
        if isinstance(index, dict) and isinstance(index.get("metadata"), dict):
            index_metadata = index["metadata"]
    except (OSError, ValueError):
        index_metadata = {}

    weight_files: list[tuple[str, int]] = []
    summed_weight_bytes = 0
    try:
        for child in sorted(root.glob("*.safetensors")):
            try:
                size = int(child.stat().st_size)
            except OSError:
                continue
            weight_files.append((child.name, size))
            summed_weight_bytes += size
    except OSError:
        pass

    total_parameters = 0
    weight_bytes = 0
    evidence = "path_tokens"
    try:
        total_parameters = int(index_metadata.get("total_parameters") or 0)
        weight_bytes = int(index_metadata.get("total_size") or 0)
    except (TypeError, ValueError):
        total_parameters = 0
        weight_bytes = 0
    if total_parameters > 0:
        evidence = "index_metadata"
    else:
        total_parameters = _estimate_parameters_from_config(config)
        if total_parameters > 0:
            evidence = "config_estimate"
    if weight_bytes <= 0:
        weight_bytes = summed_weight_bytes
        if total_parameters <= 0 and weight_bytes > 0:
            evidence = "file_sizes"

    if total_parameters > 0:
        size_class = _class_for_parameters(float(total_parameters))
    elif weight_bytes > 0:
        # Weight bytes alone: infer through 4-bit density (~0.55 byte/param
        # after metadata) as a conservative bound, then classify.
        approx_params = float(weight_bytes) / 0.55
        size_class = _class_for_parameters(approx_params)
    else:
        size_class = _class_from_path_tokens(resolved)
        evidence = "path_tokens"

    digest = hashlib.sha256()
    digest.update(config_bytes)
    digest.update(
        json.dumps(index_metadata, sort_keys=True, default=str).encode("utf-8")
    )
    for name, size in weight_files:
        digest.update(f"{name}:{size}".encode("utf-8"))
    fingerprint = digest.hexdigest() if (config_bytes or weight_files) else ""

    profile = ModelArtifactProfile(
        path=resolved,
        exists=True,
        weight_bytes=weight_bytes,
        total_parameters=total_parameters,
        quantization_bits=_quantization_bits(config),
        size_class=size_class,
        evidence=evidence,
        fingerprint=fingerprint,
    )
    declared = _class_from_path_tokens(resolved)
    if profile.measured and declared != profile.size_class and declared != "small":
        # A measured artifact whose naming DISAGREES with its contents is
        # exactly the spoof/rename hazard this module exists for — surface
        # it instead of silently trusting either side.
        logger.warning(
            "Model artifact %s measures as %s-class (%.1fB params, %.1fGB) "
            "but is NAMED %s-class; measured evidence wins.",
            os.path.basename(resolved),
            profile.size_class,
            profile.total_parameters / 1e9,
            profile.weight_gb,
            declared,
        )
    return profile


def model_size_class(model_path: str) -> str:
    """Convenience: the measured (or declared-fallback) weight class."""
    return get_model_artifact_profile(model_path).size_class


def model_is_heavy(model_path: str) -> bool:
    return get_model_artifact_profile(model_path).is_heavy
