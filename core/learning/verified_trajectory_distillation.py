"""Compile verified internal corrections into persistent recurrent adapters.

The episodic RLC can already move a failed query into a verified answer basin
with a query-scoped minimum-norm write.  Supervised text imitation does not
teach that operation: it asks the adapter to reproduce answer tokens rather
than the internal state transition that caused the successful answer.

This module fits that transition directly.  Given input activations ``X`` and
teacher-minus-incumbent projection corrections ``Y``, it solves the dual ridge
problem

    delta_W = X.T @ inv(X @ X.T + lambda I) @ Y

and computes a rank-bounded factorization without materializing a full
model-width-by-model-width matrix.  The factors use the exact orientation and
scale of ``ScopedLoRALinear`` so they can become persistent recurrent tissue.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

DISTILLATION_SCHEMA = "aura.verified_trajectory_distillation.v1"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finite_matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class DistilledTrajectoryFactors:
    """One named recurrent site's bounded low-rank correction."""

    site: str
    lora_a: np.ndarray
    lora_b: np.ndarray
    receipt: Mapping[str, Any]


def fit_verified_trajectory_factors(
    input_features: Any,
    output_corrections: Any,
    *,
    site: str,
    rank: int,
    regularization: float,
    gain: float,
    adapter_scale: float,
    normalize_corrections: bool = True,
) -> DistilledTrajectoryFactors:
    """Fit a low-rank recurrent operator to verified activation corrections.

    ``input_features`` and ``output_corrections`` have one row per private,
    independently verified teaching pair.  By default each correction is
    normalized before fitting, matching the successful episodic trajectory
    transplant and preventing long teacher traces from receiving more
    authority merely because their residual norm is larger.
    """

    if not isinstance(site, str) or not site.strip() or site != site.strip():
        raise ValueError("trajectory distillation site is invalid")
    if type(rank) is not int or rank < 1:
        raise ValueError("trajectory distillation rank must be positive")
    for value, label, lower, upper in (
        (regularization, "regularization", 0.0, 1e6),
        (gain, "gain", 0.0, 16.0),
        (adapter_scale, "adapter scale", 0.0, 4096.0),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not lower < float(value) <= upper
        ):
            raise ValueError(f"trajectory distillation {label} is invalid")

    inputs = _finite_matrix(input_features, name="input features")
    corrections = _finite_matrix(output_corrections, name="output corrections")
    if inputs.shape[0] != corrections.shape[0]:
        raise ValueError("trajectory teaching pair counts differ")
    if inputs.shape[0] < 2:
        raise ValueError("trajectory distillation requires at least two teaching pairs")
    effective_rank = min(rank, inputs.shape[0], inputs.shape[1], corrections.shape[1])

    target = corrections.copy()
    correction_norms = np.linalg.norm(target, axis=1)
    if np.any(correction_norms <= 1e-10):
        raise ValueError("trajectory correction contains a collapsed row")
    if normalize_corrections:
        target /= correction_norms[:, None]

    # Solve in sample space. This keeps memory O(n*d), not O(d_in*d_out).
    gram = inputs @ inputs.T
    system = gram + float(regularization) * np.eye(inputs.shape[0])
    try:
        dual = np.linalg.solve(system, target)
    except np.linalg.LinAlgError as exc:
        raise ValueError("trajectory ridge system is singular") from exc

    # W = P @ Q. Thin QR decompositions reduce its SVD to an n-by-n core.
    left = inputs.T
    right = dual
    q_left, r_left = np.linalg.qr(left, mode="reduced")
    q_right, r_right = np.linalg.qr(right.T, mode="reduced")
    core = r_left @ r_right.T
    u_core, singular_values, vt_core = np.linalg.svd(core, full_matrices=False)
    retained = singular_values[:effective_rank]
    if not retained.size or retained[0] <= 1e-12:
        raise ValueError("trajectory correction map collapsed")
    sqrt_s = np.sqrt(retained)
    left_factor = (q_left @ u_core[:, :effective_rank]) * sqrt_s[None, :]
    right_factor = sqrt_s[:, None] * (vt_core[:effective_rank] @ q_right.T)

    # ScopedLoRALinear emits scale * (x @ A) @ B.
    factor_scale = math.sqrt(float(gain) / float(adapter_scale))
    lora_a = (left_factor * factor_scale).astype(np.float32)
    lora_b = (right_factor * factor_scale).astype(np.float32)
    predicted = float(adapter_scale) * (inputs @ lora_a.astype(np.float64)) @ lora_b.astype(
        np.float64
    )
    residual = predicted - float(gain) * target
    target_energy = float(np.sum(np.square(float(gain) * target)))
    residual_energy = float(np.sum(np.square(residual)))
    relative_error = math.sqrt(residual_energy / max(target_energy, 1e-20))
    explained_energy = float(
        np.sum(np.square(retained)) / max(np.sum(np.square(singular_values)), 1e-20)
    )
    receipt_body = {
        "schema": DISTILLATION_SCHEMA,
        "site": site,
        "teaching_pairs": int(inputs.shape[0]),
        "input_width": int(inputs.shape[1]),
        "output_width": int(corrections.shape[1]),
        "requested_rank": rank,
        "effective_rank": effective_rank,
        "regularization": float(regularization),
        "gain": float(gain),
        "adapter_scale": float(adapter_scale),
        "corrections_normalized": bool(normalize_corrections),
        "correction_norm_min": float(np.min(correction_norms)),
        "correction_norm_max": float(np.max(correction_norms)),
        "singular_values": [float(value) for value in retained],
        "retained_operator_energy": explained_energy,
        "training_relative_error": relative_error,
        "input_features_sha256": _array_sha256(inputs),
        "output_corrections_sha256": _array_sha256(corrections),
        "lora_a_sha256": _array_sha256(lora_a),
        "lora_b_sha256": _array_sha256(lora_b),
    }
    receipt = {
        **receipt_body,
        "receipt_sha256": _canonical_sha256(receipt_body),
    }
    return DistilledTrajectoryFactors(
        site=site,
        lora_a=lora_a,
        lora_b=lora_b,
        receipt=receipt,
    )


def fit_verified_trajectory_inventory(
    teaching_pairs: Mapping[str, tuple[Any, Any]],
    *,
    rank: int,
    regularization: float,
    gain: float,
    adapter_scale: float,
) -> dict[str, DistilledTrajectoryFactors]:
    """Fit every named site and reject partial or inconsistent inventories."""

    if not isinstance(teaching_pairs, Mapping) or not teaching_pairs:
        raise ValueError("trajectory teaching inventory is empty")
    result: dict[str, DistilledTrajectoryFactors] = {}
    pair_counts: set[int] = set()
    for site in sorted(teaching_pairs):
        pair = teaching_pairs[site]
        if not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError("trajectory teaching inventory row is invalid")
        fitted = fit_verified_trajectory_factors(
            pair[0],
            pair[1],
            site=site,
            rank=rank,
            regularization=regularization,
            gain=gain,
            adapter_scale=adapter_scale,
        )
        pair_counts.add(int(fitted.receipt["teaching_pairs"]))
        result[site] = fitted
    if len(pair_counts) != 1:
        raise ValueError("trajectory teaching inventories have unequal pair counts")
    return result


def install_verified_trajectory_inventory(
    model: Any,
    inventory: Mapping[str, DistilledTrajectoryFactors],
    *,
    expected_sites: Sequence[str],
) -> dict[str, Any]:
    """Atomically install fitted factors into exact recurrence-scoped sites."""

    import mlx.core as mx

    from core.brain.llm.latent_cortex.recurrence_adapter import ScopedLoRALinear

    expected = tuple(sorted(expected_sites))
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("expected trajectory site inventory is invalid")
    if tuple(sorted(inventory)) != expected:
        raise ValueError("fitted trajectory site inventory differs from attachment")

    resolved: list[tuple[str, ScopedLoRALinear, DistilledTrajectoryFactors]] = []
    for site in expected:
        parts = site.split(".")
        if len(parts) != 5 or parts[:2] != ["model", "layers"]:
            raise ValueError(f"trajectory site path is invalid: {site}")
        try:
            layer_index = int(parts[2])
        except ValueError as exc:
            raise ValueError(f"trajectory site layer is invalid: {site}") from exc
        parent = getattr(model.model.layers[layer_index], parts[3], None)
        projection = getattr(parent, parts[4], None)
        if not isinstance(projection, ScopedLoRALinear):
            raise ValueError(f"trajectory site is not recurrence-scoped: {site}")
        factors = inventory[site]
        if (
            tuple(factors.lora_a.shape) != tuple(projection.lora_a.shape)
            or tuple(factors.lora_b.shape) != tuple(projection.lora_b.shape)
        ):
            raise ValueError(f"trajectory factor shape differs at {site}")
        resolved.append((site, projection, factors))

    snapshots = [
        (projection, projection.lora_a, projection.lora_b)
        for _, projection, _ in resolved
    ]
    try:
        for _site, projection, factors in resolved:
            projection.lora_a = mx.array(factors.lora_a).astype(projection.lora_a.dtype)
            projection.lora_b = mx.array(factors.lora_b).astype(projection.lora_b.dtype)
        mx.eval(
            *(
                tensor
                for _, projection, _ in resolved
                for tensor in (projection.lora_a, projection.lora_b)
            )
        )
    except BaseException:
        for projection, lora_a, lora_b in snapshots:
            projection.lora_a = lora_a
            projection.lora_b = lora_b
        raise

    body = {
        "schema": "aura.verified_trajectory_installation.v1",
        "sites": list(expected),
        "factor_receipt_sha256s": {
            site: str(inventory[site].receipt["receipt_sha256"]) for site in expected
        },
    }
    return {**body, "receipt_sha256": _canonical_sha256(body)}


__all__ = [
    "DISTILLATION_SCHEMA",
    "DistilledTrajectoryFactors",
    "fit_verified_trajectory_factors",
    "fit_verified_trajectory_inventory",
    "install_verified_trajectory_inventory",
]
