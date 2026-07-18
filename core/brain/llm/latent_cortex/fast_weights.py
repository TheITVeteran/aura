"""Episode-scoped fast weights: the checkpoint writes temporary synapses.

During one reasoning episode the effective weights may become

    W_t = W₀ + s·U Vᵀ        (rank-r, per selected window-layer linear)

with three hard guarantees the spec demands and this module PROVES:

1. **Identity at attach.** V is zero-initialized, so the instant a wrapper
   attaches, the model's function is bit-for-bit unchanged (same guarantee
   the expert-adapter seam relies on: a LoRA with B=0 is behaviorally inert).
2. **Erase is proven, not assumed.** ``detach`` restores the original module
   objects, and ``prove_erase`` re-runs a caller-supplied probe and asserts
   exact output equality with the pre-attach baseline. The receipt carries
   the verdict.
3. **No persistent learning without governance.** A ΔW that earns its keep
   is EXPORTED to the governed consolidation queue for the existing LoRA
   compounding loop (with its regression gates); it never mutates W₀ here.

U is seeded deterministically from the episode's workspace statistics — the
latent state literally parameterizes the temporary synapses — and both U and
V are then optimized against the episode's proxy/verifier loss with all base
weights frozen (grads flow only to U, V).
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.brain.llm.latent_cortex.types import ComputeBudget, FastWeightsConfig

logger = logging.getLogger("Aura.LatentCortex.FastWeights")
FAST_WEIGHT_OPTIMIZER = "rms_normalized_sgd_backtracking_v1"

_TARGET_ATTRS = {
    "o_proj": ("self_attn", "o_proj"),
    "down_proj": ("mlp", "down_proj"),
}


def _linear_dimension_source(module):
    """Find the projection that owns shape metadata without bypassing wrappers.

    MLX LoRA modules intentionally expose their frozen base projection as
    ``.linear`` and do not duplicate ``.weight``.  Fast weights must inspect
    that nested projection for dimensions while continuing to invoke the
    outer module, otherwise attaching an episodic delta would either crash or
    silently bypass the durable adapter.
    """
    current = module
    seen: set[int] = set()
    while True:
        identity = id(current)
        if identity in seen:
            raise TypeError("linear wrapper cycle while resolving dimensions")
        seen.add(identity)
        if hasattr(current, "weight"):
            return current
        nested = getattr(current, "linear", None)
        if nested is None:
            raise TypeError(
                f"{type(module).__name__} has no weight-bearing linear projection"
            )
        current = nested


def _linear_dims(module) -> tuple[int, int]:
    """Return ``(out_features, in_features)`` for wrapped or bare linears."""
    source = _linear_dimension_source(module)
    weight = source.weight
    if getattr(weight, "ndim", None) != 2:
        raise TypeError("linear projection weight must be two-dimensional")
    if hasattr(source, "scales"):  # QuantizedLinear packs weights
        out_features = weight.shape[0]
        bits = int(getattr(source, "bits", 4))
        if bits <= 0 or 32 % bits:
            raise TypeError(f"unsupported quantized linear bit width: {bits}")
        in_features = weight.shape[1] * (32 // bits)
        return int(out_features), int(in_features)
    return int(weight.shape[0]), int(weight.shape[1])


class EpisodicDeltaLinear:
    """y = base(x) + s·((x Vᵀ) Uᵀ) — a temporary synapse over a frozen linear.

    Not an ``nn.Module`` on purpose: keeping U/V as plain attributes outside
    the model's parameter tree means nothing about the model's trainable
    state, freeze bookkeeping, or serialization changes while a wrapper is
    attached. Gradients reach U/V functionally (see ``optimize``).
    """

    def __init__(
        self,
        base,
        rank: int,
        scale: float,
        seed_stat: float,
        tag: str,
        seed_vectors=None,
    ) -> None:
        import mlx.core as mx

        self.base = base
        self.scale = float(scale)
        self.tag = tag
        out_features, in_features = _linear_dims(base)
        seed = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "big")
        key = mx.random.key(seed)
        # U uses dimension-normalized LoRA-style scale, modulated by the
        # workspace statistic; V remains exactly zero, so attachment is still
        # bit-identical while the first V gradient stays above fp16/bf16 noise.
        seed_scale = min(1.0, max(0.1, abs(float(seed_stat))))
        init_std = seed_scale / math.sqrt(max(1, out_features))
        self.U = mx.random.normal((out_features, rank), key=key) * init_std
        # Retrieval-to-fast-weight compilation: refined retrieval slot
        # states become the leading columns of U, so the adaptation
        # SUBSPACE is spanned by retrieved knowledge — the temporary
        # synapses can write those directions into the residual stream once
        # V learns when to fire. V stays zero, so attach remains exactly
        # identity and the erase proof is untouched.
        self.retrieval_seeded_columns = 0
        if (
            seed_vectors is not None
            and getattr(seed_vectors, "ndim", 0) == 2
            and int(seed_vectors.shape[1]) == out_features
        ):
            k = min(int(rank), int(seed_vectors.shape[0]))
            if k > 0:
                target_norm = init_std * math.sqrt(max(1, out_features))
                columns = []
                for j in range(k):
                    vector = seed_vectors[j].astype(self.U.dtype)
                    norm = mx.maximum(mx.linalg.norm(vector), 1e-6)
                    columns.append(vector / norm * target_norm)
                seeded = mx.stack(columns, axis=1)
                self.U = mx.concatenate([seeded, self.U[:, k:]], axis=1)
                self.retrieval_seeded_columns = k
        self.V = mx.zeros((rank, in_features))
        mx.eval(self.U, self.V)

    def __call__(self, x):
        delta = (x @ self.V.T) @ self.U.T
        return self.base(x) + self.scale * delta


@dataclass
class FastWeightHandle:
    layer_index: int
    parent: Any
    attr: str
    original: Any
    wrapper: EpisodicDeltaLinear


@dataclass
class FastWeightsLifecycle:
    """Auditable state machine: ATTACHED → (OPTIMIZED) → ERASED, with proof."""

    attached_at: float = 0.0
    layers: list[int] = field(default_factory=list)
    target: str = ""
    rank: int = 0
    optimizer: str = FAST_WEIGHT_OPTIMIZER
    optimization_attempts: int = 0
    optimized_steps: int = 0
    rejected_steps: int = 0
    line_search_backtracks: int = 0
    budget_exhausted: bool = False
    detach_conflicts: int = 0
    canary_rescales: int = 0
    canary_erased: bool = False
    retrieval_seeded_columns: int = 0
    loss_trail: list[float] = field(default_factory=list)
    gradient_global_norm_trail: list[float] = field(default_factory=list)
    accepted_step_sizes: list[float] = field(default_factory=list)
    erased: bool = False
    erase_proven: bool | None = None
    exported: bool = False

    def to_receipt(self) -> dict[str, Any]:
        return {
            "attached_at": self.attached_at,
            "layers": list(self.layers),
            "target": self.target,
            "rank": self.rank,
            "optimizer": self.optimizer,
            "optimization_attempts": self.optimization_attempts,
            "optimized_steps": self.optimized_steps,
            "rejected_steps": self.rejected_steps,
            "line_search_backtracks": self.line_search_backtracks,
            "budget_exhausted": self.budget_exhausted,
            "detach_conflicts": self.detach_conflicts,
            "canary_rescales": self.canary_rescales,
            "canary_erased": self.canary_erased,
            "retrieval_seeded_columns": self.retrieval_seeded_columns,
            "loss_trail": [round(v, 6) for v in self.loss_trail],
            "gradient_global_norm_trail": [
                round(v, 6) for v in self.gradient_global_norm_trail
            ],
            "accepted_step_sizes": [
                round(v, 12) for v in self.accepted_step_sizes
            ],
            "erased": self.erased,
            "erase_proven": self.erase_proven,
            "exported": self.exported,
        }


class EpisodicFastWeights:
    """Owns the full lifecycle of one episode's temporary synapses."""

    def __init__(self, config: FastWeightsConfig) -> None:
        self.config = config
        self.handles: list[FastWeightHandle] = []
        self.lifecycle = FastWeightsLifecycle()
        self.last_export_receipt: dict[str, Any] | None = None

    # ── Attach / detach ─────────────────────────────────────────────────
    def attach(
        self,
        inner_model,
        layer_range: tuple[int, int],
        *,
        seed_stat: float,
        episode_id: str,
        seed_vectors=None,
    ) -> int:
        """Wrap the target linear in up to ``max_wrapped_layers`` window layers."""
        if self.handles:
            raise RuntimeError("fast weights already attached — one episode at a time")
        parent_attr, leaf_attr = _TARGET_ATTRS[self.config.target]
        start, end = layer_range
        candidates = list(range(start, end))[: max(1, self.config.max_wrapped_layers)]
        attached = False
        try:
            for i in candidates:
                layer = inner_model.layers[i]
                parent = getattr(layer, parent_attr)
                original = getattr(parent, leaf_attr)
                wrapper = EpisodicDeltaLinear(
                    original,
                    rank=self.config.rank,
                    scale=self.config.scale,
                    seed_stat=seed_stat,
                    tag=f"{episode_id}:{i}:{self.config.target}",
                    seed_vectors=seed_vectors,
                )
                setattr(parent, leaf_attr, wrapper)
                self.handles.append(
                    FastWeightHandle(
                        layer_index=i,
                        parent=parent,
                        attr=leaf_attr,
                        original=original,
                        wrapper=wrapper,
                    )
                )
            attached = True
        finally:
            if not attached:
                # Attachment is a transaction. A malformed layer in the
                # middle must not leave earlier layers wrapped, including when
                # control exits through a non-Exception base error.
                self.detach()
        self.lifecycle.attached_at = time.time()
        self.lifecycle.layers = [h.layer_index for h in self.handles]
        self.lifecycle.target = self.config.target
        self.lifecycle.rank = self.config.rank
        self.lifecycle.retrieval_seeded_columns = max(
            (h.wrapper.retrieval_seeded_columns for h in self.handles),
            default=0,
        )
        return len(self.handles)

    def detach(self) -> int:
        """Restore every original module object. Idempotent and conflict-aware."""
        restored = 0
        conflicts = 0
        remaining: list[FastWeightHandle] = []
        for handle in reversed(self.handles):
            current = getattr(handle.parent, handle.attr)
            if current is handle.original:
                continue
            if current is not handle.wrapper:
                # Another writer touched a module owned by this episode. We
                # still restore W0, but the conflict invalidates the proof.
                conflicts += 1
            try:
                setattr(handle.parent, handle.attr, handle.original)
                restored += 1
            except (AttributeError, RuntimeError, TypeError):
                remaining.append(handle)
        self.handles = list(reversed(remaining))
        self.lifecycle.detach_conflicts += conflicts
        self.lifecycle.erased = not self.handles
        return restored

    def rescale(self, factor: float) -> float:
        """Multiply every wrapper's scale — the canary ladder's step-down.

        Scale lives outside U/V, so a rescale needs no re-optimization and no
        new forward pass; the next canary measurement decides whether the
        weaker ΔW is now behaviorally safe.
        """
        if (
            isinstance(factor, bool)
            or not isinstance(factor, (int, float))
            or not math.isfinite(float(factor))
            or not 0.0 < float(factor) < 1.0
        ):
            raise ValueError("fast-weight rescale factor must be inside (0, 1)")
        if not self.handles:
            raise RuntimeError("fast-weight rescale requires attached wrappers")
        for handle in self.handles:
            handle.wrapper.scale *= float(factor)
        self.lifecycle.canary_rescales += 1
        return float(self.handles[0].wrapper.scale)

    def effective_delta_metrics(self) -> dict[str, Any]:
        """Measure the exact effective-delta RMS without materializing U@V.T.

        For D = s*U@V.T, ||D||_F^2 is
        s^2*trace((U.T@U)*(V@V.T)). Both Gram matrices are rank-by-rank, so
        this structural safety check stays cheap even on the resident 32B.
        """

        import mlx.core as mx

        rows: list[dict[str, Any]] = []
        all_finite = bool(self.handles)
        max_rms = 0.0
        for handle in self.handles:
            wrapper = handle.wrapper
            gram_u = wrapper.U.T @ wrapper.U
            gram_v = wrapper.V @ wrapper.V.T
            mx.eval(gram_u, gram_v)
            squared_frobenius = float(
                (float(wrapper.scale) ** 2) * mx.sum(gram_u * gram_v.T)
            )
            finite = math.isfinite(squared_frobenius) and squared_frobenius >= 0.0
            all_finite = all_finite and finite
            if finite:
                out_features = int(wrapper.U.shape[0])
                in_features = int(wrapper.V.shape[1])
                frobenius = math.sqrt(squared_frobenius)
                rms = frobenius / math.sqrt(max(1, out_features * in_features))
                max_rms = max(max_rms, rms)
            else:
                frobenius = math.inf
                rms = math.inf
            rows.append(
                {
                    "layer": int(handle.layer_index),
                    "scale": round(float(wrapper.scale), 12),
                    "effective_delta_frobenius": (
                        round(frobenius, 12) if math.isfinite(frobenius) else None
                    ),
                    "effective_delta_rms": (
                        round(rms, 12) if math.isfinite(rms) else None
                    ),
                    "finite": finite,
                }
            )
        return {
            "schema": "aura.fast_weight_delta_magnitude.v1",
            "finite": all_finite,
            "layer_count": len(rows),
            "max_effective_delta_rms": (
                round(max_rms, 12) if all_finite else None
            ),
            "layers": rows,
        }

    def canary_erase(self) -> None:
        """Erase ΔW because the protected battery regressed under it.

        The episode continues on base weights with its refined latent state
        intact; the lifecycle records that the canaries — not cleanup —
        removed the adaptation, and consolidation export is off the table
        because the post-detach snapshot is deliberately never taken.
        """
        self.detach()
        self.lifecycle.canary_erased = True

    def prove_erase(self, probe_fn: Callable[[], Any], baseline) -> bool:
        """Assert the model's function is EXACTLY the pre-attach baseline."""
        import mlx.core as mx

        if self.handles or not self.lifecycle.erased:
            raise RuntimeError("prove_erase called while fast weights still attached")
        after = probe_fn()
        proven = self.lifecycle.detach_conflicts == 0 and bool(
            mx.allclose(after, baseline, atol=0.0, rtol=0.0)
        )
        self.lifecycle.erase_proven = proven
        if not proven:
            from core.runtime.errors import record_degradation

            record_degradation(
                "latent_cortex",
                RuntimeError("fast-weight erase failed probe equality"),
                action="flagged episode receipt and refused consolidation export",
            )
        return proven

    # ── Optimization (grads to U/V only; base frozen by construction) ──
    def optimize(
        self,
        loss_fn: Callable[[], Any],
        *,
        steps: int | None = None,
        budget: ComputeBudget | None = None,
        layer_apps_per_forward: int = 0,
        reserve_layer_apps: int = 0,
    ) -> None:
        """Functional gradient steps on every wrapper's (U, V).

        ``loss_fn`` closes over the model (with wrappers attached) and
        returns a scalar. We lift the wrapper params into an explicit list,
        rebind them inside the traced function, and step along a per-tensor
        RMS-preconditioned descent direction with bounded backtracking. This
        keeps a resident-scale update numerically visible without allowing the
        number of adapter elements to inflate its RMS magnitude. A candidate is
        retained only when it improves the proxy beyond floating-point noise;
        base weights never appear as grad targets.
        """
        import mlx.core as mx

        if not self.handles:
            return
        n_steps = steps if steps is not None else self.config.opt_steps
        if type(n_steps) is not int or n_steps < 0:
            raise ValueError("fast-weight optimization steps must be a non-negative integer")
        if (
            isinstance(layer_apps_per_forward, bool)
            or not isinstance(layer_apps_per_forward, int)
            or layer_apps_per_forward < 0
        ):
            raise ValueError("layer_apps_per_forward must be a non-negative integer")
        if (
            isinstance(reserve_layer_apps, bool)
            or not isinstance(reserve_layer_apps, int)
            or reserve_layer_apps < 0
        ):
            raise ValueError("reserve_layer_apps must be a non-negative integer")
        if budget is not None and layer_apps_per_forward <= 0:
            raise ValueError(
                "budgeted fast-weight optimization requires a positive forward cost"
            )

        def bind_params(params) -> None:
            parameter_pairs = zip(params[0::2], params[1::2], strict=True)
            for h, (u, v) in zip(self.handles, parameter_pairs, strict=True):
                h.wrapper.U = u
                h.wrapper.V = v

        def with_params(params):
            bind_params(params)
            return loss_fn()

        params = []
        for h in self.handles:
            params.extend([h.wrapper.U, h.wrapper.V])

        grad_fn = mx.value_and_grad(with_params)
        for _ in range(n_steps):
            gradient_cost = layer_apps_per_forward * 3
            if budget is not None and (
                budget.exhausted
                or gradient_cost + reserve_layer_apps > budget.remaining_layer_apps
            ):
                self.lifecycle.budget_exhausted = True
                break
            if budget is not None:
                budget.charge_layer_apps(gradient_cost)
            self.lifecycle.optimization_attempts += 1
            value, grads = grad_fn(params)
            current_loss = float(value)
            if not self.lifecycle.loss_trail:
                self.lifecycle.loss_trail.append(current_loss)
            flat = mx.concatenate([mx.reshape(g, (-1,)) for g in grads])
            gnorm = mx.maximum(mx.linalg.norm(flat), 1e-12)
            gnorm_value = float(gnorm)
            if not math.isfinite(current_loss) or not math.isfinite(gnorm_value):
                self.lifecycle.rejected_steps += 1
                break
            self.lifecycle.gradient_global_norm_trail.append(gnorm_value)
            directions = []
            for grad in grads:
                grad_rms = mx.maximum(mx.sqrt(mx.mean(mx.square(grad))), 1e-12)
                directions.append(mx.clip(grad / grad_rms, -8.0, 8.0))
            step_size = float(self.config.lr)
            accepted = False
            for backtrack in range(12):
                candidate_cost = layer_apps_per_forward
                if budget is not None and (
                    budget.exhausted
                    or candidate_cost + reserve_layer_apps > budget.remaining_layer_apps
                ):
                    self.lifecycle.budget_exhausted = True
                    break
                if budget is not None:
                    budget.charge_layer_apps(candidate_cost)
                candidate = [
                    parameter - step_size * direction
                    for parameter, direction in zip(params, directions, strict=True)
                ]
                try:
                    candidate_value = with_params(candidate)
                    mx.eval(candidate_value, *candidate)
                except BaseException:  # noqa: BLE001 - always restore bound params on interruption
                    bind_params(params)
                    raise
                candidate_loss = float(candidate_value)
                minimum_improvement = max(1e-6, abs(current_loss) * 1e-7)
                if (
                    math.isfinite(candidate_loss)
                    and current_loss - candidate_loss >= minimum_improvement
                ):
                    params = candidate
                    self.lifecycle.loss_trail.append(candidate_loss)
                    self.lifecycle.optimized_steps += 1
                    self.lifecycle.line_search_backtracks += backtrack
                    self.lifecycle.accepted_step_sizes.append(step_size)
                    accepted = True
                    break
                step_size *= 0.5
            if not accepted:
                self.lifecycle.rejected_steps += 1
                bind_params(params)
                break
        bind_params(params)  # leave the best params installed without another forward pass

    # ── Consolidation handoff ───────────────────────────────────────────
    def export_candidate(
        self,
        queue_dir: Path | str,
        *,
        episode_id: str,
        evidence: dict[str, Any],
    ) -> Path | None:
        """Serialize ΔW + evidence into the governed consolidation queue.

        Refused unless erase was PROVEN — a candidate from an episode whose
        cleanup could not be verified is not trustworthy evidence. The
        permanent-learning decision belongs to the LoRA compounding loop's
        regression gates, never to this module.
        """
        import numpy as np

        if self.lifecycle.erase_proven is not True:
            logger.info("Consolidation export refused: erase not proven for %s", episode_id)
            return None
        if not getattr(self, "_exported_handles", None):
            logger.info(
                "Consolidation export refused: no snapshot taken before detach for %s",
                episode_id,
            )
            return None
        self.last_export_receipt = None
        try:
            from core.brain.llm.latent_cortex.persistence import (
                get_latent_cortex_persistence,
            )

            target_dir = (Path(queue_dir).expanduser() / episode_id).resolve()
            arrays: dict[str, Any] = {}
            for handle in self._exported_handles:
                arrays[f"layer{handle['layer_index']}_U"] = handle["U"]
                arrays[f"layer{handle['layer_index']}_V"] = handle["V"]
            buffer = io.BytesIO()
            np.savez(buffer, **arrays)
            delta_payload = buffer.getvalue()
            delta_sha256 = hashlib.sha256(delta_payload).hexdigest()
            lifecycle_receipt = self.lifecycle.to_receipt()
            lifecycle_receipt["exported"] = True
            payload = {
                "schema": "aura.latent_cortex.fast_weight_candidate.v1",
                "episode_id": episode_id,
                "created_at": time.time(),
                "target": self.lifecycle.target,
                "rank": self.lifecycle.rank,
                "layers": self.lifecycle.layers,
                "evidence": evidence,
                "lifecycle": lifecycle_receipt,
                "artifacts": {
                    "delta_weights.npz": {
                        "sha256": delta_sha256,
                        "size_bytes": len(delta_payload),
                    }
                },
            }
            evidence_payload = json.dumps(payload, indent=1, sort_keys=True).encode("utf-8")
            evidence_sha256 = hashlib.sha256(evidence_payload).hexdigest()
            receipt = get_latent_cortex_persistence().publish_fast_weight_candidate(
                target_dir,
                delta_payload=delta_payload,
                evidence_payload=evidence_payload,
            )
            committed_hashes = dict(receipt.sha256)
            expected_hashes = {
                str(target_dir / "delta_weights.npz"): delta_sha256,
                str(target_dir / "evidence.json"): evidence_sha256,
            }
            if set(receipt.paths) != set(expected_hashes) or committed_hashes != expected_hashes:
                raise RuntimeError("fast-weight batch receipt does not match payloads")
            self.lifecycle.exported = True
            self.last_export_receipt = {
                "transaction_id": receipt.transaction_id,
                "paths": list(receipt.paths),
                "sha256": committed_hashes,
            }
            return target_dir
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            from core.runtime.errors import record_degradation

            record_degradation(
                "latent_cortex",
                exc,
                action="dropped consolidation candidate after queue export failed",
            )
            return None

    def snapshot_for_export(self) -> None:
        """Capture U/V as numpy BEFORE detach (arrays outlive the wrappers)."""
        import numpy as np

        self._exported_handles = [
            {
                "layer_index": h.layer_index,
                "U": np.array(h.wrapper.U),
                "V": np.array(h.wrapper.V),
            }
            for h in self.handles
        ]


__all__ = [
    "EpisodicDeltaLinear",
    "EpisodicFastWeights",
    "FAST_WEIGHT_OPTIMIZER",
    "FastWeightHandle",
    "FastWeightsLifecycle",
]
