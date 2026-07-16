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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from core.brain.llm.latent_cortex.types import FastWeightsConfig

logger = logging.getLogger("Aura.LatentCortex.FastWeights")

_TARGET_ATTRS = {
    "o_proj": ("self_attn", "o_proj"),
    "down_proj": ("mlp", "down_proj"),
}


def _linear_dims(module) -> tuple[int, int]:
    """(out_features, in_features) for Linear or QuantizedLinear."""
    weight = module.weight
    if hasattr(module, "scales"):  # QuantizedLinear packs weights
        out_features = weight.shape[0]
        bits = int(getattr(module, "bits", 4))
        in_features = weight.shape[1] * (32 // bits)
        return out_features, in_features
    return int(weight.shape[0]), int(weight.shape[1])


class EpisodicDeltaLinear:
    """y = base(x) + s·((x Vᵀ) Uᵀ) — a temporary synapse over a frozen linear.

    Not an ``nn.Module`` on purpose: keeping U/V as plain attributes outside
    the model's parameter tree means nothing about the model's trainable
    state, freeze bookkeeping, or serialization changes while a wrapper is
    attached. Gradients reach U/V functionally (see ``optimize``).
    """

    def __init__(self, base, rank: int, scale: float, seed_stat: float, tag: str) -> None:
        import mlx.core as mx

        self.base = base
        self.scale = float(scale)
        self.tag = tag
        out_features, in_features = _linear_dims(base)
        seed = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "big")
        key = mx.random.key(seed)
        # U seeded from workspace statistics (scaled small); V zero ⇒ identity.
        self.U = mx.random.normal((out_features, rank), key=key) * (
            0.01 * max(1e-3, float(seed_stat))
        )
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
    optimized_steps: int = 0
    loss_trail: list[float] = field(default_factory=list)
    erased: bool = False
    erase_proven: bool | None = None
    exported: bool = False

    def to_receipt(self) -> dict[str, Any]:
        return {
            "attached_at": self.attached_at,
            "layers": list(self.layers),
            "target": self.target,
            "rank": self.rank,
            "optimized_steps": self.optimized_steps,
            "loss_trail": [round(v, 6) for v in self.loss_trail],
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

    # ── Attach / detach ─────────────────────────────────────────────────
    def attach(
        self,
        inner_model,
        layer_range: tuple[int, int],
        *,
        seed_stat: float,
        episode_id: str,
    ) -> int:
        """Wrap the target linear in up to ``max_wrapped_layers`` window layers."""
        if self.handles:
            raise RuntimeError("fast weights already attached — one episode at a time")
        parent_attr, leaf_attr = _TARGET_ATTRS[self.config.target]
        start, end = layer_range
        candidates = list(range(start, end))[: max(1, self.config.max_wrapped_layers)]
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
            )
            setattr(parent, leaf_attr, wrapper)
            self.handles.append(
                FastWeightHandle(
                    layer_index=i, parent=parent, attr=leaf_attr,
                    original=original, wrapper=wrapper,
                )
            )
        self.lifecycle.attached_at = time.time()
        self.lifecycle.layers = [h.layer_index for h in self.handles]
        self.lifecycle.target = self.config.target
        self.lifecycle.rank = self.config.rank
        return len(self.handles)

    def detach(self) -> int:
        """Restore every original module object. Idempotent."""
        restored = 0
        for handle in self.handles:
            if getattr(handle.parent, handle.attr) is handle.wrapper:
                setattr(handle.parent, handle.attr, handle.original)
                restored += 1
        self.handles = []
        self.lifecycle.erased = True
        return restored

    def prove_erase(self, probe_fn: Callable[[], Any], baseline) -> bool:
        """Assert the model's function is EXACTLY the pre-attach baseline."""
        import mlx.core as mx

        if self.handles:
            raise RuntimeError("prove_erase called while fast weights still attached")
        after = probe_fn()
        proven = bool(mx.allclose(after, baseline, atol=0.0, rtol=0.0))
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
    def optimize(self, loss_fn: Callable[[], Any], *, steps: int | None = None) -> None:
        """Functional gradient steps on every wrapper's (U, V).

        ``loss_fn`` closes over the model (with wrappers attached) and
        returns a scalar. We lift the wrapper params into an explicit list,
        rebind them inside the traced function, and step with plain SGD +
        global-norm clipping. Base weights never appear as grad targets.
        """
        import mlx.core as mx

        if not self.handles:
            return
        n_steps = steps if steps is not None else self.config.opt_steps

        def with_params(params):
            for h, (u, v) in zip(self.handles, zip(params[0::2], params[1::2])):
                h.wrapper.U = u
                h.wrapper.V = v
            return loss_fn()

        params = []
        for h in self.handles:
            params.extend([h.wrapper.U, h.wrapper.V])

        grad_fn = mx.value_and_grad(with_params)
        for _ in range(max(0, int(n_steps))):
            value, grads = grad_fn(params)
            self.lifecycle.loss_trail.append(float(value))
            flat = mx.concatenate([mx.reshape(g, (-1,)) for g in grads])
            gnorm = mx.maximum(mx.linalg.norm(flat), 1e-12)
            clip = mx.minimum(1.0, 1.0 / gnorm)
            params = [p - self.config.lr * g * clip for p, g in zip(params, grads)]
            mx.eval(*params)
            self.lifecycle.optimized_steps += 1
        with_params(params)  # leave the best params installed

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
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            target_dir = Path(queue_dir) / episode_id
            arrays: dict[str, Any] = {}
            for h_idx, handle in enumerate(self._exported_handles):
                arrays[f"layer{handle['layer_index']}_U"] = handle["U"]
                arrays[f"layer{handle['layer_index']}_V"] = handle["V"]
            buffer = io.BytesIO()
            np.savez(buffer, **arrays)
            payload = {
                "episode_id": episode_id,
                "created_at": time.time(),
                "target": self.lifecycle.target,
                "rank": self.lifecycle.rank,
                "layers": self.lifecycle.layers,
                "evidence": evidence,
                "lifecycle": self.lifecycle.to_receipt(),
            }
            gateway = get_file_write_gateway()
            with local_internal_governed_scope("latent_cortex_consolidation"):
                gateway.ensure_directory(
                    target_dir, source="latent_cortex.fast_weights"
                )
                gateway.write_bytes(
                    target_dir / "delta_weights.npz",
                    buffer.getvalue(),
                    source="latent_cortex.fast_weights",
                )
                gateway.write_text(
                    target_dir / "evidence.json",
                    json.dumps(payload, indent=1, sort_keys=True),
                    source="latent_cortex.fast_weights",
                )
            self.lifecycle.exported = True
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
    "FastWeightHandle",
    "FastWeightsLifecycle",
]
