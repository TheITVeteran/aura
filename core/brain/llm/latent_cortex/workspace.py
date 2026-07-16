"""The latent workspace: M writable continuous thought slots.

Slots are real sequence positions appended after the prompt. They are not
words — they are writable internal state, refined by recurrent computation
and finally persisted into the KV cache so the decoded answer attends to
them at every layer. That persistence is the causality contract: ablating a
slot (zeroing its K/V) measurably changes the answer, and Experiment 3
verifies exactly that.

Role anchors give branches/slots distinct starting basins (constructor,
counterexample-hunter, checker, ...) without any trained parameters: they are
deterministic unit-scale directions derived from the role name, so runs are
reproducible and roles are causally testable rather than decorative.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from core.brain.llm.latent_cortex.types import WorkspaceConfig

logger = logging.getLogger("Aura.LatentCortex.Workspace")


def _role_seed(role: str, base_seed: int) -> int:
    """Deterministic, platform-stable seed for a role anchor."""
    digest = hashlib.sha256(f"{role}:{base_seed}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def role_anchor(role: str, dim: int, base_seed: int = 0):
    """A deterministic direction in hidden space for a named cognitive role."""
    import mlx.core as mx

    key = mx.random.key(_role_seed(role, base_seed))
    vec = mx.random.normal((dim,), key=key)
    return vec / mx.maximum(mx.linalg.norm(vec), 1e-6)


def per_position_rms(x):
    """Per-position RMS over the hidden dimension: (..., L, D) → (..., L, 1)."""
    import mlx.core as mx

    return mx.sqrt(mx.mean(mx.square(x), axis=-1, keepdims=True))


@dataclass
class SlotAblation:
    """Record of one applied ablation, so restores are exact and auditable."""

    slot_index: int
    mode: str
    prior_state: Any = None


class LatentWorkspace:
    """M thought slots in hidden space, with snapshot/ablate/readout support.

    Holds the slot tensor ``z`` of shape (1, M, D) plus provenance. The
    workspace itself never touches the KV cache — the engine owns cache
    discipline. This separation keeps the workspace trivially testable.
    """

    def __init__(self, z, roles: list[str], config: WorkspaceConfig) -> None:
        self.z = z
        self.roles = list(roles)
        self.config = config
        self.seed_z = z  # immutable reference state for drift measurement
        self._ablations: list[SlotAblation] = []

    # ── Construction ────────────────────────────────────────────────────
    @classmethod
    def from_prompt_embeddings(
        cls,
        prompt_embeddings,
        config: WorkspaceConfig,
        *,
        branch_role: str | None = None,
    ) -> LatentWorkspace:
        """Seed M slots from the pooled prompt embedding + role anchors.

        Each slot starts at the prompt's mean embedding, perturbed along its
        role-anchor direction, then RMS-matched to the embedding distribution
        so the first prelude pass sees in-manifold inputs. ``branch_role``
        additionally rotates every anchor seed, giving branches distinct
        starting basins over identical weights.
        """
        import mlx.core as mx

        m = int(config.n_slots)
        dim = int(prompt_embeddings.shape[-1])
        pooled = mx.mean(prompt_embeddings, axis=1, keepdims=True)  # (1,1,D)
        target_rms = mx.mean(per_position_rms(prompt_embeddings))

        base_seed = config.seed
        if branch_role:
            base_seed = _role_seed(branch_role, base_seed)

        roles: list[str] = []
        anchors = []
        for i in range(m):
            role = config.roles[i % len(config.roles)]
            roles.append(role)
            anchors.append(role_anchor(f"{role}#{i}", dim, base_seed))
        anchor_mat = mx.stack(anchors, axis=0)[None, :, :]  # (1,M,D)

        z = mx.broadcast_to(pooled, (1, m, dim)) + (
            float(config.anchor_scale) * target_rms * anchor_mat
        )
        # RMS-match the seeds to the embedding norm distribution.
        z = z * (target_rms / mx.maximum(per_position_rms(z), 1e-6))
        mx.eval(z)
        return cls(z, roles, config)

    # ── State management ────────────────────────────────────────────────
    def snapshot(self):
        return self.z

    def restore(self, snap) -> None:
        self.z = snap

    def update(self, new_z) -> None:
        self.z = new_z

    # ── Causality instrumentation (Experiment 3) ────────────────────────
    def ablate(self, slot_index: int, mode: str = "zero") -> SlotAblation:
        """Destroy one slot's content in-place (zero or matched-RMS noise).

        Returns the ablation record; pass it to :meth:`restore_ablation` to
        prove recovery. Ablating the workspace BEFORE final persistence tests
        whether the slot carried causally necessary intermediate computation.
        """
        import mlx.core as mx

        if not 0 <= slot_index < self.z.shape[1]:
            raise ValueError(f"slot_index {slot_index} outside workspace of {self.z.shape[1]}")
        record = SlotAblation(slot_index=slot_index, mode=mode, prior_state=self.z)
        keep = self.z
        if mode == "zero":
            replacement = mx.zeros_like(keep[:, slot_index : slot_index + 1, :])
        elif mode == "noise":
            key = mx.random.key(_role_seed(f"ablate#{slot_index}", self.config.seed))
            noise = mx.random.normal(keep[:, slot_index : slot_index + 1, :].shape, key=key)
            rms_here = per_position_rms(keep[:, slot_index : slot_index + 1, :])
            replacement = noise * rms_here / mx.maximum(per_position_rms(noise), 1e-6)
        else:
            raise ValueError(f"unknown ablation mode: {mode!r}")
        self.z = mx.concatenate(
            [keep[:, :slot_index, :], replacement, keep[:, slot_index + 1 :, :]], axis=1
        )
        mx.eval(self.z)
        self._ablations.append(record)
        return record

    def restore_ablation(self, record: SlotAblation) -> None:
        self.z = record.prior_state
        if record in self._ablations:
            self._ablations.remove(record)

    # ── Readouts ────────────────────────────────────────────────────────
    def summary(self):
        """Mean slot state (1, 1, D) — the branch-exchange currency."""
        import mlx.core as mx

        return mx.mean(self.z, axis=1, keepdims=True)

    def stats(self) -> dict[str, Any]:
        """Cheap scalar readouts for receipts and health (no tensors)."""
        import mlx.core as mx

        rms_now = per_position_rms(self.z)
        drift_num = mx.linalg.norm(self.z - self.seed_z)
        drift_den = mx.maximum(mx.linalg.norm(self.seed_z), 1e-6)
        return {
            "n_slots": int(self.z.shape[1]),
            "dim": int(self.z.shape[2]),
            "mean_rms": float(mx.mean(rms_now)),
            "max_rms": float(mx.max(rms_now)),
            "seed_drift": float(drift_num / drift_den),
            "roles": list(self.roles),
            "active_ablations": len(self._ablations),
        }


__all__ = ["LatentWorkspace", "SlotAblation", "per_position_rms", "role_anchor"]
