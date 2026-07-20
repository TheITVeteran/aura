"""Controlled recurrence: the anti-naive-looping core.

2026 frozen-loop studies found naive layer repetition unstable. This module
is the difference between "run layers 16–47 again" and a governed dynamical
system:

  Z̃ₜ₊₁   = Window(Zₜ)                       (slots re-enter the layer window)
  Zₜ₊₁    = (1−αₜ)·Zₜ + αₜ·RMSMatch(Z̃ₜ₊₁, Zₜ)

- RMSMatch clamps per-position norm drift so the state stays on the
  activation manifold the subsequent layers were trained to expect.
- The α schedule trades update speed against stability (cosine decay ⇒
  aggressive early exploration, gentle convergence).
- The halting controller detects fixed points (converged), divergence
  (revert), budget exhaustion, and overthinking (score peaked earlier ⇒
  revert to the best state, not the last one).

KV discipline: every window pass appends slot K/V, which must be rewound so
only the engine's final clean pass persists. We reuse the battle-tested
snapshot/restore machinery from ``core.brain.llm.recurrent_depth`` — the same
code that guards the live resident model's recurrent depth today.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.brain.llm.latent_cortex.types import ComputeBudget, RecurrenceConfig
from core.brain.llm.latent_cortex.workspace import per_position_rms
from core.brain.llm.recurrent_depth import (
    _restore_recurrent_caches,
    _snapshot_recurrent_caches,
)

logger = logging.getLogger("Aura.LatentCortex.Recurrence")


def rms_match(new_state, ref_state, clip_ratio: float):
    """Bound ``new_state``'s per-position RMS to a trust band around ``ref_state``'s.

    Inside the band [ref/clip, ref·clip] the state passes through untouched —
    genuine, bounded norm movement is signal, not noise. Outside the band the
    RMS is pinned to the nearest edge, so a runaway pass can never leave the
    activation manifold the next layers were trained to expect.
    """
    import mlx.core as mx

    new_rms = mx.maximum(per_position_rms(new_state), 1e-6)
    ref_rms = per_position_rms(ref_state)
    target = mx.clip(new_rms, ref_rms / clip_ratio, ref_rms * clip_ratio)
    return new_state * (target / new_rms)


def alpha_at(config: RecurrenceConfig, step: int) -> float:
    """Interpolation coefficient for a given step under the schedule."""
    if config.alpha_schedule == "cosine":
        horizon = max(1, config.max_steps - 1)
        progress = min(1.0, step / horizon)
        # Decay from alpha to alpha/4 over the horizon.
        return config.alpha * (0.25 + 0.75 * 0.5 * (1.0 + math.cos(math.pi * progress)))
    return config.alpha


def relative_residual(z_next, z_prev) -> float:
    """‖Zₜ₊₁−Zₜ‖ / ‖Zₜ‖ in mean-RMS terms — the fixed-point signal."""
    import mlx.core as mx

    num = mx.mean(per_position_rms(z_next - z_prev))
    den = mx.maximum(mx.mean(per_position_rms(z_prev)), 1e-6)
    return float(num / den)


@dataclass
class HaltDecision:
    should_halt: bool
    reason: str = ""


@dataclass
class HaltingController:
    """Adaptive halting with divergence and overthinking protection.

    Tracks the best-scoring state seen so far (when an external score signal
    is provided) so the engine can revert to the trajectory's peak instead of
    shipping an over-thought state — the "excessive recurrence degrades
    results" failure mode from the recurrent-depth literature.
    """

    config: RecurrenceConfig
    baseline_rms: float = 0.0
    residual_trail: list[float] = field(default_factory=list)
    score_trail: list[float] = field(default_factory=list)
    best_step: int = -1
    best_score: float = -math.inf
    best_state: Any = None
    # Optional learned halting head (CP230/234). None => the residual policy
    # this controller has always run. Attaching a head grants nothing on its
    # own: the head is zero-initialised, so an untrained one never fires.
    halting_head: Any = None
    head_halts: int = 0

    def observe(
        self,
        step: int,
        z_next,
        residual: float,
        *,
        score: float | None = None,
        budget: ComputeBudget | None = None,
    ) -> HaltDecision:
        import mlx.core as mx

        self.residual_trail.append(residual)

        # Divergence guard: non-finite state or runaway norms ⇒ halt now.
        if not bool(mx.all(mx.isfinite(z_next))):
            return HaltDecision(True, "diverged_nonfinite")
        mean_rms = float(mx.mean(per_position_rms(z_next)))
        if self.baseline_rms > 0 and mean_rms > self.baseline_rms * self.config.divergence_ratio:
            return HaltDecision(True, "diverged_norm")

        # Best-state tracking (overthinking protection). Without an external
        # score, convergence quality (negative residual) is the proxy.
        effective_score = score if score is not None else -residual
        self.score_trail.append(effective_score)
        if effective_score > self.best_score:
            self.best_score = effective_score
            self.best_step = step
            self.best_state = z_next

        if budget is not None and budget.exhausted:
            return HaltDecision(True, "budget_exhausted")
        if step + 1 >= self.config.max_steps:
            return HaltDecision(True, "max_steps")
        if (
            not self.config.fixed_depth
            and step + 1 >= self.config.min_steps
            and residual < self.config.convergence_eps
        ):
            return HaltDecision(True, "converged")

        # Learned allocation, consulted only AFTER the convergence floor.
        # Residual halting answers "has this loop stopped changing?"; the
        # head answers "does this problem deserve more thought?" CP226
        # measured where those come apart -- a loop still moving healthily
        # (deltas 0.55, 0.50, 0.32) while accuracy fell to zero. Residual
        # halting sees motion and keeps going.
        if (
            self.halting_head is not None
            and not self.config.fixed_depth
            and step + 1 >= self.config.min_steps
        ):
            probability = float(self.halting_head.halt_probability(z_next))
            if probability >= self.halting_head.threshold:
                self.head_halts += 1
                return HaltDecision(True, "head_satisfied")
        return HaltDecision(False)

    def final_state(self, z_last) -> tuple[Any, bool]:
        """Return (state to ship, reverted?) — best state if it beats last."""
        if self.config.fixed_depth:
            return z_last, False
        if self.best_state is not None and (
            not self.score_trail or self.best_step < len(self.score_trail) - 1
        ):
            return self.best_state, True
        return z_last, False


class WindowRunner:
    """Runs hidden states through a contiguous layer window with KV discipline.

    ``persist=False`` (recurrent passes): slot K/V appended by the pass is
    rewound so cache offsets never drift — RoPE positions stay identical
    across passes, which the mechanics probe proved is what keeps recurrence
    stable. ``persist=True`` (the engine's final clean pass): K/V stays, so
    the decoded answer attends to the refined thoughts.
    """

    def __init__(self, inner_model, budget: ComputeBudget, mask_fn: Callable | None = None):
        self._inner = inner_model
        self._budget = budget
        self._mask_fn = mask_fn
        self._adapter_calls = 0
        self._adapter_adapted_positions = 0
        self._adapter_observed_positions = 0

    def adapter_receipt(self) -> dict[str, int | str | bool]:
        """Aggregate proof that scoped weights ran only inside slot windows."""

        return {
            "schema": "aura.recurrence_adapter_activation.v1",
            "scope": "latent_slots_only",
            "calls": self._adapter_calls,
            "adapted_positions": self._adapter_adapted_positions,
            "observed_positions": self._adapter_observed_positions,
            "active": self._adapter_calls > 0,
        }

    def _mask(self, h, cache_slice):
        if self._mask_fn is not None:
            return self._mask_fn(h, cache_slice)
        try:
            from mlx_lm.models.base import create_attention_mask
        except ImportError:  # pragma: no cover - ancient mlx_lm
            from mlx_lm.models.qwen2 import create_attention_mask  # type: ignore
        return create_attention_mask(h, cache_slice)

    def run(self, h, cache, start: int, end: int, *, persist: bool) -> Any:
        import mlx.core as mx

        from core.brain.llm.latent_cortex.recurrence_adapter import (
            recurrence_adapter_scope,
        )

        tokens = int(h.shape[1])
        layers = end - start
        if not self._budget.can_afford(tokens, layers):
            raise RuntimeError(
                f"compute budget cannot afford window [{start}:{end}) for {tokens} slots"
            )
        # Reserve and account the whole atomic pass before execution. A layer
        # fault can consume partial compute, so failed work must not disappear
        # from the conservative ledger or become available to a fallback.
        self._budget.charge(tokens=tokens, layers=layers)
        snaps = None
        if not persist:
            snaps = _snapshot_recurrent_caches(cache, start, end)
        adapter_activation = None
        try:
            mask = self._mask(h, cache[start:end])
            # A WindowRunner call is the live proof boundary that these inputs
            # are thought slots. Recurrent adapters remain dark for all direct
            # prompt, lexical decode, and unrelated model calls.
            with recurrence_adapter_scope() as adapter_activation:
                for i in range(start, end):
                    h = self._inner.layers[i](h, mask, cache[i])
            mx.eval(h)
        finally:
            if adapter_activation is not None:
                self._adapter_calls += adapter_activation.calls
                self._adapter_adapted_positions += (
                    adapter_activation.adapted_positions
                )
                self._adapter_observed_positions += (
                    adapter_activation.observed_positions
                )
            if snaps is not None:
                _restore_recurrent_caches(cache, start, end, snaps)
        return h


def recurrence_step(
    z,
    runner: WindowRunner,
    cache,
    start: int,
    end: int,
    config: RecurrenceConfig,
    step: int,
    *,
    anchor=None,
    alpha_override: float | None = None,
):
    """One controlled update: window pass (rewound) + anchored RMSMatch + α-blend.

    ``anchor`` is the manifold reference for the RMS trust band — normally the
    post-prelude seed state Z₀ ("the norm distribution expected by the
    subsequent layers"). Banding against a FIXED anchor is what prevents the
    ratchet failure: a band around the moving previous state would permit
    clip_ratio× growth per step, compounding without bound.
    """
    import mlx.core as mx

    z_raw = runner.run(z, cache, start, end, persist=False)
    alpha = alpha_override if alpha_override is not None else alpha_at(config, step)
    reference = anchor if anchor is not None else z
    z_next = (1.0 - alpha) * z + alpha * rms_match(z_raw, reference, config.rms_clip_ratio)
    mx.eval(z_next)
    return z_next


__all__ = [
    "HaltDecision",
    "HaltingController",
    "WindowRunner",
    "alpha_at",
    "recurrence_step",
    "relative_residual",
    "rms_match",
]
