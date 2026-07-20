"""Force reasoning THROUGH the latent workspace (CP224).

Measured: destroying the slots destroys the answer (6/6), so the workspace
is causally read. Also measured: recurrent depth 1 -> 8 leaves 32B accuracy
flat (25 / 29 / 25 / 25). Both can be true at once, and together they say
something precise:

    The model READS the workspace, but does not ROUTE REASONING through it.

The answer attends to ~4 slot positions against a couple hundred prompt
positions. The prompt contains the whole problem, so the cheapest policy is
to solve it from the prompt in native chain-of-thought and treat the slots
as a weak prior. Nothing in training ever made that policy costly. Extra
recurrence then cannot help: it refines a channel the model is not relying
on.

This module supplies the pressure that makes reliance necessary. During
TRAINING only, prompt positions are stochastically masked from the answer's
view, so some fraction of examples can only be answered from what the
recurrence deposited in the slots. A model that routes computation into the
workspace keeps scoring; one that ignores it cannot.

Deliberately training-only. At inference the prompt is always fully
visible: the goal is a model that CHOSE to use the workspace, not one
crippled into it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

WORKSPACE_BOTTLENECK_SCHEMA = "aura.workspace_bottleneck.v1"


@dataclass(frozen=True)
class BottleneckSchedule:
    """How hard to squeeze, and how that changes over training.

    Starts at zero so early training matches the unmodified objective
    exactly, then ramps: a model asked to rely on an untrained workspace
    from step one learns nothing except that the task is impossible.
    """

    start_fraction: float = 0.0
    end_fraction: float = 0.5
    warmup_steps: int = 200
    # Never mask the final tokens of the prompt: the question itself must
    # remain visible or the task becomes unanswerable rather than harder.
    protected_tail: int = 24

    def __post_init__(self) -> None:
        for name, value in (
            ("start_fraction", self.start_fraction),
            ("end_fraction", self.end_fraction),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) < 1.0
            ):
                raise ValueError(f"{name} must be inside [0, 1)")
        if self.end_fraction < self.start_fraction:
            raise ValueError("end_fraction cannot be below start_fraction")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer")
        if type(self.protected_tail) is not int or self.protected_tail < 0:
            raise ValueError("protected_tail must be a non-negative integer")

    def fraction_at(self, step: int) -> float:
        if type(step) is not int or step < 0:
            raise ValueError("step must be a non-negative integer")
        if self.warmup_steps == 0:
            return self.end_fraction
        progress = min(1.0, step / self.warmup_steps)
        return self.start_fraction + progress * (
            self.end_fraction - self.start_fraction
        )


def prompt_visibility_mask(
    prompt_length: int,
    *,
    fraction: float,
    seed: int,
    protected_tail: int = 24,
) -> Any:
    """Per-position keep-mask over the prompt, shape (1, prompt_length, 1).

    Deterministic in ``seed`` so a training step is reproducible and an
    identical example masks identically on resume. The tail is always kept:
    hiding the question turns a harder task into an impossible one, and a
    model cannot learn to route through the workspace by guessing.
    """
    import mlx.core as mx

    if type(prompt_length) is not int or prompt_length < 1:
        raise ValueError("prompt_length must be a positive integer")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not 0.0 <= float(fraction) < 1.0
    ):
        raise ValueError("fraction must be inside [0, 1)")
    keep = mx.ones((prompt_length,))
    if fraction <= 0.0 or prompt_length <= protected_tail:
        return mx.reshape(keep, (1, prompt_length, 1))
    maskable = prompt_length - protected_tail
    draws = mx.random.uniform(shape=(maskable,), key=mx.random.key(seed))
    body = (draws >= fraction).astype(keep.dtype)
    keep = mx.concatenate([body, mx.ones((protected_tail,))])
    return mx.reshape(keep, (1, prompt_length, 1))


def apply_bottleneck(
    prompt_hidden: Any,
    *,
    fraction: float,
    seed: int,
    protected_tail: int = 24,
) -> tuple[Any, dict[str, Any]]:
    """Hide a fraction of prompt positions from the answer's view.

    Masked positions are zeroed rather than removed so every downstream
    shape, position index and RoPE offset is unchanged -- the model sees the
    same geometry with less information, which is the intended pressure.
    Returns ``(masked_hidden, receipt)``.
    """
    import mlx.core as mx

    prompt_length = int(prompt_hidden.shape[1])
    keep = prompt_visibility_mask(
        prompt_length,
        fraction=fraction,
        seed=seed,
        protected_tail=protected_tail,
    )
    hidden = prompt_hidden * keep.astype(prompt_hidden.dtype)
    kept = float(mx.sum(keep))
    return hidden, {
        "schema": WORKSPACE_BOTTLENECK_SCHEMA,
        "prompt_positions": prompt_length,
        "kept_positions": int(kept),
        "hidden_positions": prompt_length - int(kept),
        "requested_fraction": round(float(fraction), 4),
        "effective_fraction": round(
            1.0 - kept / max(prompt_length, 1), 4
        ),
        "protected_tail": protected_tail,
    }


def reliance_score(
    intact_loss: float,
    ablated_loss: float,
) -> dict[str, Any]:
    """How much the answer depends on the workspace.

    ``ablated_loss`` is measured with the slots destroyed. A model that
    routes reasoning through the workspace degrades sharply when it is
    removed; one that merely tolerates it barely moves. This is the number
    the bottleneck is trying to raise, and it is meaningful in a way that
    'the slots are causal' is not -- causality is binary, reliance is a
    magnitude.
    """
    if intact_loss <= 0.0:
        raise ValueError("intact_loss must be positive")
    delta = ablated_loss - intact_loss
    return {
        "schema": WORKSPACE_BOTTLENECK_SCHEMA,
        "intact_loss": round(float(intact_loss), 6),
        "ablated_loss": round(float(ablated_loss), 6),
        "reliance": round(float(delta / intact_loss), 6),
        "workspace_load_bearing": bool(delta / intact_loss > 0.10),
    }


__all__ = [
    "WORKSPACE_BOTTLENECK_SCHEMA",
    "BottleneckSchedule",
    "apply_bottleneck",
    "prompt_visibility_mask",
    "reliance_score",
]
