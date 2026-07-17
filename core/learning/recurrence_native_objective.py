"""Recurrence-native training objective — teach the weights to think in loops.

The RLC makes recurrence useful on a checkpoint that was never trained for
it; frozen-loop literature says that ceiling is real. The single largest
credible improvement in the whole architecture list is training the model
WHILE it operates recurrently, so the window layers learn to be a
reusable reasoning operator instead of a one-shot pipeline stage.

This module is the OBJECTIVE, built to be consumed by the existing
governed training lanes (LoRA adapter proposals → durable-learning train →
regression gates → activation → proven rollback). It deliberately owns no
optimizer loop of its own beyond what tests need: training runs stay
operator-launched and gated, like every other weight-touching path.

    loss = CE(answer tokens | recurrent forward)

where the recurrent forward is the SAME shape the engine executes:
prelude layers once → window layers T times under anchored RMS-matched
α-interpolation → coda layers once → logits. Gradients flow through every
recurrent application, so descent shapes the window layers' fixed-point
behavior, not just their single-pass output.

The anchored interpolation matters: matching each step's norms against the
PRELUDE state (not the previous step) is the same fix that turned live
recurrence from a norm ratchet into a genuine contraction — training
against the ratcheting variant would teach the model to expect
off-manifold inputs.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Aura.RecurrenceNativeObjective")

RECURRENCE_NATIVE_SCHEMA = "aura.recurrence_native_objective.v1"

# Train/inference parity: the objective must apply EXACTLY the norm control
# the engine executes at inference — the anchored trust band from
# recurrence.rms_match (identity inside the band, pinned at the edge
# outside), with the same default clip ratio as RecurrenceConfig. Training
# under a different rescale (e.g. a hard pull to anchor norms) would teach
# the window layers dynamics the live episode never produces.
_CLIP_RATIO = 3.0


def _rms_match_anchored(new_state, anchor, clip_ratio: float = _CLIP_RATIO):
    """The engine's anchored RMS trust band, reused verbatim for training."""
    from core.brain.llm.latent_cortex.recurrence import rms_match

    return rms_match(new_state, anchor, clip_ratio)


def recurrent_forward_logits(
    model,
    input_tokens: list[int],
    *,
    recurrent_steps: int = 2,
    alpha: float = 0.5,
    prelude_frac: float = 0.25,
    coda_frac: float = 0.25,
):
    """Full-sequence recurrent forward: prelude → T× window → coda → logits.

    Differentiable end to end (no KV caches, no rewinds — this is the
    TRAINING view of the same computation the engine performs over slots).
    """
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    if type(recurrent_steps) is not int or recurrent_steps < 1:
        raise ValueError("recurrent_steps must be a positive integer")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not 0.0 < float(alpha) <= 1.0
    ):
        raise ValueError("alpha must be inside (0, 1]")
    inner = model.model
    n_layers = len(inner.layers)
    prelude_end = max(1, int(n_layers * float(prelude_frac)))
    coda_start = min(n_layers - 1, n_layers - int(n_layers * float(coda_frac)))
    if coda_start - prelude_end < 1:
        raise ValueError(
            f"recurrent region empty for {n_layers} layers "
            f"(prelude_end={prelude_end}, coda_start={coda_start})"
        )
    hidden = inner.embed_tokens(mx.array([list(input_tokens)]))
    mask = create_attention_mask(hidden, None)
    for layer in inner.layers[:prelude_end]:
        hidden = layer(hidden, mask, None)
    anchor = hidden
    state = hidden
    for _ in range(recurrent_steps):
        candidate = state
        for layer in inner.layers[prelude_end:coda_start]:
            candidate = layer(candidate, mask, None)
        state = (1.0 - float(alpha)) * state + float(alpha) * _rms_match_anchored(
            candidate, anchor
        )
    hidden = state
    for layer in inner.layers[coda_start:]:
        hidden = layer(hidden, mask, None)
    hidden = inner.norm(hidden)
    head = getattr(model, "lm_head", None)
    if head is not None and not isinstance(head, type(inner.embed_tokens)):
        return head(hidden)
    return inner.embed_tokens.as_linear(hidden)


def recurrence_native_loss(
    model,
    input_tokens: list[int],
    answer_start: int,
    *,
    recurrent_steps: int = 2,
    alpha: float = 0.5,
    prelude_frac: float = 0.25,
    coda_frac: float = 0.25,
):
    """Answer-span cross-entropy under the recurrent forward.

    ``answer_start`` marks where the answer begins inside ``input_tokens``;
    only answer-token predictions carry loss (the prompt is context, not a
    target — we are training reasoning-under-recurrence, not prompt
    memorization).
    """
    import mlx.core as mx
    import mlx.nn as nn

    tokens = list(input_tokens)
    if len(tokens) < 2:
        raise ValueError("need at least two tokens for a next-token target")
    if type(answer_start) is not int or not 1 <= answer_start < len(tokens):
        raise ValueError("answer_start must index inside the token sequence")
    logits = recurrent_forward_logits(
        model,
        tokens,
        recurrent_steps=recurrent_steps,
        alpha=alpha,
        prelude_frac=prelude_frac,
        coda_frac=coda_frac,
    )
    # Predict token[t+1] from position t; loss only where t+1 is answer.
    shifted_logits = logits[0, :-1, :]
    targets = mx.array(tokens[1:])
    per_token = nn.losses.cross_entropy(shifted_logits, targets, reduction="none")
    answer_mask = mx.array(
        [1.0 if index + 1 >= answer_start else 0.0 for index in range(len(tokens) - 1)]
    )
    denominator = mx.maximum(answer_mask.sum(), 1.0)
    return (per_token * answer_mask).sum() / denominator


def depth_curriculum_loss(
    model,
    input_tokens: list[int],
    answer_start: int,
    *,
    depths: tuple[int, ...] = (1, 2, 4),
    monotonicity_weight: float = 0.5,
    alpha: float = 0.5,
    prelude_frac: float = 0.25,
    coda_frac: float = 0.25,
):
    """The recurrent-depth curriculum: reward accuracy that GROWS with depth.

    Trainable form of the spec's S(x, T+1) ≥ S(x, T): the mean answer-span
    loss across the depth ladder anchors capability at every depth, and a
    hinge penalty relu(loss(T_deeper) − loss(T_shallower)) fires whenever
    extra recurrence makes the model WORSE — the late-step degradation the
    live overthinking guard has to revert today. Descent on this objective
    teaches the window layers that more thought must never cost accuracy,
    which is what makes a learned compute allocator meaningful.
    """
    import mlx.core as mx

    ladder = tuple(depths)
    if (
        len(ladder) < 2
        or any(type(depth) is not int or depth < 1 for depth in ladder)
        or sorted(set(ladder)) != list(ladder)
    ):
        raise ValueError("depths must be a strictly increasing tuple of ints")
    if (
        isinstance(monotonicity_weight, bool)
        or not isinstance(monotonicity_weight, (int, float))
        or not 0.0 <= float(monotonicity_weight) <= 10.0
    ):
        raise ValueError("monotonicity_weight must be inside [0, 10]")
    losses = [
        recurrence_native_loss(
            model,
            input_tokens,
            answer_start,
            recurrent_steps=depth,
            alpha=alpha,
            prelude_frac=prelude_frac,
            coda_frac=coda_frac,
        )
        for depth in ladder
    ]
    total = losses[0]
    for loss in losses[1:]:
        total = total + loss
    mean_loss = total / len(losses)
    penalty = mx.zeros(())
    for shallow, deep in zip(losses, losses[1:]):
        penalty = penalty + mx.maximum(deep - shallow, 0.0)
    return mean_loss + float(monotonicity_weight) * penalty


def objective_receipt(
    *,
    recurrent_steps: int,
    alpha: float,
    batch_count: int,
    mean_loss: float,
) -> dict[str, Any]:
    """Receipt shape the durable-learning train records per proposal."""
    return {
        "schema": RECURRENCE_NATIVE_SCHEMA,
        "recurrent_steps": int(recurrent_steps),
        "alpha": round(float(alpha), 4),
        "batch_count": int(batch_count),
        "mean_loss": round(float(mean_loss), 6),
        "objective": "answer_span_ce_under_recurrent_forward",
    }


__all__ = [
    "RECURRENCE_NATIVE_SCHEMA",
    "depth_curriculum_loss",
    "objective_receipt",
    "recurrence_native_loss",
    "recurrent_forward_logits",
]
