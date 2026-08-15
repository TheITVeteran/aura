"""Teacher-free decode lanes for recurrent process transfer experiments."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import mlx.core as mx

from core.learning.unified_intrinsic_objective import UnifiedIntrinsicTrainingSpec
from core.learning.unified_intrinsic_recurrence import (
    UnifiedRecurrentController,
    unified_recurrent_logits,
)


def decode_base_greedy_tokens(
    model: Any,
    public_tokens: Sequence[int],
    *,
    eos_token_id: int | None,
    max_tokens: int,
    prefill_tokens: Sequence[int] = (),
    completion_check: Callable[[tuple[int, ...]], bool] | None = None,
    progress: Callable[[int], None] | None = None,
) -> tuple[tuple[int, ...], bool, int]:
    """Decode the frozen model once through its canonical cached greedy lane.

    ``prefill_tokens`` is a matched serialization surface, not an answer
    channel. It lets every experimental arm start from the same public wire
    prefix while the model still generates the complete semantic payload.
    """

    public = tuple(public_tokens)
    prefill = tuple(prefill_tokens)
    if (
        not public
        or any(type(token) is not int or token < 0 for token in public)
        or any(type(token) is not int or token < 0 for token in prefill)
        or type(max_tokens) is not int
        or max_tokens < 1
    ):
        raise ValueError("base transfer decode dimensions are invalid")
    from mlx_lm.generate import generate_step

    tokens = mx.array([*public, *prefill], dtype=mx.int32)
    generated: list[int] = list(prefill)
    stopped = False
    started = time.perf_counter()
    for token_id, _logprobs in generate_step(
        tokens,
        model,
        max_tokens=max_tokens,
        sampler=lambda values: mx.argmax(values, axis=-1),
    ):
        token_id = int(token_id)
        generated.append(token_id)
        if progress is not None:
            progress(len(generated) - len(prefill))
        if eos_token_id is not None and token_id == eos_token_id:
            stopped = True
            break
        if completion_check is not None and completion_check(tuple(generated)):
            stopped = True
            break
    elapsed_ms = max(0, int(round((time.perf_counter() - started) * 1000.0)))
    return tuple(generated), stopped, elapsed_ms


def decode_typed_process_tokens(
    model: Any,
    controller: UnifiedRecurrentController,
    spec: UnifiedIntrinsicTrainingSpec,
    public_tokens: Sequence[int],
    *,
    recurrence_depth: int,
    eos_token_id: int | None,
    max_tokens: int,
    typed_action_lesion: bool = False,
    transition_processor_lesion: bool = False,
    transition_history_lesion: bool = False,
    process_tape_lesion: bool = False,
    completion_check: Callable[[tuple[int, ...]], bool] | None = None,
    progress: Callable[[int], None] | None = None,
) -> tuple[tuple[int, ...], bool, int]:
    """Decode through typed recurrent process tissue and the ordinary LM head.

    The public prompt is the only evidence surface. State/action slots execute
    at every token, while terminal grammar and digit pointers remain absent.
    Full-prefix replay is intentional: incremental recurrent caches exclude
    typed slots and would silently select a different mechanism.
    """

    if (
        not isinstance(controller, UnifiedRecurrentController)
        or not isinstance(spec, UnifiedIntrinsicTrainingSpec)
        or not public_tokens
        or type(recurrence_depth) is not int
        or recurrence_depth < 1
        or type(max_tokens) is not int
        or max_tokens < 1
        or type(typed_action_lesion) is not bool
        or type(transition_processor_lesion) is not bool
        or type(transition_history_lesion) is not bool
        or type(process_tape_lesion) is not bool
    ):
        raise ValueError("typed transfer decode dimensions are invalid")
    prompt_count = len(public_tokens)
    tokens = mx.array([list(public_tokens)], dtype=mx.int32)
    generated: list[int] = []
    stopped = False
    started = time.perf_counter()
    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_scope,
    )

    with recurrence_adapter_scope(start=None, stop=None):
        for _index in range(max_tokens):
            logits, _telemetry = unified_recurrent_logits(
                model,
                tokens,
                spec.plan_at(recurrence_depth),
                controller,
                state_slot_start=prompt_count,
                answer_digit_pointer_enabled=False,
                typed_action_lesion=typed_action_lesion,
                transition_processor_lesion=transition_processor_lesion,
                transition_history_lesion=transition_history_lesion,
                process_tape_lesion=process_tape_lesion,
            )
            token_id = int(mx.argmax(logits[0, -1]).item())
            generated.append(token_id)
            if progress is not None:
                progress(len(generated))
            if eos_token_id is not None and token_id == eos_token_id:
                stopped = True
                break
            if completion_check is not None and completion_check(tuple(generated)):
                stopped = True
                break
            tokens = mx.concatenate(
                [tokens, mx.array([[token_id]], dtype=tokens.dtype)],
                axis=1,
            )
    elapsed_ms = max(0, int(round((time.perf_counter() - started) * 1000.0)))
    return tuple(generated), stopped, elapsed_ms


__all__ = ["decode_base_greedy_tokens", "decode_typed_process_tokens"]
