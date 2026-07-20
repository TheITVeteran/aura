"""Live-path recurrence-native objective over latent slots.

Unlike v1, this objective never recurs lexical prompt or answer states. It
reproduces the live causal layout in a differentiable no-cache view:

1. embed and prefill the prompt;
2. create deterministic role-seeded latent slots;
3. recur only those slots against the fixed prompt prefix;
4. exchange branch consensus and apply the live anti-collapse perturbation;
5. persist each final branch through prelude/window/coda at slot positions;
6. score teacher-forced answer tokens after the persisted slots.

The no-cache view is mathematically equivalent to prompt KV reuse because the
attention mask is causal and the scoped adapter is zero on prompt/answer
positions. Tiny-Qwen parity tests compare it directly with the live cache path.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.brain.llm.latent_cortex.recurrence import rms_match
from core.brain.llm.latent_cortex.recurrence_adapter import (
    current_recurrence_adapter_scope,
    recurrence_adapter_scope,
)
from core.brain.llm.latent_cortex.types import WorkspaceConfig
from core.brain.llm.latent_cortex.workspace import LatentWorkspace, per_position_rms

RECURRENCE_NATIVE_SCHEMA_V2 = "aura.recurrence_native_objective.v2"


@dataclass
class _LayerCheckpointState:
    model: Any
    parameters: Any
    group_size: int
    wrappers: dict[
        tuple[tuple[int, ...], int | None, int | None],
        Callable[..., Any],
    ]
    transition_wrappers: dict[int, Callable[..., Any]]


_LAYER_CHECKPOINTS: ContextVar[_LayerCheckpointState | None] = ContextVar(
    "aura_recurrence_layer_checkpoints",
    default=None,
)

# Per-step phase scale. A ContextVar rather than an RLCExecutionSpec field
# on purpose: the spec is hash-bound into adapter identity receipts, so
# adding a key would invalidate every existing bundle. Default 0.0 keeps
# behavior bit-identical unless a caller explicitly opts in.
_PHASE_SCALE: ContextVar[float] = ContextVar(
    "aura_recurrence_phase_scale",
    default=0.0,
)


@contextmanager
def recurrent_phase(scale: float) -> Iterator[None]:
    """Give each recurrent step an identity for the duration of a forward."""
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not 0.0 <= float(scale) <= 1.0
    ):
        raise ValueError("phase scale must be inside [0, 1]")
    token = _PHASE_SCALE.set(float(scale))
    try:
        yield
    finally:
        _PHASE_SCALE.reset(token)


@dataclass(frozen=True)
class LivePathForward:
    """Differentiable outputs and structural evidence from one depth."""

    branch_logits: tuple[Any, ...]
    branch_states: tuple[Any, ...]
    exchanges: int
    prompt_tokens: int
    answer_tokens: int
    bridge_tokens: int


@dataclass(frozen=True)
class _PreparedLivePath:
    prompt_embeddings: Any
    tail_embeddings: Any
    seeds: tuple[Any, ...]
    prompts_at_window: tuple[Any, ...]
    states: tuple[Any, ...]
    anchors: tuple[Any, ...]
    prelude_end: int
    coda_start: int
    prompt_count: int
    bridge_count: int
    answer_count: int


def _boundaries(model: Any, spec: RLCExecutionSpec) -> tuple[int, int, int]:
    n_layers = len(model.model.layers)
    prelude_end = max(1, int(n_layers * spec.prelude_frac))
    coda_start = min(n_layers - 1, n_layers - int(n_layers * spec.coda_frac))
    if coda_start - prelude_end < 1:
        raise ValueError("execution spec leaves no recurrent layer window")
    return n_layers, prelude_end, coda_start


def _logits(model: Any, hidden: Any) -> Any:
    inner = model.model
    hidden = inner.norm(hidden)
    head = getattr(model, "lm_head", None)
    if head is not None and not isinstance(head, type(inner.embed_tokens)):
        return head(hidden)
    return inner.embed_tokens.as_linear(hidden)


@contextmanager
def transformer_layer_group_checkpointing(
    model: Any,
    parameters: Any,
    *,
    group_size: int = 4,
) -> Iterator[None]:
    """Rematerialize bounded layer groups while preserving graph semantics."""

    layers = tuple(model.model.layers)
    if not layers:
        raise ValueError("model has no transformer layers")
    if type(group_size) is not int or not 1 <= group_size <= len(layers):
        raise ValueError("group_size must be inside [1, model layer count]")
    token = _LAYER_CHECKPOINTS.set(
        _LayerCheckpointState(
            model=model,
            parameters=parameters,
            group_size=group_size,
            wrappers={},
            transition_wrappers={},
        )
    )
    try:
        yield
    finally:
        _LAYER_CHECKPOINTS.reset(token)


def _causal_layers(layers: Sequence[Any], hidden: Any) -> Any:
    from mlx_lm.models.base import create_attention_mask

    layer_sequence = tuple(layers)
    if not layer_sequence:
        return hidden
    checkpointed = _LAYER_CHECKPOINTS.get()
    if checkpointed is None:
        for layer in layer_sequence:
            hidden = layer(hidden, create_attention_mask(hidden, None), None)
        return hidden
    activation = current_recurrence_adapter_scope()
    start = activation.start if activation is not None else None
    stop = activation.stop if activation is not None else None
    import mlx.core as mx

    for offset in range(0, len(layer_sequence), checkpointed.group_size):
        group = layer_sequence[offset : offset + checkpointed.group_size]
        key = (tuple(id(layer) for layer in group), start, stop)
        call = checkpointed.wrappers.get(key)
        if call is None:

            def layer_group_call(
                all_parameters: Any,
                value: Any,
                _group: tuple[Any, ...] = group,
                _start: int | None = start,
                _stop: int | None = stop,
            ) -> Any:
                checkpointed.model.update(all_parameters)

                def run(current: Any) -> Any:
                    for member in _group:
                        current = member(
                            current,
                            create_attention_mask(current, None),
                            None,
                        )
                    return current

                if _start is None or _stop is None:
                    return run(value)
                with recurrence_adapter_scope(start=_start, stop=_stop):
                    return run(value)

            call = mx.checkpoint(layer_group_call)
            checkpointed.wrappers[key] = call
        hidden = call(checkpointed.parameters, hidden)
    return hidden


def _seed_branch(
    prompt_embeddings: Any,
    spec: RLCExecutionSpec,
    branch_role: str,
) -> Any:
    workspace = LatentWorkspace.from_prompt_embeddings(
        prompt_embeddings,
        WorkspaceConfig(
            n_slots=spec.n_slots,
            seed=spec.slot_seed,
            roles=spec.slot_roles,
            anchor_scale=spec.anchor_scale,
        ),
        branch_role=branch_role,
    )
    return workspace.seed_z


def _prelude_prompt_and_slots(
    model: Any,
    prompt_embeddings: Any,
    slot_seed: Any,
    prelude_end: int,
) -> tuple[Any, Any]:
    import mlx.core as mx

    prompt_length = int(prompt_embeddings.shape[1])
    hidden = mx.concatenate([prompt_embeddings, slot_seed], axis=1)
    hidden = _causal_layers(model.model.layers[:prelude_end], hidden)
    return hidden[:, :prompt_length, :], hidden[:, prompt_length:, :]


def recurrent_phase_code(step: int, hidden: int) -> Any:
    """Parameter-free sinusoidal code identifying a recurrent step (CP210).

    The recurrence applies the SAME operator every step, so no step can
    know which step it is and no staged algorithm (encode -> retrieve ->
    compare -> verify) is expressible. Measured consequence: the operator
    is a contraction (residual 0.302 -> 0.026, asymptoting) that reaches a
    fixed point by step ~10 and stops computing — which is why depth
    saturates at 8, why deeper mildly hurts, and why branches (all falling
    into the same fixed point) collapse.

    Injecting this code gives each step an identity, exactly as positional
    encoding differentiates otherwise-identical tokens. Measured on the
    untrained 1.5B over khop: best-depth CE 1.8072 -> 1.6958 (-6.2%), with
    the gain GROWING at depth (d4 -2.4%, d8 -6.2%, d16 -7.6%).

    It is an input-side signal, so it does not by itself break the
    contraction (residual ratio moved only 0.1142 -> 0.1048); a trained
    phase-conditioned OPERATOR is required for that. This is the free part.
    """
    import mlx.core as mx

    positions = mx.arange(hidden, dtype=mx.float32)
    frequency = mx.exp(
        -math.log(10000.0) * (2 * mx.floor(positions / 2)) / hidden
    )
    angle = float(step) * frequency
    return mx.where(positions % 2 == 0, mx.sin(angle), mx.cos(angle))


def _window_pass(
    model: Any,
    prompt_at_window: Any,
    slots: Any,
    prelude_end: int,
    coda_start: int,
    *,
    phase_step: int | None = None,
) -> Any:
    import mlx.core as mx

    phase_scale = _PHASE_SCALE.get()
    if phase_step is not None and phase_scale > 0.0:
        rms = mx.sqrt(mx.mean(mx.square(slots)) + 1e-9)
        code = recurrent_phase_code(phase_step, int(slots.shape[-1]))
        slots = slots + phase_scale * rms * code[None, None, :]
    prompt_length = int(prompt_at_window.shape[1])
    slot_count = int(slots.shape[1])
    prompt_hidden = prompt_at_window
    slot_hidden = slots
    with recurrence_adapter_scope(
        start=prompt_length,
        stop=prompt_length + slot_count,
    ):
        joined = mx.concatenate([prompt_hidden, slot_hidden], axis=1)
        joined = _causal_layers(model.model.layers[prelude_end:coda_start], joined)
        prompt_hidden = joined[:, :prompt_length, :]
        slot_hidden = joined[:, prompt_length:, :]
    return slot_hidden


def _alpha_at(spec: RLCExecutionSpec, step: int) -> float:
    if spec.alpha_schedule != "cosine":
        return spec.alpha
    import math

    horizon = max(1, spec.recurrent_steps - 1)
    progress = min(1.0, step / horizon)
    return spec.alpha * (
        0.25 + 0.75 * 0.5 * (1.0 + math.cos(math.pi * progress))
    )


def _exchange_and_decorrelate(
    states: list[Any],
    spec: RLCExecutionSpec,
    step_number: int,
) -> list[Any]:
    import mlx.core as mx

    if len(states) < 2:
        return states
    summaries = [mx.mean(state, axis=1, keepdims=True) for state in states]
    stack = mx.concatenate(summaries, axis=1)
    mean = mx.mean(stack, axis=1, keepdims=True)

    def cosine(left: Any, right: Any) -> Any:
        denominator = mx.maximum(
            mx.linalg.norm(left) * mx.linalg.norm(right), 1e-6
        )
        return mx.sum(left * right) / denominator

    agreements = mx.stack([cosine(summary, mean) for summary in summaries])
    weights = mx.softmax(agreements, axis=0)
    consensus = sum(
        weight * summary
        for weight, summary in zip(weights, summaries, strict=True)
    )
    slot = spec.comm_slot
    exchanged: list[Any] = []
    for state in states:
        comm = (
            (1.0 - spec.exchange_gamma) * state[:, slot : slot + 1, :]
            + spec.exchange_gamma * consensus
        )
        exchanged.append(
            mx.concatenate(
                [state[:, :slot, :], comm, state[:, slot + 1 :, :]],
                axis=1,
            )
        )

    for left_index in range(len(exchanged)):
        for right_index in range(left_index + 1, len(exchanged)):
            left = exchanged[left_index]
            right = exchanged[right_index]
            similarity = cosine(
                mx.mean(left, axis=1, keepdims=True),
                mx.mean(right, axis=1, keepdims=True),
            )
            gate = (similarity > spec.collapse_cos_threshold).astype(right.dtype)
            key = mx.random.key(
                1000 + 31 * left_index + right_index + step_number
            )
            jitter = mx.random.normal(right.shape, key=key)
            jitter = jitter * (
                spec.jitter_scale
                * per_position_rms(right)
                / mx.maximum(per_position_rms(jitter), 1e-6)
            )
            exchanged[right_index] = right + gate * jitter
    return exchanged


def _advance_recurrent_states(
    model: Any,
    prompts_at_window: Sequence[Any],
    states: Sequence[Any],
    anchors: Sequence[Any],
    spec: RLCExecutionSpec,
    step: int,
    prelude_end: int,
    coda_start: int,
) -> list[Any]:
    updated: list[Any] = []
    alpha = _alpha_at(spec, step)
    for prompt_at_window, state, anchor in zip(
        prompts_at_window,
        states,
        anchors,
        strict=True,
    ):
        # Publish the recurrent step so any attached depth-conditioned
        # operator bank selects this step's effective transform. A no-op
        # when no bank is attached.
        from core.learning.depth_conditioned_lora import recurrent_depth_index

        with recurrent_depth_index(step):
            candidate = _window_pass(
                model,
                prompt_at_window,
                state,
                prelude_end,
                coda_start,
                phase_step=step,
            )
        updated.append(
            (1.0 - alpha) * state
            + alpha * rms_match(candidate, anchor, spec.rms_clip_ratio)
        )
    if len(updated) > 1 and (step + 1) % spec.exchange_interval == 0:
        return _exchange_and_decorrelate(updated, spec, step + 1)
    return updated


def _checkpointed_recurrent_transition(
    model: Any,
    prompts_at_window: Sequence[Any],
    states: Sequence[Any],
    anchors: Sequence[Any],
    spec: RLCExecutionSpec,
    step: int,
    prelude_end: int,
    coda_start: int,
) -> list[Any]:
    checkpointed = _LAYER_CHECKPOINTS.get()
    if checkpointed is None:
        return _advance_recurrent_states(
            model,
            prompts_at_window,
            states,
            anchors,
            spec,
            step,
            prelude_end,
            coda_start,
        )
    branch_count = len(states)
    call = checkpointed.transition_wrappers.get(step)
    if call is None:
        import mlx.core as mx

        def transition(all_parameters: Any, *values: Any) -> tuple[Any, ...]:
            checkpointed.model.update(all_parameters)
            prompts = values[:branch_count]
            current_states = values[branch_count : 2 * branch_count]
            current_anchors = values[2 * branch_count :]
            token = _LAYER_CHECKPOINTS.set(None)
            try:
                return tuple(
                    _advance_recurrent_states(
                        checkpointed.model,
                        prompts,
                        current_states,
                        current_anchors,
                        spec,
                        step,
                        prelude_end,
                        coda_start,
                    )
                )
            finally:
                _LAYER_CHECKPOINTS.reset(token)

        call = mx.checkpoint(transition)
        checkpointed.transition_wrappers[step] = call
    return list(
        call(
            checkpointed.parameters,
            *prompts_at_window,
            *states,
            *anchors,
        )
    )


def _persist_and_score(
    model: Any,
    prompt_embeddings: Any,
    slot_seed: Any,
    final_slots: Any,
    tail_embeddings: Any,
    *,
    bridge_count: int,
    answer_count: int,
    prelude_end: int,
    coda_start: int,
) -> Any:
    import mlx.core as mx

    prompt_length = int(prompt_embeddings.shape[1])
    slot_count = int(slot_seed.shape[1])
    hidden = mx.concatenate([prompt_embeddings, slot_seed, tail_embeddings], axis=1)
    hidden = _causal_layers(model.model.layers[:prelude_end], hidden)
    slot_start = prompt_length
    slot_stop = prompt_length + slot_count
    hidden = mx.concatenate(
        [hidden[:, :slot_start, :], final_slots, hidden[:, slot_stop:, :]],
        axis=1,
    )
    with recurrence_adapter_scope(start=slot_start, stop=slot_stop):
        hidden = _causal_layers(model.model.layers[prelude_end:coda_start], hidden)
    hidden = _causal_layers(model.model.layers[coda_start:], hidden)
    all_logits = _logits(model, hidden)
    answer_start = prompt_length + slot_count + bridge_count
    prediction_start = answer_start - 1
    return all_logits[:, prediction_start : prediction_start + answer_count, :]


def _prepare_live_path(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    bridge_tokens: Sequence[int],
) -> _PreparedLivePath:
    import mlx.core as mx

    problems = spec.validate()
    if problems:
        raise ValueError(f"invalid execution spec: {problems}")
    prompt = list(prompt_tokens)
    answer = list(answer_tokens)
    bridge = list(bridge_tokens)
    if not prompt or any(type(token) is not int or token < 0 for token in prompt):
        raise ValueError("prompt_tokens must contain non-negative integers")
    if not answer or any(type(token) is not int or token < 0 for token in answer):
        raise ValueError("answer_tokens must contain non-negative integers")
    if any(type(token) is not int or token < 0 for token in bridge):
        raise ValueError("bridge_tokens must contain non-negative integers")
    if spec.decode_bridge_policy == "none" and bridge:
        raise ValueError("bridge tokens supplied while decode bridge is disabled")
    if spec.decode_bridge_policy != "none" and not bridge:
        raise ValueError("execution spec requires decode bridge tokens")

    _n_layers, prelude_end, coda_start = _boundaries(model, spec)
    inner = model.model
    prompt_embeddings = inner.embed_tokens(mx.array([prompt]))
    tail_embeddings = inner.embed_tokens(mx.array([bridge + answer]))
    seeds: list[Any] = []
    prompts_at_window: list[Any] = []
    states: list[Any] = []
    anchors: list[Any] = []
    for role in spec.branch_roles:
        seed = _seed_branch(prompt_embeddings, spec, role)
        prompt_at_window, state = _prelude_prompt_and_slots(
            model,
            prompt_embeddings,
            seed,
            prelude_end,
        )
        seeds.append(seed)
        prompts_at_window.append(prompt_at_window)
        states.append(state)
        anchors.append(state)
    return _PreparedLivePath(
        prompt_embeddings=prompt_embeddings,
        tail_embeddings=tail_embeddings,
        seeds=tuple(seeds),
        prompts_at_window=tuple(prompts_at_window),
        states=tuple(states),
        anchors=tuple(anchors),
        prelude_end=prelude_end,
        coda_start=coda_start,
        prompt_count=len(prompt),
        bridge_count=len(bridge),
        answer_count=len(answer),
    )


def live_path_forward(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    bridge_tokens: Sequence[int] = (),
) -> LivePathForward:
    """Run the differentiable latent-slot path and return per-branch logits."""

    prepared = _prepare_live_path(
        model,
        prompt_tokens,
        answer_tokens,
        spec=spec,
        bridge_tokens=bridge_tokens,
    )
    states = list(prepared.states)

    exchanges = 0
    for step in range(spec.recurrent_steps):
        states = _checkpointed_recurrent_transition(
            model,
            prepared.prompts_at_window,
            states,
            prepared.anchors,
            spec,
            step,
            prepared.prelude_end,
            prepared.coda_start,
        )
        if len(states) > 1 and (step + 1) % spec.exchange_interval == 0:
            exchanges += 1

    branch_logits = tuple(
        _persist_and_score(
            model,
            prepared.prompt_embeddings,
            seed,
            state,
            prepared.tail_embeddings,
            bridge_count=prepared.bridge_count,
            answer_count=prepared.answer_count,
            prelude_end=prepared.prelude_end,
            coda_start=prepared.coda_start,
        )
        for seed, state in zip(prepared.seeds, states, strict=True)
    )
    return LivePathForward(
        branch_logits=branch_logits,
        branch_states=tuple(states),
        exchanges=exchanges,
        prompt_tokens=prepared.prompt_count,
        answer_tokens=prepared.answer_count,
        bridge_tokens=prepared.bridge_count,
    )


def branch_mean_answer_loss(forward: LivePathForward, answer_tokens: Sequence[int]) -> Any:
    """Mean answer CE: every role must remain competent, not only an oracle arm."""

    import mlx.core as mx
    import mlx.nn as nn

    targets = mx.array(list(answer_tokens))[None, :]
    losses = [
        nn.losses.cross_entropy(logits, targets, reduction="mean")
        for logits in forward.branch_logits
    ]
    return sum(losses) / len(losses)


def exact_adjoint_live_path_value_and_grad(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    bridge_tokens: Sequence[int] = (),
    diversity_weight: float = 0.0,
    diversity_target_cos: float = 0.98,
) -> tuple[float, Any, float, list[float]]:
    """Compute the exact live-path gradient with bounded graph residency.

    Only recurrent LoRA parameters are trainable. Prelude outputs are therefore
    parameter-independent boundary values. Recurrence states are materialized
    between transitions, then exact vector-Jacobian products replay those
    transitions in reverse. Terminal branch losses are differentiated one at a
    time and their parameter/state gradients are accumulated algebraically.
    """

    import math
    import re

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_map

    if (
        isinstance(diversity_weight, bool)
        or not isinstance(diversity_weight, (int, float))
        or not math.isfinite(float(diversity_weight))
        or not 0.0 <= float(diversity_weight) <= 10.0
    ):
        raise ValueError("diversity_weight must be inside [0, 10]")
    parameters = model.trainable_parameters()
    prepared = _prepare_live_path(
        model,
        prompt_tokens,
        answer_tokens,
        spec=spec,
        bridge_tokens=bridge_tokens,
    )
    layer_pattern = re.compile(r"model\.layers\.(\d+)\.")
    for path, _value in tree_flatten(parameters):
        match = layer_pattern.match(path)
        if match is None or not (
            prepared.prelude_end <= int(match.group(1)) < prepared.coda_start
        ):
            raise RuntimeError("exact_adjoint_requires_window_only_trainables")

    def detached(values: Sequence[Any]) -> tuple[Any, ...]:
        result = tuple(mx.stop_gradient(value) for value in values)
        mx.eval(result)
        return result

    prompt_embeddings = mx.stop_gradient(prepared.prompt_embeddings)
    tail_embeddings = mx.stop_gradient(prepared.tail_embeddings)
    seeds = detached(prepared.seeds)
    mx.eval(prompt_embeddings, tail_embeddings)
    prompts = detached(prepared.prompts_at_window)
    anchors = detached(prepared.anchors)
    history: list[tuple[Any, ...]] = [detached(prepared.states)]
    current = history[0]
    for step in range(spec.recurrent_steps):
        outputs = _advance_recurrent_states(
            model,
            prompts,
            current,
            anchors,
            spec,
            step,
            prepared.prelude_end,
            prepared.coda_start,
        )
        current = detached(outputs)
        history.append(current)
        del outputs
        mx.clear_cache()

    accumulated: Any | None = None

    def add_parameter_gradient(gradient: Any, scale: float = 1.0) -> None:
        nonlocal accumulated
        scaled = tree_map(lambda value: scale * value, gradient)
        accumulated = (
            scaled
            if accumulated is None
            else tree_map(lambda left, right: left + right, accumulated, scaled)
        )
        mx.eval(accumulated)

    targets = mx.array(list(answer_tokens))[None, :]
    branch_scale = 1.0 / len(current)
    branch_values: list[float] = []
    state_cotangents: list[Any] = []
    for seed, state in zip(seeds, current, strict=True):

        def terminal_loss(
            parameter_tree: Any,
            final_state: Any,
            _seed: Any = seed,
        ) -> Any:
            model.update(parameter_tree)
            logits = _persist_and_score(
                model,
                prompt_embeddings,
                _seed,
                final_state,
                tail_embeddings,
                bridge_count=prepared.bridge_count,
                answer_count=prepared.answer_count,
                prelude_end=prepared.prelude_end,
                coda_start=prepared.coda_start,
            )
            return nn.losses.cross_entropy(logits, targets, reduction="mean")

        value, (parameter_gradient, state_gradient) = mx.value_and_grad(
            terminal_loss,
            argnums=(0, 1),
        )(parameters, state)
        mx.eval(value, parameter_gradient, state_gradient)
        branch_values.append(float(value))
        add_parameter_gradient(parameter_gradient, branch_scale)
        state_cotangents.append(mx.stop_gradient(branch_scale * state_gradient))
        del value, parameter_gradient, state_gradient
        mx.clear_cache()

    from core.learning.recurrence_native_objective_v3 import (
        branch_diversity_penalty,
    )

    terminal_forward = LivePathForward(
        branch_logits=(),
        branch_states=current,
        exchanges=0,
        prompt_tokens=prepared.prompt_count,
        answer_tokens=prepared.answer_count,
        bridge_tokens=prepared.bridge_count,
    )
    diversity_penalty, cosines = branch_diversity_penalty(
        terminal_forward,
        target_cos=diversity_target_cos,
    )
    mx.eval(diversity_penalty)
    diversity_value = float(diversity_penalty)
    if float(diversity_weight) > 0.0:

        def diversity_loss(final_states: tuple[Any, ...]) -> Any:
            forward = LivePathForward(
                branch_logits=(),
                branch_states=final_states,
                exchanges=0,
                prompt_tokens=prepared.prompt_count,
                answer_tokens=prepared.answer_count,
                bridge_tokens=prepared.bridge_count,
            )
            penalty, _cosines = branch_diversity_penalty(
                forward,
                target_cos=diversity_target_cos,
            )
            return penalty

        _value, diversity_gradients = mx.value_and_grad(diversity_loss)(current)
        mx.eval(diversity_gradients)
        state_cotangents = [
            mx.stop_gradient(existing + float(diversity_weight) * diversity)
            for existing, diversity in zip(
                state_cotangents,
                diversity_gradients,
                strict=True,
            )
        ]
        mx.eval(state_cotangents)
        del diversity_gradients
        mx.clear_cache()

    cotangents = tuple(state_cotangents)
    for step in range(spec.recurrent_steps - 1, -1, -1):
        input_states = history[step]

        def transition_pullback(
            parameter_tree: Any,
            prior_states: tuple[Any, ...],
            _step: int = step,
            _cotangents: tuple[Any, ...] = cotangents,
        ) -> Any:
            model.update(parameter_tree)
            outputs = _advance_recurrent_states(
                model,
                prompts,
                prior_states,
                anchors,
                spec,
                _step,
                prepared.prelude_end,
                prepared.coda_start,
            )
            return sum(
                mx.sum(output * cotangent)
                for output, cotangent in zip(outputs, _cotangents, strict=True)
            )

        _pullback, (parameter_gradient, input_cotangents) = mx.value_and_grad(
            transition_pullback,
            argnums=(0, 1),
        )(parameters, input_states)
        mx.eval(parameter_gradient, input_cotangents)
        add_parameter_gradient(parameter_gradient)
        cotangents = detached(input_cotangents)
        del parameter_gradient, input_cotangents
        mx.clear_cache()

    if accumulated is None:
        raise RuntimeError("exact adjoint parameter gradient is empty")
    base_value = sum(branch_values) / len(branch_values)
    total_value = base_value + float(diversity_weight) * diversity_value
    return total_value, accumulated, base_value, [float(value) for value in cosines]


def live_path_loss(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    bridge_tokens: Sequence[int] = (),
) -> Any:
    forward = live_path_forward(
        model,
        prompt_tokens,
        answer_tokens,
        spec=spec,
        bridge_tokens=bridge_tokens,
    )
    return branch_mean_answer_loss(forward, answer_tokens)


def detached_monotonicity_penalty(losses: Sequence[Any]) -> Any:
    """Penalize deeper regression without rewarding damage to shallow depth."""

    import mlx.core as mx

    if len(losses) < 2:
        raise ValueError("monotonicity penalty needs at least two depths")
    penalty = mx.zeros(())
    for shallow, deep in zip(losses, losses[1:], strict=False):
        penalty = penalty + mx.maximum(deep - mx.stop_gradient(shallow), 0.0)
    return penalty


def depth_curriculum_loss_v2(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    depths: tuple[int, ...] = (1, 2, 4),
    monotonicity_weight: float = 0.5,
    bridge_tokens: Sequence[int] = (),
) -> Any:
    """Answer CE over a depth ladder plus a shallow-detached monotonic hinge."""

    if (
        len(depths) < 2
        or any(type(depth) is not int or depth < 1 for depth in depths)
        or tuple(sorted(set(depths))) != depths
    ):
        raise ValueError("depths must be a strictly increasing tuple")
    if (
        isinstance(monotonicity_weight, bool)
        or not isinstance(monotonicity_weight, (int, float))
        or not 0.0 <= float(monotonicity_weight) <= 10.0
    ):
        raise ValueError("monotonicity_weight must be inside [0, 10]")
    losses = [
        live_path_loss(
            model,
            prompt_tokens,
            answer_tokens,
            spec=spec.with_depth(depth),
            bridge_tokens=bridge_tokens,
        )
        for depth in depths
    ]
    return sum(losses) / len(losses) + float(
        monotonicity_weight
    ) * detached_monotonicity_penalty(losses)


__all__ = [
    "LivePathForward",
    "RECURRENCE_NATIVE_SCHEMA_V2",
    "branch_mean_answer_loss",
    "depth_curriculum_loss_v2",
    "detached_monotonicity_penalty",
    "exact_adjoint_live_path_value_and_grad",
    "live_path_forward",
    "live_path_loss",
]
