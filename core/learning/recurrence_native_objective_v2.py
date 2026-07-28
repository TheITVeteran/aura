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

import hashlib
import inspect
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from core.brain.llm.latent_cortex.branch_exchange import private_exchange_slots
from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.brain.llm.latent_cortex.loop_core import (
    alpha_for_step,
    build_loop_core_contract,
    controlled_recurrent_update,
)
from core.brain.llm.latent_cortex.recurrence_adapter import (
    current_recurrence_adapter_scope,
    recurrence_adapter_scope,
)
from core.brain.llm.latent_cortex.types import WorkspaceConfig
from core.brain.llm.latent_cortex.workspace import LatentWorkspace, per_position_rms

RECURRENCE_NATIVE_SCHEMA_V2 = "aura.recurrence_native_objective.v2"
RECURRENT_TRANSITION_STATE_SCHEMA = "aura.recurrent_transition_state.v1"
EXACT_ADJOINT_TRAJECTORY_SCHEMA = "aura.exact_adjoint_trajectory_objective.v1"


@dataclass(frozen=True, slots=True)
class ExactAdjointTrajectoryConfig:
    """Auxiliary trajectory terms replayed through the bounded exact adjoint.

    The terminal policy objective and these terms remain separate. Callers can
    therefore measure each term's gradient in isolation before admitting a
    composite, while the resident path still keeps only one recurrent
    transition graph live at a time.
    """

    probe_steps: tuple[int, ...] = (1, 2)
    improvement_weight: float = 0.0
    improvement_margin: float = 0.02
    displacement_weight: float = 0.0
    displacement_floor: float = 0.01
    oscillation_weight: float = 0.0

    def __post_init__(self) -> None:
        if (
            not self.probe_steps
            or any(type(step) is not int or step < 1 for step in self.probe_steps)
            or tuple(sorted(set(self.probe_steps))) != self.probe_steps
        ):
            raise ValueError("probe_steps must be strictly increasing positive integers")
        for name, value, high in (
            ("improvement_weight", self.improvement_weight, 100.0),
            ("improvement_margin", self.improvement_margin, 10.0),
            ("displacement_weight", self.displacement_weight, 100.0),
            ("displacement_floor", self.displacement_floor, 1.0),
            ("oscillation_weight", self.oscillation_weight, 100.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= high
            ):
                raise ValueError(f"{name} must be finite inside [0, {high:g}]")
        if float(self.improvement_weight) > 0.0 and len(self.probe_steps) < 2:
            raise ValueError("improvement requires at least two probe steps")
        if not any(
            float(weight) > 0.0
            for weight in (
                self.improvement_weight,
                self.displacement_weight,
                self.oscillation_weight,
            )
        ):
            raise ValueError("trajectory objective must enable at least one term")

    def validate_depth(self, depth: int) -> None:
        if self.probe_steps[-1] > depth:
            raise ValueError("trajectory probe step exceeds recurrent depth")
        if float(self.oscillation_weight) > 0.0 and depth < 2:
            raise ValueError("oscillation objective requires at least two transitions")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EXACT_ADJOINT_TRAJECTORY_SCHEMA,
            "probe_steps": list(self.probe_steps),
            "improvement_weight": float(self.improvement_weight),
            "improvement_margin": float(self.improvement_margin),
            "displacement_weight": float(self.displacement_weight),
            "displacement_floor": float(self.displacement_floor),
            "oscillation_weight": float(self.oscillation_weight),
        }

    @classmethod
    def from_dict(cls, value: Any) -> ExactAdjointTrajectoryConfig:
        required = {
            "schema",
            "probe_steps",
            "improvement_weight",
            "improvement_margin",
            "displacement_weight",
            "displacement_floor",
            "oscillation_weight",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("trajectory objective config fields do not match")
        if value.get("schema") != EXACT_ADJOINT_TRAJECTORY_SCHEMA:
            raise ValueError("trajectory objective config schema is unsupported")
        probe_steps = value.get("probe_steps")
        if not isinstance(probe_steps, list):
            raise ValueError("trajectory objective probe_steps must be a list")
        return cls(
            probe_steps=tuple(probe_steps),
            improvement_weight=value["improvement_weight"],
            improvement_margin=value["improvement_margin"],
            displacement_weight=value["displacement_weight"],
            displacement_floor=value["displacement_floor"],
            oscillation_weight=value["oscillation_weight"],
        )


@dataclass(frozen=True, slots=True)
class ExactAdjointLivePathResult:
    """One exact-adjoint value/gradient result with replayable term telemetry."""

    value: float
    gradients: Any
    terminal_value: float
    diversity_value: float
    trajectory_values: Mapping[str, float]
    step_losses: Mapping[int, tuple[float, ...]]
    displacements: tuple[float, ...]
    oscillation_cosines: tuple[float, ...]
    diversity_cosines: tuple[float, ...]
    branch_indices: tuple[int, ...]
    trajectory_config: ExactAdjointTrajectoryConfig | None
    execution_spec_sha256: str
    recurrent_depth: int
    execution_branch_count: int
    diversity_weight: float
    diversity_target_cos: float

    def receipt(self) -> dict[str, Any]:
        payload = {
            "schema": EXACT_ADJOINT_TRAJECTORY_SCHEMA,
            "value": float(self.value),
            "terminal_value": float(self.terminal_value),
            "diversity_value": float(self.diversity_value),
            "trajectory_values": {
                name: float(value) for name, value in sorted(self.trajectory_values.items())
            },
            "step_losses": {
                str(step): [float(value) for value in values]
                for step, values in sorted(self.step_losses.items())
            },
            "displacements": [float(value) for value in self.displacements],
            "oscillation_cosines": [float(value) for value in self.oscillation_cosines],
            "diversity_cosines": [float(value) for value in self.diversity_cosines],
            "branch_indices": list(self.branch_indices),
            "execution_spec_sha256": self.execution_spec_sha256,
            "recurrent_depth": self.recurrent_depth,
            "execution_branch_count": self.execution_branch_count,
            "diversity_weight": self.diversity_weight,
            "diversity_target_cos": self.diversity_target_cos,
            "trajectory_config": (
                self.trajectory_config.to_dict() if self.trajectory_config is not None else None
            ),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return {**payload, "receipt_sha256": hashlib.sha256(encoded).hexdigest()}


def validate_exact_adjoint_live_path_receipt(value: Any) -> dict[str, Any]:
    """Independently replay an exact-adjoint trajectory objective receipt."""

    required = {
        "schema",
        "value",
        "terminal_value",
        "diversity_value",
        "trajectory_values",
        "step_losses",
        "displacements",
        "oscillation_cosines",
        "diversity_cosines",
        "branch_indices",
        "execution_spec_sha256",
        "recurrent_depth",
        "execution_branch_count",
        "diversity_weight",
        "diversity_target_cos",
        "trajectory_config",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("exact-adjoint trajectory receipt fields do not match")
    receipt = dict(value)
    observed = receipt.pop("receipt_sha256")
    encoded = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    if not isinstance(observed, str) or observed != hashlib.sha256(encoded).hexdigest():
        raise ValueError("exact-adjoint trajectory receipt commitment mismatch")
    if receipt["schema"] != EXACT_ADJOINT_TRAJECTORY_SCHEMA:
        raise ValueError("exact-adjoint trajectory receipt schema is unsupported")
    digest = receipt["execution_spec_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("exact-adjoint execution spec digest is invalid")
    depth = receipt["recurrent_depth"]
    branch_count = receipt["execution_branch_count"]
    branches = receipt["branch_indices"]
    if (
        type(depth) is not int
        or depth < 1
        or type(branch_count) is not int
        or branch_count < 1
        or not isinstance(branches, list)
        or not branches
        or any(type(index) is not int or index < 0 for index in branches)
        or any(index >= branch_count for index in branches)
        or len(set(branches)) != len(branches)
    ):
        raise ValueError("exact-adjoint depth or branch identity is invalid")

    def finite_number(item: Any, *, role: str) -> float:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"exact-adjoint {role} is not finite")
        return float(item)

    terminal = finite_number(receipt["terminal_value"], role="terminal value")
    diversity = finite_number(receipt["diversity_value"], role="diversity value")
    diversity_weight = finite_number(receipt["diversity_weight"], role="diversity weight")
    diversity_target = finite_number(receipt["diversity_target_cos"], role="diversity target")
    if not 0.0 <= diversity_weight <= 10.0 or not 0.0 <= diversity_target <= 1.0:
        raise ValueError("exact-adjoint diversity configuration is invalid")
    total = finite_number(receipt["value"], role="total value")
    terms = receipt["trajectory_values"]
    if not isinstance(terms, Mapping) or set(terms) != {
        "improvement",
        "displacement",
        "oscillation",
    }:
        raise ValueError("exact-adjoint trajectory term set is invalid")
    term_values = {
        str(name): finite_number(number, role=f"{name} value") for name, number in terms.items()
    }
    expected_total = terminal + diversity + sum(term_values.values())
    if not math.isclose(total, expected_total, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("exact-adjoint total does not replay from its terms")

    config_value = receipt["trajectory_config"]
    config = (
        ExactAdjointTrajectoryConfig.from_dict(config_value) if config_value is not None else None
    )
    if config is not None:
        config.validate_depth(depth)
    step_losses = receipt["step_losses"]
    if not isinstance(step_losses, Mapping):
        raise ValueError("exact-adjoint step losses must be a mapping")
    normalized_steps: dict[int, list[Any]] = {}
    for key, losses in step_losses.items():
        if not isinstance(key, str) or not key.isdigit() or not isinstance(losses, list):
            raise ValueError("exact-adjoint step-loss row is invalid")
        normalized_steps[int(key)] = losses
        if len(losses) != len(branches):
            raise ValueError("exact-adjoint step-loss branches do not align")
        for loss in losses:
            finite_number(loss, role="step loss")
    expected_steps = (
        set(config.probe_steps)
        if config is not None and float(config.improvement_weight) > 0.0
        else set()
    )
    if set(normalized_steps) != expected_steps:
        raise ValueError("exact-adjoint step-loss probes do not match the config")

    for role in ("displacements", "oscillation_cosines", "diversity_cosines"):
        sequence = receipt[role]
        if not isinstance(sequence, list):
            raise ValueError(f"exact-adjoint {role} must be a list")
        for item in sequence:
            finite_number(item, role=role)
    expected_diversity_count = branch_count * (branch_count - 1) // 2
    if len(receipt["diversity_cosines"]) != expected_diversity_count:
        raise ValueError("exact-adjoint diversity cardinality is invalid")
    replayed_diversity = (
        diversity_weight
        * sum(
            max(float(cosine) - diversity_target, 0.0) ** 2
            for cosine in receipt["diversity_cosines"]
        )
        / expected_diversity_count
        if expected_diversity_count
        else 0.0
    )
    if not math.isclose(
        diversity,
        replayed_diversity,
        rel_tol=0.0,
        # The producer evaluates the penalty in MLX float32 while this replay
        # uses the sealed Python floats. The tolerance covers that one
        # representation crossing, not a statistical or model-level margin.
        abs_tol=1e-6,
    ):
        raise ValueError("exact-adjoint diversity does not replay")
    expected_displacements = (
        depth * len(branches)
        if config is not None and float(config.displacement_weight) > 0.0
        else 0
    )
    expected_oscillations = (
        (depth - 1) * len(branches)
        if config is not None and float(config.oscillation_weight) > 0.0
        else 0
    )
    if len(receipt["displacements"]) != expected_displacements:
        raise ValueError("exact-adjoint displacement cardinality is invalid")
    if len(receipt["oscillation_cosines"]) != expected_oscillations:
        raise ValueError("exact-adjoint oscillation cardinality is invalid")
    if config is None and any(abs(number) > 0.0 for number in term_values.values()):
        raise ValueError("exact-adjoint receipt has terms without a trajectory config")
    return dict(value)


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
    loop_core: dict[str, Any] = field(default_factory=dict)


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


@dataclass(frozen=True, slots=True)
class PreparedFinalRecurrentTransition:
    """Frozen parent and child ensembles around the final recurrent update.

    The object carries tensors for immediate decode and a tensor-free receipt
    for durable custody.  ``child_states`` are computed from ``parent_states``
    by exactly one invocation of the live transition operator; neither state
    is reconstructed from text or from a second independent episode.
    """

    prompt_embeddings: Any
    seeds: tuple[Any, ...]
    parent_states: tuple[Any, ...]
    child_states: tuple[Any, ...]
    prelude_end: int
    coda_start: int
    transition_index: int
    execution_spec_sha256: str
    prompt_tokens_sha256: str
    parent_branch_sha256s: tuple[str, ...]
    child_branch_sha256s: tuple[str, ...]
    parent_ensemble_sha256: str
    child_ensemble_sha256: str
    transition_source_sha256: str
    receipt_sha256: str

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": RECURRENT_TRANSITION_STATE_SCHEMA,
            "execution_spec_sha256": self.execution_spec_sha256,
            "prompt_tokens_sha256": self.prompt_tokens_sha256,
            "transition_index": self.transition_index,
            "parent_depth": self.transition_index,
            "child_depth": self.transition_index + 1,
            "branch_count": len(self.parent_states),
            "parent_branch_sha256s": list(self.parent_branch_sha256s),
            "child_branch_sha256s": list(self.child_branch_sha256s),
            "parent_ensemble_sha256": self.parent_ensemble_sha256,
            "child_ensemble_sha256": self.child_ensemble_sha256,
            "transition_source_sha256": self.transition_source_sha256,
            "receipt_sha256": self.receipt_sha256,
        }


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
    frequency = mx.exp(-math.log(10000.0) * (2 * mx.floor(positions / 2)) / hidden)
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
    return alpha_for_step(
        alpha=spec.alpha,
        schedule=spec.alpha_schedule,
        max_steps=spec.recurrent_steps,
        step=step,
    )


def _exchange_and_decorrelate(
    states: list[Any],
    spec: RLCExecutionSpec,
    step_number: int,
) -> list[Any]:
    import mlx.core as mx

    if len(states) < 2:
        return states
    source_slots = private_exchange_slots(
        n_slots=int(states[0].shape[1]),
        comm_slot=int(spec.comm_slot),
        context_slots=(),
    )
    if len(source_slots) > spec.exchange_source_slot_limit:
        raise ValueError("training exchange source exceeds execution spec")
    summaries = [
        mx.mean(
            mx.concatenate(
                [state[:, index : index + 1, :] for index in source_slots],
                axis=1,
            ),
            axis=1,
            keepdims=True,
        )
        for state in states
    ]
    stack = mx.concatenate(summaries, axis=1)
    mean = mx.mean(stack, axis=1, keepdims=True)

    def cosine(left: Any, right: Any) -> Any:
        denominator = mx.maximum(mx.linalg.norm(left) * mx.linalg.norm(right), 1e-6)
        return mx.sum(left * right) / denominator

    agreements = mx.stack([cosine(summary, mean) for summary in summaries])
    weights = mx.softmax(agreements, axis=0)
    consensus = sum(weight * summary for weight, summary in zip(weights, summaries, strict=True))
    slot = spec.comm_slot
    exchanged: list[Any] = []
    for state in states:
        comm = (1.0 - spec.exchange_gamma) * state[
            :, slot : slot + 1, :
        ] + spec.exchange_gamma * consensus
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
            key = mx.random.key(1000 + 31 * left_index + right_index + step_number)
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
            controlled_recurrent_update(
                state,
                candidate,
                anchor,
                alpha=alpha,
                clip_ratio=spec.rms_clip_ratio,
            )
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


def _token_sequence_sha256(tokens: Sequence[int]) -> str:
    encoded = json.dumps(list(tokens), separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: Any) -> str:
    import mlx.core as mx
    import numpy as np

    mx.eval(value)
    try:
        array = np.asarray(value)
    except RuntimeError:
        array = np.asarray(value.astype(mx.float32))
    digest = hashlib.sha256()
    for part in (
        str(value.dtype).encode("ascii"),
        json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"),
        array.tobytes(order="C"),
    ):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _ensemble_sha256(branch_sha256s: Sequence[str]) -> str:
    encoded = json.dumps(list(branch_sha256s), separators=(",", ":"), allow_nan=False).encode(
        "ascii"
    )
    return hashlib.sha256(b"aura.recurrent_ensemble.v1\0" + encoded).hexdigest()


def _seal_transition_receipt(body: dict[str, Any]) -> str:
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _prepare_recurrent_prefix(
    model: Any,
    prompt_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
) -> tuple[
    Any,
    tuple[Any, ...],
    tuple[Any, ...],
    tuple[Any, ...],
    tuple[Any, ...],
    int,
    int,
]:
    import mlx.core as mx

    problems = spec.validate()
    if problems:
        raise ValueError(f"invalid execution spec: {problems}")
    prompt = list(prompt_tokens)
    if not prompt or any(type(token) is not int or token < 0 for token in prompt):
        raise ValueError("prompt_tokens must contain non-negative integers")
    _n_layers, prelude_end, coda_start = _boundaries(model, spec)
    prompt_embeddings = model.model.embed_tokens(mx.array([prompt]))
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
    return (
        prompt_embeddings,
        tuple(seeds),
        tuple(prompts_at_window),
        tuple(states),
        tuple(anchors),
        prelude_end,
        coda_start,
    )


def _prepare_live_path(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    bridge_tokens: Sequence[int],
) -> _PreparedLivePath:
    import mlx.core as mx

    prompt = list(prompt_tokens)
    answer = list(answer_tokens)
    bridge = list(bridge_tokens)
    if not answer or any(type(token) is not int or token < 0 for token in answer):
        raise ValueError("answer_tokens must contain non-negative integers")
    if any(type(token) is not int or token < 0 for token in bridge):
        raise ValueError("bridge_tokens must contain non-negative integers")
    if spec.decode_bridge_policy == "none" and bridge:
        raise ValueError("bridge tokens supplied while decode bridge is disabled")
    if spec.decode_bridge_policy != "none" and not bridge:
        raise ValueError("execution spec requires decode bridge tokens")

    (
        prompt_embeddings,
        seeds,
        prompts_at_window,
        states,
        anchors,
        prelude_end,
        coda_start,
    ) = _prepare_recurrent_prefix(model, prompt, spec=spec)
    tail_embeddings = model.model.embed_tokens(mx.array([bridge + answer]))
    return _PreparedLivePath(
        prompt_embeddings=prompt_embeddings,
        tail_embeddings=tail_embeddings,
        seeds=seeds,
        prompts_at_window=prompts_at_window,
        states=states,
        anchors=anchors,
        prelude_end=prelude_end,
        coda_start=coda_start,
        prompt_count=len(prompt),
        bridge_count=len(bridge),
        answer_count=len(answer),
    )


def prepare_final_recurrent_transition(
    model: Any,
    prompt_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
) -> PreparedFinalRecurrentTransition:
    """Freeze ``S[k]`` and ``S[k+1]`` around the final configured update."""

    import mlx.core as mx

    (
        prompt_embeddings,
        seeds,
        prompts_at_window,
        initial_states,
        anchors,
        prelude_end,
        coda_start,
    ) = _prepare_recurrent_prefix(model, prompt_tokens, spec=spec)
    states = list(initial_states)
    transition_index = spec.recurrent_steps - 1
    for step in range(transition_index):
        states = _checkpointed_recurrent_transition(
            model,
            prompts_at_window,
            states,
            anchors,
            spec,
            step,
            prelude_end,
            coda_start,
        )
    parent_states = tuple(mx.stop_gradient(state) for state in states)
    mx.eval(parent_states)
    child = _checkpointed_recurrent_transition(
        model,
        prompts_at_window,
        parent_states,
        anchors,
        spec,
        transition_index,
        prelude_end,
        coda_start,
    )
    child_states = tuple(mx.stop_gradient(state) for state in child)
    mx.eval(child_states)
    parent_branch_sha256s = tuple(_tensor_sha256(state) for state in parent_states)
    child_branch_sha256s = tuple(_tensor_sha256(state) for state in child_states)
    transition_source_sha256 = hashlib.sha256(
        inspect.getsource(_advance_recurrent_states).encode("utf-8")
    ).hexdigest()
    body = {
        "schema": RECURRENT_TRANSITION_STATE_SCHEMA,
        "execution_spec_sha256": spec.sha256,
        "prompt_tokens_sha256": _token_sequence_sha256(prompt_tokens),
        "transition_index": transition_index,
        "parent_depth": transition_index,
        "child_depth": transition_index + 1,
        "branch_count": len(parent_states),
        "parent_branch_sha256s": list(parent_branch_sha256s),
        "child_branch_sha256s": list(child_branch_sha256s),
        "parent_ensemble_sha256": _ensemble_sha256(parent_branch_sha256s),
        "child_ensemble_sha256": _ensemble_sha256(child_branch_sha256s),
        "transition_source_sha256": transition_source_sha256,
    }
    receipt_sha256 = _seal_transition_receipt(body)
    return PreparedFinalRecurrentTransition(
        prompt_embeddings=mx.stop_gradient(prompt_embeddings),
        seeds=tuple(mx.stop_gradient(seed) for seed in seeds),
        parent_states=parent_states,
        child_states=child_states,
        prelude_end=prelude_end,
        coda_start=coda_start,
        transition_index=transition_index,
        execution_spec_sha256=spec.sha256,
        prompt_tokens_sha256=body["prompt_tokens_sha256"],
        parent_branch_sha256s=parent_branch_sha256s,
        child_branch_sha256s=child_branch_sha256s,
        parent_ensemble_sha256=body["parent_ensemble_sha256"],
        child_ensemble_sha256=body["child_ensemble_sha256"],
        transition_source_sha256=transition_source_sha256,
        receipt_sha256=receipt_sha256,
    )


def validate_final_recurrent_transition_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the durable, tensor-free edge certificate."""

    required = {
        "schema",
        "execution_spec_sha256",
        "prompt_tokens_sha256",
        "transition_index",
        "parent_depth",
        "child_depth",
        "branch_count",
        "parent_branch_sha256s",
        "child_branch_sha256s",
        "parent_ensemble_sha256",
        "child_ensemble_sha256",
        "transition_source_sha256",
        "receipt_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required:
        raise ValueError("recurrent_transition_receipt_schema_invalid")
    normalized = dict(receipt)
    if normalized.get("schema") != RECURRENT_TRANSITION_STATE_SCHEMA:
        raise ValueError("recurrent_transition_receipt_version_invalid")
    branch_count = normalized.get("branch_count")
    transition_index = normalized.get("transition_index")
    parent = normalized.get("parent_branch_sha256s")
    child = normalized.get("child_branch_sha256s")
    digests = (
        normalized.get("execution_spec_sha256"),
        normalized.get("prompt_tokens_sha256"),
        normalized.get("parent_ensemble_sha256"),
        normalized.get("child_ensemble_sha256"),
        normalized.get("transition_source_sha256"),
        normalized.get("receipt_sha256"),
    )
    if (
        type(branch_count) is not int
        or branch_count < 1
        or type(transition_index) is not int
        or transition_index < 0
        or normalized.get("parent_depth") != transition_index
        or normalized.get("child_depth") != transition_index + 1
        or not isinstance(parent, list)
        or not isinstance(child, list)
        or len(parent) != branch_count
        or len(child) != branch_count
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in (*digests, *parent, *child)
        )
        or normalized["parent_ensemble_sha256"] != _ensemble_sha256(parent)
        or normalized["child_ensemble_sha256"] != _ensemble_sha256(child)
    ):
        raise ValueError("recurrent_transition_receipt_identity_invalid")
    unsigned = dict(normalized)
    observed = unsigned.pop("receipt_sha256")
    if _seal_transition_receipt(unsigned) != observed:
        raise ValueError("recurrent_transition_receipt_digest_mismatch")
    return normalized


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
        loop_core=build_loop_core_contract(
            prelude_end=prepared.prelude_end,
            coda_start=prepared.coda_start,
            max_steps=spec.recurrent_steps,
            min_steps=spec.recurrent_steps,
            alpha=spec.alpha,
            alpha_schedule=spec.alpha_schedule,
            rms_clip_ratio=spec.rms_clip_ratio,
            convergence_eps=1e-9,
            divergence_ratio=1000.0,
            fixed_depth=True,
        ),
    )


def live_path_branch_answer_ce_trail(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    branch_index: int,
    bridge_tokens: Sequence[int] = (),
) -> list[float]:
    """Answer CE after each recurrent step for one live-path branch.

    GRPO's final verifier can mark an entire sampled group wrong, leaving
    zero group-relative advantage. For recurrence-native training, that wastes
    the most important early signal: which internal state trajectories moved
    toward the known correct answer before the sampled decode missed. This
    function measures that signal on the same live recurrent graph used by the
    exact-adjoint objective. It is telemetry/credit assignment only; it does
    not replace the external verifier.
    """

    import mlx.core as mx
    import mlx.nn as nn

    prepared = _prepare_live_path(
        model,
        prompt_tokens,
        answer_tokens,
        spec=spec,
        bridge_tokens=bridge_tokens,
    )
    if type(branch_index) is not int or not 0 <= branch_index < len(prepared.states):
        raise ValueError("branch_index is outside the live-path branch set")

    targets = mx.array(list(answer_tokens))[None, :]
    states = list(prepared.states)
    trail: list[float] = []
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
        logits = _persist_and_score(
            model,
            prepared.prompt_embeddings,
            prepared.seeds[branch_index],
            states[branch_index],
            prepared.tail_embeddings,
            bridge_count=prepared.bridge_count,
            answer_count=prepared.answer_count,
            prelude_end=prepared.prelude_end,
            coda_start=prepared.coda_start,
        )
        losses = nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="none")
        value = mx.mean(losses)
        mx.eval(value)
        trail.append(float(value))
        del logits, losses, value
        mx.clear_cache()
    return trail


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


def _exact_adjoint_live_path_result(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    bridge_tokens: Sequence[int] = (),
    diversity_weight: float = 0.0,
    diversity_target_cos: float = 0.98,
    token_loss_weights: Sequence[float] | None = None,
    branch_index: int | None = None,
    trajectory_config: ExactAdjointTrajectoryConfig | None = None,
) -> ExactAdjointLivePathResult:
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
    if token_loss_weights is None:
        normalized_token_weights = (1.0,) * len(answer_tokens)
    else:
        normalized_token_weights = tuple(float(value) for value in token_loss_weights)
        if len(normalized_token_weights) != len(answer_tokens) or any(
            not math.isfinite(value) for value in normalized_token_weights
        ):
            raise ValueError("token_loss_weights must align and be finite")
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
        if match is None or not (prepared.prelude_end <= int(match.group(1)) < prepared.coda_start):
            raise RuntimeError("exact_adjoint_requires_window_only_trainables")
    if branch_index is not None and (
        type(branch_index) is not int or not 0 <= branch_index < len(prepared.states)
    ):
        raise ValueError("branch_index is outside the live-path branch set")
    if trajectory_config is not None:
        if not isinstance(trajectory_config, ExactAdjointTrajectoryConfig):
            raise TypeError("trajectory_config must be an ExactAdjointTrajectoryConfig")
        trajectory_config.validate_depth(spec.recurrent_steps)

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
    token_weights = mx.array(normalized_token_weights, dtype=mx.float32)[None, :]
    selected_indices = tuple(range(len(current))) if branch_index is None else (branch_index,)
    branch_scale = 1.0 / len(selected_indices)
    branch_values: list[float] = []
    direct_cotangents: list[list[Any]] = [
        [mx.zeros_like(state) for state in states] for states in history
    ]
    for selected_index in selected_indices:
        seed = seeds[selected_index]
        state = current[selected_index]

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
            token_losses = nn.losses.cross_entropy(
                logits.astype(mx.float32), targets, reduction="none"
            )
            return mx.mean(token_losses * token_weights)

        value, (parameter_gradient, state_gradient) = mx.value_and_grad(
            terminal_loss,
            argnums=(0, 1),
        )(parameters, state)
        mx.eval(value, parameter_gradient, state_gradient)
        branch_values.append(float(value))
        add_parameter_gradient(parameter_gradient, branch_scale)
        direct_cotangents[-1][selected_index] = mx.stop_gradient(
            direct_cotangents[-1][selected_index] + branch_scale * state_gradient
        )
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
        direct_cotangents[-1] = [
            mx.stop_gradient(existing + float(diversity_weight) * diversity)
            for existing, diversity in zip(direct_cotangents[-1], diversity_gradients, strict=True)
        ]
        mx.eval(direct_cotangents[-1])
        del diversity_gradients
        mx.clear_cache()

    trajectory_values = {
        "improvement": 0.0,
        "displacement": 0.0,
        "oscillation": 0.0,
    }
    step_loss_values: dict[int, tuple[float, ...]] = {}
    measured_displacements: list[float] = []
    measured_oscillation_cosines: list[float] = []
    if trajectory_config is not None:
        probe_losses: dict[int, list[float]] = {}
        probe_gradients: dict[int, list[tuple[Any, Any]]] = {}
        if float(trajectory_config.improvement_weight) > 0.0:
            for depth in trajectory_config.probe_steps:
                depth_losses: list[float] = []
                depth_gradients: list[tuple[Any, Any]] = []
                for selected_index in selected_indices:
                    seed = seeds[selected_index]

                    def intermediate_loss(
                        parameter_tree: Any,
                        state: Any,
                        _seed: Any = seed,
                    ) -> Any:
                        model.update(parameter_tree)
                        logits = _persist_and_score(
                            model,
                            prompt_embeddings,
                            _seed,
                            state,
                            tail_embeddings,
                            bridge_count=prepared.bridge_count,
                            answer_count=prepared.answer_count,
                            prelude_end=prepared.prelude_end,
                            coda_start=prepared.coda_start,
                        )
                        return nn.losses.cross_entropy(
                            logits.astype(mx.float32),
                            targets,
                            reduction="mean",
                        )

                    loss, (parameter_gradient, state_gradient) = mx.value_and_grad(
                        intermediate_loss,
                        argnums=(0, 1),
                    )(parameters, history[depth][selected_index])
                    mx.eval(loss, parameter_gradient, state_gradient)
                    depth_losses.append(float(loss))
                    depth_gradients.append(
                        (
                            parameter_gradient,
                            mx.stop_gradient(state_gradient),
                        )
                    )
                probe_losses[depth] = depth_losses
                probe_gradients[depth] = depth_gradients
                step_loss_values[depth] = tuple(depth_losses)

            hinge_count = (len(trajectory_config.probe_steps) - 1) * len(selected_indices)
            hinge_scale = float(trajectory_config.improvement_weight) / hinge_count
            improvement_value = 0.0
            for previous_depth, current_depth in zip(
                trajectory_config.probe_steps,
                trajectory_config.probe_steps[1:],
                strict=False,
            ):
                for offset, selected_index in enumerate(selected_indices):
                    hinge = max(
                        probe_losses[current_depth][offset]
                        - probe_losses[previous_depth][offset]
                        + float(trajectory_config.improvement_margin),
                        0.0,
                    )
                    improvement_value += hinge_scale * hinge
                    if hinge <= 0.0:
                        continue
                    parameter_gradient, state_gradient = probe_gradients[current_depth][offset]
                    add_parameter_gradient(parameter_gradient, hinge_scale)
                    direct_cotangents[current_depth][selected_index] = mx.stop_gradient(
                        direct_cotangents[current_depth][selected_index]
                        + hinge_scale * state_gradient
                    )
            trajectory_values["improvement"] = improvement_value

        if float(trajectory_config.displacement_weight) > 0.0:
            term_count = spec.recurrent_steps * len(selected_indices)
            term_scale = float(trajectory_config.displacement_weight) / term_count
            displacement_value = 0.0
            for depth in range(1, spec.recurrent_steps + 1):
                for selected_index in selected_indices:

                    def displacement_loss(previous: Any, current_state: Any) -> Any:
                        numerator = mx.linalg.norm(mx.reshape(current_state - previous, (-1,)))
                        denominator = mx.maximum(
                            mx.linalg.norm(mx.reshape(previous, (-1,))),
                            1e-9,
                        )
                        return mx.maximum(
                            float(trajectory_config.displacement_floor) - numerator / denominator,
                            0.0,
                        )

                    value, (previous_gradient, current_gradient) = mx.value_and_grad(
                        displacement_loss,
                        argnums=(0, 1),
                    )(
                        history[depth - 1][selected_index],
                        history[depth][selected_index],
                    )
                    mx.eval(value, previous_gradient, current_gradient)
                    previous_state = history[depth - 1][selected_index]
                    current_state = history[depth][selected_index]
                    displacement = float(
                        mx.linalg.norm(mx.reshape(current_state - previous_state, (-1,)))
                        / mx.maximum(
                            mx.linalg.norm(mx.reshape(previous_state, (-1,))),
                            1e-9,
                        )
                    )
                    measured_displacements.append(displacement)
                    displacement_value += term_scale * float(value)
                    direct_cotangents[depth - 1][selected_index] = mx.stop_gradient(
                        direct_cotangents[depth - 1][selected_index]
                        + term_scale * previous_gradient
                    )
                    direct_cotangents[depth][selected_index] = mx.stop_gradient(
                        direct_cotangents[depth][selected_index] + term_scale * current_gradient
                    )
            trajectory_values["displacement"] = displacement_value

        if float(trajectory_config.oscillation_weight) > 0.0:
            pair_count = (spec.recurrent_steps - 1) * len(selected_indices)
            pair_scale = float(trajectory_config.oscillation_weight) / pair_count
            oscillation_value = 0.0
            for depth in range(2, spec.recurrent_steps + 1):
                for selected_index in selected_indices:

                    def oscillation_loss(
                        previous: Any,
                        middle: Any,
                        current_state: Any,
                    ) -> Any:
                        first = mx.reshape(middle - previous, (-1,))
                        second = mx.reshape(current_state - middle, (-1,))
                        denominator = mx.maximum(
                            mx.linalg.norm(first) * mx.linalg.norm(second),
                            1e-9,
                        )
                        cosine = mx.sum(first * second) / denominator
                        return mx.maximum(-cosine, 0.0)

                    value, gradients = mx.value_and_grad(
                        oscillation_loss,
                        argnums=(0, 1, 2),
                    )(
                        history[depth - 2][selected_index],
                        history[depth - 1][selected_index],
                        history[depth][selected_index],
                    )
                    mx.eval(value, gradients)
                    first = mx.reshape(
                        history[depth - 1][selected_index] - history[depth - 2][selected_index],
                        (-1,),
                    )
                    second = mx.reshape(
                        history[depth][selected_index] - history[depth - 1][selected_index],
                        (-1,),
                    )
                    cosine = float(
                        mx.sum(first * second)
                        / mx.maximum(
                            mx.linalg.norm(first) * mx.linalg.norm(second),
                            1e-9,
                        )
                    )
                    measured_oscillation_cosines.append(cosine)
                    oscillation_value += pair_scale * float(value)
                    for state_depth, gradient in zip(
                        (depth - 2, depth - 1, depth),
                        gradients,
                        strict=True,
                    ):
                        direct_cotangents[state_depth][selected_index] = mx.stop_gradient(
                            direct_cotangents[state_depth][selected_index] + pair_scale * gradient
                        )
            trajectory_values["oscillation"] = oscillation_value

    cotangents = tuple(direct_cotangents[-1])
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
        cotangents = tuple(
            mx.stop_gradient(incoming + direct)
            for incoming, direct in zip(
                input_cotangents,
                direct_cotangents[step],
                strict=True,
            )
        )
        mx.eval(cotangents)
        del parameter_gradient, input_cotangents
        mx.clear_cache()

    if accumulated is None:
        raise RuntimeError("exact adjoint parameter gradient is empty")
    base_value = sum(branch_values) / len(branch_values)
    total_value = (
        base_value + float(diversity_weight) * diversity_value + sum(trajectory_values.values())
    )
    return ExactAdjointLivePathResult(
        value=total_value,
        gradients=accumulated,
        terminal_value=base_value,
        diversity_value=float(diversity_weight) * diversity_value,
        trajectory_values=trajectory_values,
        step_losses=step_loss_values,
        displacements=tuple(measured_displacements),
        oscillation_cosines=tuple(measured_oscillation_cosines),
        diversity_cosines=tuple(float(value) for value in cosines),
        branch_indices=selected_indices,
        trajectory_config=trajectory_config,
        execution_spec_sha256=spec.sha256,
        recurrent_depth=spec.recurrent_steps,
        execution_branch_count=len(current),
        diversity_weight=float(diversity_weight),
        diversity_target_cos=float(diversity_target_cos),
    )


def exact_adjoint_live_path_value_and_grad(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    bridge_tokens: Sequence[int] = (),
    diversity_weight: float = 0.0,
    diversity_target_cos: float = 0.98,
    token_loss_weights: Sequence[float] | None = None,
    branch_index: int | None = None,
) -> tuple[float, Any, float, list[float]]:
    """Compatibility surface for the terminal exact-adjoint objective."""

    result = _exact_adjoint_live_path_result(
        model,
        prompt_tokens,
        answer_tokens,
        spec=spec,
        bridge_tokens=bridge_tokens,
        diversity_weight=diversity_weight,
        diversity_target_cos=diversity_target_cos,
        token_loss_weights=token_loss_weights,
        branch_index=branch_index,
    )
    return (
        result.value,
        result.gradients,
        result.terminal_value,
        list(result.diversity_cosines),
    )


def exact_adjoint_trajectory_live_path_value_and_grad(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    trajectory_config: ExactAdjointTrajectoryConfig,
    bridge_tokens: Sequence[int] = (),
    branch_index: int | None = None,
    diversity_weight: float = 0.0,
    diversity_target_cos: float = 0.98,
    token_loss_weights: Sequence[float] | None = None,
) -> ExactAdjointLivePathResult:
    """Compute terminal and trajectory gradients with bounded graph residency."""

    return _exact_adjoint_live_path_result(
        model,
        prompt_tokens,
        answer_tokens,
        spec=spec,
        bridge_tokens=bridge_tokens,
        diversity_weight=diversity_weight,
        diversity_target_cos=diversity_target_cos,
        token_loss_weights=token_loss_weights,
        branch_index=branch_index,
        trajectory_config=trajectory_config,
    )


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
    return sum(losses) / len(losses) + float(monotonicity_weight) * detached_monotonicity_penalty(
        losses
    )


__all__ = [
    "EXACT_ADJOINT_TRAJECTORY_SCHEMA",
    "ExactAdjointLivePathResult",
    "ExactAdjointTrajectoryConfig",
    "LivePathForward",
    "PreparedFinalRecurrentTransition",
    "RECURRENCE_NATIVE_SCHEMA_V2",
    "RECURRENT_TRANSITION_STATE_SCHEMA",
    "branch_mean_answer_loss",
    "depth_curriculum_loss_v2",
    "detached_monotonicity_penalty",
    "exact_adjoint_live_path_value_and_grad",
    "exact_adjoint_trajectory_live_path_value_and_grad",
    "live_path_branch_answer_ce_trail",
    "live_path_forward",
    "live_path_loss",
    "prepare_final_recurrent_transition",
    "validate_exact_adjoint_live_path_receipt",
    "validate_final_recurrent_transition_receipt",
]
