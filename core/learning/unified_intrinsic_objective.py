"""Semantic training objective for Aura's unified intrinsic recurrence."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from core.learning.intrinsic_recurrence import RecurrentDepthPlan, _run
from core.learning.protected_memory import MemoryLayout
from core.learning.recurrent_action_schema import (
    ACTION_SLOT_NAMES,
    RecurrentActionTargets,
    action_targets_from_program,
)
from core.learning.recurrent_state_schema import (
    STATE_CARDINALITY,
    STATE_SLOT_LOSS_WEIGHTS,
    RecurrentStateTargets,
    state_targets_from_trace,
)
from core.learning.unified_intrinsic_recurrence import (
    UnifiedRecurrentController,
    unified_recurrent_hidden_states,
)

UNIFIED_INTRINSIC_OBJECTIVE_SCHEMA: Final = "aura.unified_intrinsic_objective.v1"


@dataclass(frozen=True, slots=True)
class UnifiedIntrinsicTrainingSpec:
    prelude_end: int
    coda_start: int
    train_depths: tuple[int, ...] = (1, 2, 4)
    heldout_depths: tuple[int, ...] = (8, 16)
    answer_weight: float = 1.0
    anchor_weight: float = 1.0
    trajectory_weight: float = 0.25
    progression_margin: float = 0.01
    halt_weight: float = 0.1
    state_weight: float = 1.0
    stutter_weight: float = 0.1
    anchor_injection: float = 0.0
    renormalize: bool = True
    schema: str = UNIFIED_INTRINSIC_OBJECTIVE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != UNIFIED_INTRINSIC_OBJECTIVE_SCHEMA:
            raise ValueError("unified objective schema differs")
        if self.prelude_end >= self.coda_start:
            raise ValueError("prelude_end must precede coda_start")
        if not self.train_depths or 1 not in self.train_depths:
            raise ValueError("train depths must include the T=1 anchor")
        if not self.heldout_depths:
            raise ValueError("heldout depths must not be empty")
        if any(type(depth) is not int or depth < 1 for depth in self.depths):
            raise ValueError("all recurrence depths must be positive integers")
        if set(self.train_depths) & set(self.heldout_depths):
            raise ValueError("train and heldout depths must be disjoint")
        if min(self.heldout_depths) <= max(self.train_depths):
            raise ValueError("heldout depths must extrapolate beyond training")
        for name in (
            "answer_weight",
            "anchor_weight",
            "trajectory_weight",
            "progression_margin",
            "halt_weight",
            "state_weight",
            "stutter_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 10.0
            ):
                raise ValueError(f"{name} must be finite and inside [0, 10]")

    @property
    def depths(self) -> tuple[int, ...]:
        return self.train_depths + self.heldout_depths

    def plan_at(self, depth: int) -> RecurrentDepthPlan:
        if depth not in self.depths:
            raise ValueError("requested depth is outside the frozen ladder")
        return RecurrentDepthPlan(
            prelude_end=self.prelude_end,
            coda_start=self.coda_start,
            iterations=depth,
            anchor_injection=self.anchor_injection,
            renormalize=self.renormalize,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def readout_fingerprint(model: Any, coda_start: int) -> str:
    """Hash the coda, final norm, and LM head that must remain frozen."""

    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if not layers or type(coda_start) is not int or not 0 <= coda_start < len(layers):
        raise ValueError("readout fingerprint coda boundary is invalid")
    inventory: list[tuple[str, Any]] = []
    for layer_index in range(coda_start, len(layers)):
        inventory.extend(
            (f"model.layers.{layer_index}.{name}", value)
            for name, value in tree_flatten(layers[layer_index].parameters())
        )
    inventory.extend(
        (f"model.norm.{name}", value) for name, value in tree_flatten(inner.norm.parameters())
    )
    head = getattr(model, "lm_head", None)
    if head is not None:
        inventory.extend(
            (f"lm_head.{name}", value) for name, value in tree_flatten(head.parameters())
        )
    else:
        inventory.extend(
            (f"tied_readout.{name}", value)
            for name, value in tree_flatten(inner.embed_tokens.parameters())
        )
    digest = hashlib.sha256()
    for name, value in sorted(inventory, key=lambda row: row[0]):
        mx.eval(value)
        digest.update(name.encode("utf-8"))
        digest.update(bytes(memoryview(value)))
    return digest.hexdigest()


def _answer_ce_from_hidden(
    model: Any,
    hidden: Any,
    answer_tokens: Any,
    answer_start: int,
    *,
    controller: UnifiedRecurrentController | None = None,
    role_logits: Any | None = None,
    place_logits: Any | None = None,
    state_probabilities: Any | None = None,
) -> Any:
    if getattr(model, "lm_head", None) is not None:
        logits = model.lm_head(hidden)
    else:
        logits = model.model.embed_tokens.as_linear(hidden)
    answer_count = int(answer_tokens.shape[-1])
    predicted = logits[:, answer_start : answer_start + answer_count, :]
    pointer_values = (controller, role_logits, place_logits, state_probabilities)
    if any(value is not None for value in pointer_values):
        if any(value is None for value in pointer_values):
            raise ValueError("answer digit pointer inputs are incomplete")
        predicted = controller.apply_answer_digit_pointer(
            predicted,
            role_logits[:, :answer_count, :],
            place_logits[:, :answer_count, :],
            state_probabilities,
        )
    return mx.mean(
        nn.losses.cross_entropy(
            predicted.astype(mx.float32),
            answer_tokens,
            reduction="none",
        )
    )


def unified_answer_and_recurrent_trajectory(
    model: Any,
    tokens: Any,
    answer_tokens: Any,
    plan: RecurrentDepthPlan,
    controller: UnifiedRecurrentController,
    *,
    memory_layout: MemoryLayout | None = None,
    decoder_input_tokens: Any | None = None,
    use_state_slots: bool = False,
    state_teacher_values: Sequence[Sequence[int]] | None = None,
    action_teacher_values: Sequence[Sequence[int]] | None = None,
    public_action_values: Sequence[Sequence[int]] | None = None,
    initial_state_teacher_values: Sequence[int] | None = None,
    state_teacher_forcing_probability: float = 0.0,
    microcode_lesion: bool = False,
    transition_processor_lesion: bool = False,
    transition_processor_mode: str = "residual",
    transition_opcode_expert_routing: str = "opcode",
    transition_history_lesion: bool = False,
    initial_state_logit_trajectory: list[Any] | None = None,
    action_logit_trajectory: list[Any] | None = None,
    answer_role_logit_trajectory: list[Any] | None = None,
    answer_place_logit_trajectory: list[Any] | None = None,
    answer_binding_feature_trajectory: list[tuple[Any, Any, Any]] | None = None,
    answer_digit_pointer_enabled: bool = True,
    final_answer_only: bool = False,
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """Decode every recurrent state through one frozen coda and readout.

    Labels and decoder inputs are deliberately separate.  The default is exact
    teacher forcing; an answer-aligned ``decoder_input_tokens`` tensor permits
    student roll-in without ever relabeling the generated mistakes as truth.
    """

    if int(answer_tokens.shape[-1]) < 1:
        raise ValueError("answer tokens must not be empty")
    if type(answer_digit_pointer_enabled) is not bool:
        raise TypeError("answer digit pointer flag must be boolean")
    decoder_inputs = answer_tokens if decoder_input_tokens is None else decoder_input_tokens
    if decoder_inputs.shape != answer_tokens.shape:
        raise ValueError("decoder inputs must be answer-aligned")
    full = mx.concatenate([tokens, decoder_inputs], axis=1)
    state_slot_start = int(tokens.shape[-1]) if use_state_slots else None
    state_slots = controller.config.state_slots if use_state_slots else 0
    effective_memory_layout = memory_layout
    if use_state_slots:
        if memory_layout is not None:
            raise ValueError("state slots own the protected-memory layout")
        # These are machine-state registers, not immutable episodic memory.
        # Their write authority is the typed categorical transition below; the
        # generic sparse memory gate would make a changing program counter and
        # register trace physically impossible to learn.
        effective_memory_layout = None
    state_logits: list[Any] = []
    state_probabilities: list[Any] = []
    role_logits = answer_role_logit_trajectory if answer_role_logit_trajectory is not None else []
    place_logits = (
        answer_place_logit_trajectory if answer_place_logit_trajectory is not None else []
    )
    decode_states: list[Any] = []
    _final, recurrent_states, _telemetry = unified_recurrent_hidden_states(
        model,
        full,
        plan,
        controller,
        memory_layout=effective_memory_layout,
        soft_memory_writes=True,
        state_slot_start=state_slot_start,
        state_logit_trajectory=state_logits if use_state_slots else None,
        action_logit_trajectory=action_logit_trajectory if use_state_slots else None,
        initial_state_logit_trajectory=(
            initial_state_logit_trajectory if use_state_slots else None
        ),
        decode_state_trajectory=decode_states if use_state_slots else None,
        state_probability_trajectory=(state_probabilities if use_state_slots else None),
        answer_role_logit_trajectory=(role_logits if use_state_slots else None),
        answer_place_logit_trajectory=(place_logits if use_state_slots else None),
        answer_binding_feature_trajectory=(
            answer_binding_feature_trajectory if use_state_slots else None
        ),
        state_teacher_values=state_teacher_values,
        action_teacher_values=action_teacher_values,
        public_action_values=public_action_values,
        initial_state_teacher_values=initial_state_teacher_values,
        state_teacher_forcing_probability=state_teacher_forcing_probability,
        microcode_lesion=microcode_lesion,
        transition_processor_lesion=transition_processor_lesion,
        transition_processor_mode=transition_processor_mode,
        transition_opcode_expert_routing=transition_opcode_expert_routing,
        transition_history_lesion=transition_history_lesion,
    )
    answer_start = int(tokens.shape[-1]) + state_slots - 1
    hidden_states: list[Any] = []
    losses: list[Any] = []
    output_states = decode_states if use_state_slots else recurrent_states
    if use_state_slots and not (
        len(output_states) == len(state_probabilities) == len(role_logits) == len(place_logits)
    ):
        raise RuntimeError("answer pointer trajectory differs from recurrent states")
    if not isinstance(final_answer_only, bool):
        raise TypeError("final_answer_only must be boolean")
    decode_indices = (
        range(len(output_states) - 1, len(output_states))
        if final_answer_only
        else range(len(output_states))
    )
    for index in decode_indices:
        state = output_states[index]
        hidden = _run(model.model.layers[plan.coda_start :], state)
        hidden = model.model.norm(hidden)
        hidden_states.append(hidden)
        losses.append(
            _answer_ce_from_hidden(
                model,
                hidden,
                answer_tokens,
                answer_start,
                controller=(
                    controller if use_state_slots and answer_digit_pointer_enabled else None
                ),
                role_logits=(
                    role_logits[index] if use_state_slots and answer_digit_pointer_enabled else None
                ),
                place_logits=(
                    place_logits[index]
                    if use_state_slots and answer_digit_pointer_enabled
                    else None
                ),
                state_probabilities=(
                    state_probabilities[index]
                    if use_state_slots and answer_digit_pointer_enabled
                    else None
                ),
            )
        )
    return recurrent_states, hidden_states, losses, state_logits


def unified_answer_trajectory(
    model: Any,
    tokens: Any,
    answer_tokens: Any,
    plan: RecurrentDepthPlan,
    controller: UnifiedRecurrentController,
    *,
    memory_layout: MemoryLayout | None = None,
    decoder_input_tokens: Any | None = None,
    use_state_slots: bool = False,
) -> tuple[list[Any], list[Any]]:
    """Decode every recurrent state through the same frozen coda and readout."""

    _recurrent, hidden, losses, _state_logits = unified_answer_and_recurrent_trajectory(
        model,
        tokens,
        answer_tokens,
        plan,
        controller,
        memory_layout=memory_layout,
        decoder_input_tokens=decoder_input_tokens,
        use_state_slots=use_state_slots,
    )
    return hidden, losses


def _progression_loss(losses: Sequence[Any], margin: float) -> Any:
    if len(losses) < 2:
        return mx.zeros(())
    return mx.mean(
        mx.stack(
            [
                mx.maximum(current - previous + margin, 0.0)
                for previous, current in zip(losses, losses[1:], strict=False)
            ]
        )
    )


def _halt_loss(
    controller: UnifiedRecurrentController,
    states: Sequence[Any],
    losses: Sequence[Any],
) -> Any:
    if len(states) < 2:
        return mx.zeros(())
    detached = [float(value.item()) for value in losses]
    best_index = min(range(len(detached)), key=detached.__getitem__)
    terms = []
    for index in range(1, len(states)):
        probability = controller.halt_probability(states[index - 1], states[index])
        target = mx.array(float(index >= best_index), dtype=mx.float32)
        terms.append(
            -(target * mx.log(probability + 1e-6))
            - (1.0 - target) * mx.log(1.0 - probability + 1e-6)
        )
    return mx.mean(mx.stack(terms))


def _supervised_halt_loss(
    controller: UnifiedRecurrentController,
    states: Sequence[Any],
    targets: RecurrentStateTargets,
) -> Any:
    """Train completion from the interpreter's done bit, not answer likelihood."""

    if len(states) != len(targets.values):
        raise ValueError("halt supervision differs from recurrent trajectory")
    if len(states) < 2:
        return mx.zeros(())
    terms = []
    for index in range(1, len(states)):
        probability = controller.halt_probability(states[index - 1], states[index])
        target = mx.array(float(targets.values[index][-1]), dtype=mx.float32)
        terms.append(
            -(target * mx.log(probability + 1e-6))
            - (1.0 - target) * mx.log(1.0 - probability + 1e-6)
        )
    return mx.mean(mx.stack(terms))


def structured_state_loss(
    controller: UnifiedRecurrentController,
    states: Sequence[Any],
    targets: RecurrentStateTargets,
    *,
    public_token_count: int,
    state_slot_start: int | None = None,
    state_logits: Sequence[Any] | None = None,
) -> tuple[Any, float, tuple[float, ...]]:
    """Measure exact categorical machine state from public prompt activations."""

    if len(states) != len(targets.values):
        raise ValueError("state supervision differs from recurrent trajectory")
    if state_logits is not None and len(state_logits) != len(states):
        raise ValueError("state decision trajectory differs from recurrent trajectory")
    losses = []
    accuracies: list[float] = []
    decisions = state_logits if state_logits is not None else (None,) * len(states)
    for state, decision, values, masks in zip(
        states,
        decisions,
        targets.values,
        targets.masks,
        strict=True,
    ):
        logits = (
            decision
            if decision is not None
            else controller.state_logits(
                state,
                public_token_count=(public_token_count if state_slot_start is None else None),
                state_slot_start=state_slot_start,
            )
        )
        if int(logits.shape[0]) != 1:
            raise ValueError("structured state supervision requires one task per batch")
        labels = mx.array(values, dtype=mx.int32)
        mask = mx.array(masks, dtype=mx.float32)
        weights = mx.array(STATE_SLOT_LOSS_WEIGHTS, dtype=mx.float32) * mask
        per_slot = nn.losses.cross_entropy(
            logits[0],
            labels,
            reduction="none",
        )
        losses.append(mx.sum(per_slot * weights) / mx.maximum(mx.sum(weights), 1.0))
        predictions = mx.argmax(logits[0], axis=-1)
        correct = (predictions == labels).astype(mx.float32)
        accuracies.append(float((mx.sum(correct * mask) / mx.sum(mask)).item()))
    return mx.mean(mx.stack(losses)), sum(accuracies) / len(accuracies), tuple(accuracies)


def structured_initial_state_loss(
    logits: Any,
    targets: RecurrentStateTargets,
) -> tuple[Any, float]:
    """Supervise the public-prefix state initializer with active slots only."""

    if logits.shape != (1, len(targets.initial_values), STATE_CARDINALITY):
        raise ValueError("initial state decision shape differs from the state schema")
    labels = mx.array(targets.initial_values, dtype=mx.int32)
    mask = mx.array(targets.initial_masks, dtype=mx.float32)
    weights = mx.array(STATE_SLOT_LOSS_WEIGHTS, dtype=mx.float32) * mask
    per_slot = nn.losses.cross_entropy(logits[0], labels, reduction="none")
    loss = mx.sum(per_slot * weights) / mx.maximum(mx.sum(weights), 1.0)
    predictions = mx.argmax(logits[0], axis=-1)
    correct = (predictions == labels).astype(mx.float32)
    accuracy = float((mx.sum(correct * mask) / mx.sum(mask)).item())
    return loss, accuracy


def structured_state_accuracy_breakdown(
    logits: Sequence[Any],
    targets: RecurrentStateTargets,
) -> dict[str, float | None]:
    """Separate computational value accuracy from control bookkeeping."""

    if len(logits) != len(targets.values):
        raise ValueError("state breakdown differs from recurrent trajectory")
    value_correct = 0.0
    value_total = 0.0
    control_correct = 0.0
    control_total = 0.0
    value_exact = 0.0
    value_exact_count = 0
    for decision, labels_row, masks_row in zip(
        logits,
        targets.values,
        targets.masks,
        strict=True,
    ):
        labels = mx.array(labels_row, dtype=mx.int32)
        masks = mx.array(masks_row, dtype=mx.float32)
        predictions = mx.argmax(decision[0], axis=-1)
        correct = (predictions == labels).astype(mx.float32)
        value_mask = masks * mx.array((0.0, 1.0, 1.0, 1.0, 0.0))
        control_mask = masks * mx.array((1.0, 0.0, 0.0, 0.0, 1.0))
        value_correct += float(mx.sum(correct * value_mask).item())
        value_total += float(mx.sum(value_mask).item())
        control_correct += float(mx.sum(correct * control_mask).item())
        control_total += float(mx.sum(control_mask).item())
        if float(mx.sum(value_mask).item()) > 0.0:
            value_exact += float((mx.sum(correct * value_mask) == mx.sum(value_mask)).item())
            value_exact_count += 1
    return {
        "value_accuracy": value_correct / value_total if value_total else None,
        "value_exact_accuracy": (value_exact / value_exact_count if value_exact_count else None),
        "control_accuracy": control_correct / control_total if control_total else None,
    }


def structured_state_trajectory_diagnostics(
    logits: Sequence[Any],
    targets: RecurrentStateTargets,
    *,
    active_steps: int,
) -> dict[str, Any]:
    """Measure active execution separately from terminal padding.

    ``state_targets_from_trace`` intentionally repeats the terminal state when
    an evaluation depth exceeds the program depth.  Those repetitions are a
    useful stability probe, but averaging them into execution exactness makes
    T16 a mixture of algorithm quality and padding length.  This diagnostic
    keeps the two claims separate and reports where an autonomous rollout first
    leaves, and whether it later re-enters, the verified public trajectory.
    """

    if len(logits) != len(targets.values):
        raise ValueError("state trajectory diagnostics differ from recurrent trajectory")
    if type(active_steps) is not int or not 1 <= active_steps <= len(logits):
        raise ValueError("active transition count differs from recurrent trajectory")

    exact_steps: list[bool] = []
    value_exact_steps: list[bool] = []
    register_correct = [0] * len(STATE_SLOT_LOSS_WEIGHTS)
    register_total = [0] * len(STATE_SLOT_LOSS_WEIGHTS)
    predictions: list[tuple[int, ...]] = []
    for step_index, (decision, labels_row, masks_row) in enumerate(
        zip(logits, targets.values, targets.masks, strict=True)
    ):
        predicted = tuple(int(value) for value in mx.argmax(decision[0], axis=-1).tolist())
        predictions.append(predicted)
        active = [index for index, mask in enumerate(masks_row) if mask]
        values = [index for index in active if 0 < index < len(masks_row) - 1]
        exact_steps.append(all(predicted[index] == labels_row[index] for index in active))
        value_exact_steps.append(
            bool(values)
            and all(predicted[index] == labels_row[index] for index in values)
        )
        if step_index < active_steps:
            for index in active:
                register_total[index] += 1
                register_correct[index] += int(predicted[index] == labels_row[index])

    active_exact = exact_steps[:active_steps]
    active_value_exact = value_exact_steps[:active_steps]
    first_error = next(
        (index + 1 for index, exact in enumerate(active_exact) if not exact),
        None,
    )
    recovery_observable = first_error is not None and first_error < active_steps
    recovered = (
        any(active_exact[first_error:])
        if recovery_observable and first_error is not None
        else None
    )
    sustained_recovery = (
        any(all(active_exact[index:]) for index in range(first_error, active_steps))
        if recovery_observable and first_error is not None
        else None
    )
    prior_correct = active_exact[:-1]
    subsequent_correct = active_exact[1:]
    correct_after_correct = sum(
        current
        for previous, current in zip(prior_correct, subsequent_correct, strict=True)
        if previous
    )
    correct_predecessors = sum(prior_correct)
    correct_after_wrong = sum(
        current
        for previous, current in zip(prior_correct, subsequent_correct, strict=True)
        if not previous
    )
    wrong_predecessors = len(prior_correct) - correct_predecessors
    padding_predictions = predictions[active_steps:]
    terminal_target = targets.values[active_steps - 1]
    terminal_mask = targets.masks[active_steps - 1]
    terminal_stable = (
        all(
            all(
                prediction[index] == terminal_target[index]
                for index, active in enumerate(terminal_mask)
                if active
            )
            for prediction in padding_predictions
        )
        if padding_predictions
        else None
    )
    terminal_self_stable = (
        all(prediction == padding_predictions[0] for prediction in padding_predictions[1:])
        if padding_predictions
        else None
    )
    return {
        "active_steps": active_steps,
        "padding_steps": len(logits) - active_steps,
        "active_state_exact_accuracy": sum(active_exact) / active_steps,
        "active_value_exact_accuracy": sum(active_value_exact) / active_steps,
        "active_trajectory_exact": all(active_exact),
        "first_error_step": first_error,
        "first_error_fraction": (
            1.0 if first_error is None else (first_error - 1) / active_steps
        ),
        "recovery_observable": recovery_observable,
        "recovered_after_first_error": recovered,
        "sustained_recovery_after_first_error": sustained_recovery,
        "conditional_transition_counts": {
            "correct_after_correct": correct_after_correct,
            "correct_predecessors": correct_predecessors,
            "correct_after_wrong": correct_after_wrong,
            "wrong_predecessors": wrong_predecessors,
        },
        "p_correct_given_previous_correct": (
            correct_after_correct / correct_predecessors
            if correct_predecessors
            else None
        ),
        "p_correct_given_previous_wrong": (
            correct_after_wrong / wrong_predecessors if wrong_predecessors else None
        ),
        "terminal_stability_observable": terminal_stable is not None,
        "terminal_correct_stable": terminal_stable,
        "terminal_self_stable": terminal_self_stable,
        "per_register_accuracy": {
            name: (
                register_correct[index] / register_total[index]
                if register_total[index]
                else None
            )
            for index, name in enumerate(("pc", "value0", "value1", "value2", "done"))
        },
    }


def structured_initial_state_accuracy_breakdown(
    logits: Any,
    targets: RecurrentStateTargets,
) -> dict[str, float | None]:
    """Apply the same value/control split to the public-prefix initializer."""

    return structured_state_accuracy_breakdown(
        (logits,),
        RecurrentStateTargets(
            family=targets.family,
            field_names=targets.field_names,
            initial_values=targets.initial_values,
            initial_masks=targets.initial_masks,
            values=(targets.initial_values,),
            masks=(targets.initial_masks,),
            trace_sha256=targets.trace_sha256,
        ),
    )


def structured_action_loss(
    logits: Sequence[Any],
    targets: RecurrentActionTargets,
) -> tuple[Any, float, tuple[float, ...]]:
    """Train all action slots while reporting accuracy only on active fields."""

    if len(logits) != len(targets.values):
        raise ValueError("action supervision differs from recurrent trajectory")
    losses = []
    accuracies: list[float] = []
    active_correct_total = 0.0
    active_field_total = 0.0
    for decision, values, masks in zip(
        logits,
        targets.values,
        targets.masks,
        strict=True,
    ):
        if int(decision.shape[0]) != 1:
            raise ValueError("structured action supervision requires one task per batch")
        labels = mx.array(values, dtype=mx.int32)
        per_slot = nn.losses.cross_entropy(decision[0], labels, reduction="none")
        mask = mx.array(masks, dtype=mx.float32)
        active_count = mx.sum(mask)
        inactive = 1.0 - mask
        inactive_count = mx.sum(inactive)
        active_mean = mx.sum(per_slot * mask) / mx.maximum(active_count, 1.0)
        # One weak field invalidates an executable instruction even when every
        # other field is already certain. Mean CE alone can keep spending
        # gradient on those easy fields. This smooth worst-field term equals
        # the mean when all active fields are equally difficult, then focuses
        # additional pressure on the field currently limiting exactness.
        active_terms = mx.where(
            mask > 0.0,
            per_slot,
            mx.full_like(per_slot, -1e9),
        )
        weakest_link = mx.logsumexp(active_terms) - mx.log(mx.maximum(active_count, 1.0))
        active_loss = 0.5 * (active_mean + weakest_link)
        null_loss = mx.sum(per_slot * inactive) / mx.maximum(inactive_count, 1.0)
        # Active operation fields carry the computation. Null slots remain
        # trained for post-completion stability but cannot dominate the parser.
        losses.append(mx.where(active_count > 0.0, active_loss + 0.1 * null_loss, null_loss))
        predictions = mx.argmax(decision[0], axis=-1)
        correct = (predictions == labels).astype(mx.float32)
        active = float(mx.sum(mask).item())
        active_correct = float(mx.sum(correct * mask).item())
        if active > 0.0:
            active_correct_total += active_correct
            active_field_total += active
        accuracies.append(
            active_correct / active if active > 0.0 else float(mx.all(predictions == labels).item())
        )
    active_accuracy = (
        active_correct_total / active_field_total
        if active_field_total > 0.0
        else sum(accuracies) / len(accuracies)
    )
    return mx.mean(mx.stack(losses)), active_accuracy, tuple(accuracies)


def structured_action_accuracy_breakdown(
    logits: Sequence[Any],
    targets: RecurrentActionTargets,
) -> dict[str, Any]:
    """Report executable whole-instruction accuracy and per-slot evidence."""

    if len(logits) != len(targets.values):
        raise ValueError("action breakdown differs from recurrent trajectory")
    slot_correct = [0.0] * len(targets.values[0])
    slot_total = [0.0] * len(targets.values[0])
    exact = 0.0
    exact_count = 0
    for decision, values, masks in zip(
        logits,
        targets.values,
        targets.masks,
        strict=True,
    ):
        predictions = mx.argmax(decision[0], axis=-1)
        labels = mx.array(values, dtype=mx.int32)
        correct = (predictions == labels).astype(mx.float32)
        active_indices = [index for index, active in enumerate(masks) if active]
        if active_indices:
            exact += float(mx.all(correct[mx.array(active_indices)] > 0.0).item())
            exact_count += 1
        for index in active_indices:
            slot_correct[index] += float(correct[index].item())
            slot_total[index] += 1.0
    return {
        "instruction_exact_accuracy": exact / exact_count if exact_count else None,
        "slot_accuracy": {
            ACTION_SLOT_NAMES[index]: (
                slot_correct[index] / slot_total[index] if slot_total[index] else None
            )
            for index in range(len(ACTION_SLOT_NAMES))
        },
    }


def _stutter_loss(states: Sequence[Any], targets: RecurrentStateTargets) -> Any:
    """Make post-completion recurrence preserve the terminal hidden state."""

    terms = []
    for index in range(1, len(states)):
        if not targets.values[index - 1][-1]:
            continue
        previous = mx.stop_gradient(states[index - 1].astype(mx.float32))
        current = states[index].astype(mx.float32)
        scale = mx.maximum(mx.mean(previous**2), 1e-6)
        terms.append(mx.mean((current - previous) ** 2) / scale)
    return mx.mean(mx.stack(terms)) if terms else mx.zeros(())


def unified_intrinsic_training_loss(
    model: Any,
    tokens: Any,
    answer_tokens: Any,
    controller: UnifiedRecurrentController,
    spec: UnifiedIntrinsicTrainingSpec,
    *,
    memory_layout: MemoryLayout | None = None,
    readout_sha256: str | None = None,
    decoder_input_tokens: Any | None = None,
    transition_trace: Any | None = None,
    transition_program: Any | None = None,
    state_teacher_forcing_probability: float = 0.0,
    answer_digit_pointer_enabled: bool = True,
    objective_depth: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Train semantics at shallow depths while keeping readout immutable.

    ``objective_depth`` exposes one algebraically additive depth contribution so
    resident training can materialize and release each frozen-coda gradient
    before constructing the next. Its denominators remain those of the full
    objective, so summing every selected contribution is the same loss.
    """

    if readout_sha256 is None:
        readout_sha256 = readout_fingerprint(model, spec.coda_start)
    elif re.fullmatch(r"[0-9a-f]{64}", readout_sha256) is None:
        raise ValueError("readout commitment is invalid")
    if objective_depth is not None and objective_depth not in spec.train_depths:
        raise ValueError("objective depth is outside the trained recurrence ladder")
    selected_depths = spec.train_depths if objective_depth is None else (objective_depth,)
    final_losses: list[Any] = []
    progression_terms: list[Any] = []
    halt_terms: list[Any] = []
    state_terms: list[Any] = []
    stutter_terms: list[Any] = []
    state_commitments: dict[str, dict[str, Any]] = {}
    per_depth: dict[str, dict[str, Any]] = {}
    for depth in selected_depths:
        depth_targets = (
            state_targets_from_trace(transition_trace, depth)
            if transition_trace is not None
            else None
        )
        action_targets = (
            action_targets_from_program(transition_program, depth)
            if transition_program is not None
            else None
        )
        if action_targets is not None and depth_targets is None:
            raise ValueError("action supervision requires state supervision")
        initial_state_logits: list[Any] = []
        action_logits: list[Any] = []
        recurrent_states, states, losses, state_logits = unified_answer_and_recurrent_trajectory(
            model,
            tokens,
            answer_tokens,
            spec.plan_at(depth),
            controller,
            memory_layout=memory_layout,
            decoder_input_tokens=decoder_input_tokens,
            use_state_slots=transition_trace is not None,
            state_teacher_values=(depth_targets.values if depth_targets is not None else None),
            action_teacher_values=(action_targets.values if action_targets is not None else None),
            initial_state_teacher_values=(
                depth_targets.initial_values if depth_targets is not None else None
            ),
            state_teacher_forcing_probability=(
                state_teacher_forcing_probability if depth_targets is not None else 0.0
            ),
            answer_digit_pointer_enabled=answer_digit_pointer_enabled,
            initial_state_logit_trajectory=initial_state_logits,
            action_logit_trajectory=action_logits,
        )
        final_losses.append(losses[-1])
        progression = _progression_loss(losses, spec.progression_margin)
        targets = (
            depth_targets if depth_targets is not None and depth == max(spec.train_depths) else None
        )
        if targets is None:
            state_loss = mx.zeros(())
            state_accuracy = None
            state_step_accuracy: tuple[float, ...] = ()
            initial_state_loss = mx.zeros(())
            initial_state_accuracy = None
            action_loss = mx.zeros(())
            action_accuracy = None
            action_step_accuracy: tuple[float, ...] = ()
            stuttering = mx.zeros(())
            halting = (
                mx.zeros(())
                if transition_trace is not None
                else _halt_loss(controller, states, losses)
            )
        else:
            state_loss, state_accuracy, state_step_accuracy = structured_state_loss(
                controller,
                recurrent_states,
                targets,
                public_token_count=int(tokens.shape[-1]),
                state_slot_start=int(tokens.shape[-1]),
                state_logits=state_logits,
            )
            if len(initial_state_logits) != 1:
                raise RuntimeError("typed recurrence emitted no initial state decision")
            initial_state_loss, initial_state_accuracy = structured_initial_state_loss(
                initial_state_logits[0],
                targets,
            )
            if action_targets is None:
                action_loss = mx.zeros(())
                action_accuracy = None
                action_step_accuracy = ()
                state_loss = 0.5 * (state_loss + initial_state_loss)
            else:
                action_loss, action_accuracy, action_step_accuracy = structured_action_loss(
                    action_logits, action_targets
                )
                state_loss = (state_loss + initial_state_loss + action_loss) / 3.0
            stuttering = _stutter_loss(recurrent_states, targets)
            halting = _supervised_halt_loss(controller, recurrent_states, targets)
            state_commitments[f"T{depth}"] = targets.commitment()
            if action_targets is not None:
                state_commitments[f"T{depth}"]["action"] = action_targets.commitment()
        progression_terms.append(progression)
        if transition_trace is None or targets is not None:
            halt_terms.append(halting)
        if targets is not None:
            state_terms.append(state_loss)
            stutter_terms.append(stuttering)
        per_depth[f"T{depth}"] = {
            "step_ce": [float(value.item()) for value in losses],
            "final_ce": float(losses[-1].item()),
            "progression_loss": float(progression.item()),
            "halt_loss": float(halting.item()),
            "state_loss": float(state_loss.item()),
            "state_accuracy": state_accuracy,
            "initial_state_loss": float(initial_state_loss.item()),
            "initial_state_accuracy": initial_state_accuracy,
            "action_loss": float(action_loss.item()),
            "action_accuracy": action_accuracy,
            "action_step_accuracy": list(action_step_accuracy),
            "state_step_accuracy": list(state_step_accuracy),
            "stutter_loss": float(stuttering.item()),
        }
    depth_count = len(spec.train_depths)
    anchor = final_losses[selected_depths.index(1)] if 1 in selected_depths else mx.zeros(())
    # A one-depth invocation is a summand, not a reweighted smaller objective.
    final_mean = mx.sum(mx.stack(final_losses)) / depth_count
    progression_mean = mx.sum(mx.stack(progression_terms)) / depth_count
    halt_denominator = depth_count if transition_trace is None else 1
    halt_mean = mx.sum(mx.stack(halt_terms)) / halt_denominator if halt_terms else mx.zeros(())
    state_mean = mx.sum(mx.stack(state_terms)) if state_terms else mx.zeros(())
    stutter_mean = mx.sum(mx.stack(stutter_terms)) if stutter_terms else mx.zeros(())
    total = (
        spec.answer_weight * final_mean
        + spec.anchor_weight * anchor
        + spec.trajectory_weight * progression_mean
        + spec.halt_weight * halt_mean
        + spec.state_weight * state_mean
        + spec.stutter_weight * stutter_mean
    )
    return total, {
        "schema": UNIFIED_INTRINSIC_OBJECTIVE_SCHEMA,
        "spec": spec.to_dict(),
        "per_depth": per_depth,
        "anchor_ce": float(anchor.item()),
        "final_mean_ce": float(final_mean.item()),
        "progression_loss": float(progression_mean.item()),
        "halt_loss": float(halt_mean.item()),
        "state_loss": float(state_mean.item()),
        "stutter_loss": float(stutter_mean.item()),
        "state_supervision": {
            "available": transition_trace is not None,
            "evaluator_only": True,
            "serialized_into_model_input": False,
            "commitments": state_commitments,
            "teacher_forcing_probability": state_teacher_forcing_probability,
            "teacher_available_at_inference": False,
        },
        "total": float(total.item()),
        "readout_sha256": readout_sha256,
        "readout_frozen_by_training_contract": True,
        "decoder_history": (
            "teacher_forced" if decoder_input_tokens is None else "student_rollin_answer_aligned"
        ),
        "labels_from_generated_tokens": False,
        "heldout_depths_unopened": list(spec.heldout_depths),
        "objective_depth": objective_depth,
    }


def unified_process_training_loss(
    model: Any,
    tokens: Any,
    controller: UnifiedRecurrentController,
    plan: RecurrentDepthPlan,
    *,
    transition_trace: Any,
    transition_program: Any,
    state_teacher_forcing_probability: float,
    state_weight: float = 1.0,
    component: str = "joint",
    public_action_values: Sequence[Sequence[int]] | None = None,
    microcode_lesion: bool = False,
    transition_processor_mode: str = "residual",
    transition_opcode_expert_routing: str = "opcode",
) -> tuple[Any, dict[str, Any]]:
    """Train the autonomous typed process without constructing answer graphs.

    State acquisition needs the public-prefix transformer parser and the typed
    transition controller. It does not need teacher-forced answer tokens, one
    coda decode per recurrent step, or zero-weight shallower-depth objectives.
    Keeping those graphs made the 1.5B process lane exceed the host's 48 GB
    intervention ceiling despite gradient checkpointing.
    """

    if transition_trace is None or transition_program is None:
        raise ValueError("process training requires exact state and action supervision")
    if component not in {
        "initializer",
        "action",
        "action_workspace",
        "transition",
        "joint",
    }:
        raise ValueError("process training component is invalid")
    if (
        isinstance(state_weight, bool)
        or not isinstance(state_weight, (int, float))
        or not 0.0 < float(state_weight) <= 10.0
    ):
        raise ValueError("process state weight must be inside (0, 10]")
    targets = state_targets_from_trace(transition_trace, plan.iterations)
    action_targets = action_targets_from_program(transition_program, plan.iterations)
    if public_action_values is not None and component in {"action", "action_workspace"}:
        raise ValueError("public actions bypass the learned action component")
    initial_state_logits: list[Any] = []
    action_logits: list[Any] = []
    state_logits: list[Any] = []
    recurrent_states: list[Any]
    _final, recurrent_states, _telemetry = unified_recurrent_hidden_states(
        model,
        tokens,
        plan,
        controller,
        soft_memory_writes=True,
        state_slot_start=int(tokens.shape[-1]),
        state_logit_trajectory=state_logits,
        action_logit_trajectory=action_logits,
        initial_state_logit_trajectory=initial_state_logits,
        state_teacher_values=targets.values,
        action_teacher_values=(
            None if public_action_values is not None else action_targets.values
        ),
        public_action_values=public_action_values,
        initial_state_teacher_values=targets.initial_values,
        state_teacher_forcing_probability=state_teacher_forcing_probability,
        microcode_lesion=microcode_lesion,
        transition_processor_mode=transition_processor_mode,
        transition_opcode_expert_routing=transition_opcode_expert_routing,
        process_only=True,
        detach_problem_evidence=False,
    )
    if len(initial_state_logits) != 1:
        raise RuntimeError("typed process emitted no initial-state decision")
    state_loss, state_accuracy, state_step_accuracy = structured_state_loss(
        controller,
        recurrent_states,
        targets,
        public_token_count=int(tokens.shape[-1]),
        state_slot_start=int(tokens.shape[-1]),
        state_logits=state_logits,
    )
    initial_loss, initial_accuracy = structured_initial_state_loss(
        initial_state_logits[0],
        targets,
    )
    if public_action_values is None:
        action_loss, action_accuracy, action_step_accuracy = structured_action_loss(
            action_logits,
            action_targets,
        )
    else:
        action_loss = mx.zeros(())
        action_accuracy = None
        action_step_accuracy = ()
    component_losses = {
        "initializer": initial_loss,
        "action": action_loss,
        "action_workspace": action_loss,
        "transition": state_loss,
        "joint": (
            (state_loss + initial_loss) / 2.0
            if public_action_values is not None
            else (state_loss + initial_loss + action_loss) / 3.0
        ),
    }
    process_loss = component_losses[component]
    return float(state_weight) * process_loss, {
        "schema": UNIFIED_INTRINSIC_OBJECTIVE_SCHEMA,
        "objective": "prompt_only_typed_process",
        "component": component,
        "depth": plan.iterations,
        "state_accuracy": state_accuracy,
        "initial_state_accuracy": initial_accuracy,
        "action_accuracy": action_accuracy,
        "component_losses": {
            "initializer": float(initial_loss.item()),
            "action": float(action_loss.item()),
            "action_workspace": float(action_loss.item()),
            "transition": float(state_loss.item()),
            "joint": float(component_losses["joint"].item()),
        },
        "state_step_accuracy": list(state_step_accuracy),
        "action_step_accuracy": list(action_step_accuracy),
        "teacher_forcing_probability": state_teacher_forcing_probability,
        "public_action_program": public_action_values is not None,
        "public_actions_are_correctness_authority": False,
        "exact_microcode_available": not microcode_lesion,
        "answer_tokens_exposed": False,
        "answer_or_coda_graph_constructed": False,
        "problem_evidence_gradient": "scoped_transformer_enabled",
        "total": float((float(state_weight) * process_loss).item()),
    }


def unified_typed_transition_processor_loss(
    controller: UnifiedRecurrentController,
    plan: RecurrentDepthPlan,
    *,
    transition_trace: Any,
    transition_program: Any,
    public_action_values: Sequence[Sequence[int]],
    opcode_expert_routing: str = "opcode",
) -> tuple[Any, dict[str, Any]]:
    """Train exact categorical transition algebra without a transformer graph.

    The verified state trace remains training-only supervision. Actions come
    from the independently compiled public prompt program, matching the
    teacher-free runtime surface. The objective reaches only typed transition
    memory and processor tensors; it cannot turn the frozen transformer or
    readout into a parallel answer producer.
    """

    if transition_trace is None or transition_program is None:
        raise ValueError("direct transition training requires verified process evidence")
    targets = state_targets_from_trace(transition_trace, plan.iterations)
    action_targets = action_targets_from_program(transition_program, plan.iterations)
    public_actions = tuple(tuple(int(value) for value in row) for row in public_action_values)
    if public_actions != action_targets.values:
        raise ValueError("public transition actions differ from the verified program")

    action_history: list[Any] = []
    losses: list[Any] = []
    correct = mx.zeros((), dtype=mx.float32)
    required = mx.zeros((), dtype=mx.float32)
    state_probabilities = controller.exact_probabilities(
        targets.initial_values,
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    active_transitions = min(
        plan.iterations,
        int(transition_trace.depth),
        len(transition_program.actions),
    )
    for action_values, next_values, masks in zip(
        public_actions[:active_transitions],
        targets.values[:active_transitions],
        targets.masks[:active_transitions],
        strict=True,
    ):
        action_probabilities = controller.exact_probabilities(
            action_values,
            slots=controller.config.action_slots,
            cardinality=controller.config.action_cardinality,
        )
        action_history.append(action_probabilities)
        history_memory = controller._typed_transition_memory(action_history)
        logits = controller.resolve_transition_processor_logits(
            None,
            state_probabilities,
            action_probabilities,
            history_memory,
            transition_processor_mode="authoritative",
            opcode_expert_routing=opcode_expert_routing,
        )
        labels = mx.array((next_values,), dtype=mx.int32)
        mask = mx.array((masks,), dtype=mx.float32)
        token_losses = nn.losses.cross_entropy(
            logits.astype(mx.float32),
            labels,
            reduction="none",
        )
        denominator = mx.maximum(mx.sum(mask), 1.0)
        losses.append(mx.sum(token_losses * mask) / denominator)
        predictions = mx.argmax(logits, axis=-1)
        correct = correct + mx.sum((predictions == labels).astype(mx.float32) * mask)
        required = required + mx.sum(mask)
        # Match deployment exactly: after the public initial state, recurrent
        # execution consumes its own hard categorical decision.  The
        # straight-through estimator keeps the entire rollout differentiable
        # while preventing the private trace from becoming an inference input.
        state_probabilities = controller.straight_through_probabilities(logits)
    if not losses:
        raise ValueError("direct transition training has no active transitions")
    loss = mx.mean(mx.stack(losses))
    accuracy = correct / mx.maximum(required, 1.0)
    return loss, {
        "schema": UNIFIED_INTRINSIC_OBJECTIVE_SCHEMA,
        "objective": "verified_typed_transition_processor",
        "depth": plan.iterations,
        "transitions": len(losses),
        "state_accuracy": float(accuracy.item()),
        "public_action_program": True,
        "public_actions_are_correctness_authority": False,
        "verified_state_teacher_available": True,
        "teacher_available_at_inference": False,
        "initial_state_authority": "verified_public_initial_state",
        "rollout_state_authority": "student_prediction_after_initial",
        "closed_loop_student_rollout": True,
        "deployed_transition_policy": "processor_authoritative",
        "legacy_transition_logits_available": False,
        "opcode_expert_routing": opcode_expert_routing,
        "active_transitions": active_transitions,
        "post_terminal_transitions_trained": 0,
        "answer_tokens_exposed": False,
        "transformer_graph_constructed": False,
        "readout_graph_constructed": False,
        "total": float(loss.item()),
    }


__all__ = [
    "UNIFIED_INTRINSIC_OBJECTIVE_SCHEMA",
    "UnifiedIntrinsicTrainingSpec",
    "readout_fingerprint",
    "structured_state_loss",
    "structured_state_accuracy_breakdown",
    "structured_state_trajectory_diagnostics",
    "structured_initial_state_accuracy_breakdown",
    "structured_initial_state_loss",
    "structured_action_loss",
    "structured_action_accuracy_breakdown",
    "unified_answer_and_recurrent_trajectory",
    "unified_answer_trajectory",
    "unified_intrinsic_training_loss",
    "unified_process_training_loss",
    "unified_typed_transition_processor_loss",
]
