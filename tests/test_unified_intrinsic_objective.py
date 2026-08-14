"""The unified recurrence learns semantics without moving its readout."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
optim = pytest.importorskip("mlx.optimizers")
pytest.importorskip("mlx_lm")

from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.learning.recurrence_curriculum import (  # noqa: E402
    StructuredTransitionProgram,
    StructuredTransitionTrace,
)
from core.learning.recurrent_action_schema import (  # noqa: E402
    action_targets_from_program,
)
from core.learning.unified_intrinsic_objective import (  # noqa: E402
    UnifiedIntrinsicTrainingSpec,
    readout_fingerprint,
    structured_action_accuracy_breakdown,
    structured_action_loss,
    structured_initial_state_accuracy_breakdown,
    structured_initial_state_loss,
    unified_answer_and_recurrent_trajectory,
    unified_answer_trajectory,
    unified_intrinsic_training_loss,
)
from core.learning.unified_intrinsic_recurrence import (  # noqa: E402
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
)


def _model() -> Model:
    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=32,
            num_hidden_layers=6,
            intermediate_size=64,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=64,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rope_theta=10_000.0,
        )
    )
    model.freeze()
    mx.eval(model.parameters())
    return model


def _spec() -> UnifiedIntrinsicTrainingSpec:
    return UnifiedIntrinsicTrainingSpec(
        prelude_end=2,
        coda_start=4,
        train_depths=(1, 2, 3),
        heldout_depths=(5, 8),
    )


def _controller(
    *,
    literal_digit_token_ids: tuple[int, ...] = (),
) -> UnifiedRecurrentController:
    return UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=32,
            correction_rank=4,
            minimum_iterations=1,
            literal_digit_token_ids=literal_digit_token_ids,
        )
    )


TOKENS = mx.array([[2, 7, 11, 13]])
ANSWERS = mx.array([[17, 19]])


def test_depth_split_is_strict_and_extrapolating() -> None:
    spec = _spec()
    assert not (set(spec.train_depths) & set(spec.heldout_depths))
    assert min(spec.heldout_depths) > max(spec.train_depths)
    with pytest.raises(ValueError, match="disjoint"):
        UnifiedIntrinsicTrainingSpec(2, 4, (1, 2), (2, 4))


def test_state_only_objective_removes_answer_and_anchor_gradients() -> None:
    spec = UnifiedIntrinsicTrainingSpec(
        2,
        4,
        (1, 2, 3),
        (5, 8),
        answer_weight=0.0,
        anchor_weight=0.0,
        trajectory_weight=0.0,
        halt_weight=0.0,
        state_weight=1.0,
        stutter_weight=0.0,
    )
    assert spec.answer_weight == 0.0
    assert spec.state_weight == 1.0


def test_every_recurrent_state_is_decoded_through_the_same_readout() -> None:
    states, losses = unified_answer_trajectory(
        _model(),
        TOKENS,
        ANSWERS,
        _spec().plan_at(3),
        _controller(),
    )
    assert len(states) == len(losses) == 3
    assert all(value.shape == () for value in losses)
    assert all(float(value.item()) > 0.0 for value in losses)


def test_final_only_evaluation_preserves_terminal_answer_loss() -> None:
    model = _model()
    controller = _controller()
    plan = _spec().plan_at(5)
    recurrent, all_hidden, all_losses, _state_logits = (
        unified_answer_and_recurrent_trajectory(
            model,
            TOKENS,
            ANSWERS,
            plan,
            controller,
        )
    )
    final_recurrent, final_hidden, final_losses, _final_state_logits = (
        unified_answer_and_recurrent_trajectory(
            model,
            TOKENS,
            ANSWERS,
            plan,
            controller,
            final_answer_only=True,
        )
    )
    mx.eval(recurrent, all_hidden, all_losses, final_recurrent, final_hidden, final_losses)
    assert len(recurrent) == len(final_recurrent) == 5
    assert len(all_hidden) == len(all_losses) == 5
    assert len(final_hidden) == len(final_losses) == 1
    assert bool(mx.array_equal(recurrent[-1], final_recurrent[-1]))
    assert float(final_losses[-1].item()) == pytest.approx(
        float(all_losses[-1].item()),
        rel=1e-6,
        abs=1e-6,
    )


def test_typed_semantic_objective_can_disable_legacy_digit_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("legacy digit pointer must be absent")

    monkeypatch.setattr(
        UnifiedRecurrentController,
        "apply_answer_digit_pointer",
        fail_if_called,
    )
    _recurrent, hidden, losses, _states = unified_answer_and_recurrent_trajectory(
        _model(),
        TOKENS,
        ANSWERS,
        _spec().plan_at(3),
        _controller(literal_digit_token_ids=tuple(range(10))),
        use_state_slots=True,
        answer_digit_pointer_enabled=False,
        final_answer_only=True,
    )

    assert len(hidden) == len(losses) == 1
    assert float(losses[0].item()) > 0.0


def test_student_rollin_changes_history_without_changing_labels() -> None:
    model = _model()
    controller = _controller()
    teacher_states, teacher_losses = unified_answer_trajectory(
        model,
        TOKENS,
        ANSWERS,
        _spec().plan_at(3),
        controller,
    )
    rollin = mx.array([[23, 29]])
    student_states, student_losses = unified_answer_trajectory(
        model,
        TOKENS,
        ANSWERS,
        _spec().plan_at(3),
        controller,
        decoder_input_tokens=rollin,
    )
    mx.eval(teacher_states, teacher_losses, student_states, student_losses)
    assert not bool(mx.array_equal(teacher_states[-1], student_states[-1]))
    assert float(teacher_losses[-1].item()) != pytest.approx(
        float(student_losses[-1].item())
    )
    _loss, receipt = unified_intrinsic_training_loss(
        model,
        TOKENS,
        ANSWERS,
        controller,
        _spec(),
        decoder_input_tokens=rollin,
    )
    assert receipt["decoder_history"] == "student_rollin_answer_aligned"
    assert receipt["labels_from_generated_tokens"] is False
    with pytest.raises(ValueError, match="answer-aligned"):
        unified_answer_trajectory(
            model,
            TOKENS,
            ANSWERS,
            _spec().plan_at(3),
            controller,
            decoder_input_tokens=mx.array([[23]]),
        )


def test_optimizer_moves_controller_but_not_frozen_readout() -> None:
    model = _model()
    controller = _controller()
    spec = _spec()
    before_readout = readout_fingerprint(model, spec.coda_start)
    before_controller = controller.parameter_sha256()

    def objective(candidate: UnifiedRecurrentController):
        loss, _telemetry = unified_intrinsic_training_loss(
            model,
            TOKENS,
            ANSWERS,
            candidate,
            spec,
            readout_sha256=before_readout,
        )
        return loss

    loss, gradients = nn.value_and_grad(controller, objective)(controller)
    flat = dict(tree_flatten(gradients))
    mx.eval(loss, gradients)
    assert float(mx.max(mx.abs(flat["correction_b"]))) > 0.0
    assert float(mx.max(mx.abs(flat["halt_state_weight"]))) > 0.0
    assert float(mx.abs(flat["transport_bias"])) > 0.0
    assert float(mx.abs(flat["transport_decay_logit"])) > 0.0
    assert float(mx.max(mx.abs(flat["transport_state_weight"]))) > 0.0
    assert float(mx.max(mx.abs(flat["transport_motion_weight"]))) > 0.0
    optimizer = optim.Adam(learning_rate=0.01)
    optimizer.update(controller, gradients)
    mx.eval(controller.parameters(), optimizer.state)

    assert controller.parameter_sha256() != before_controller
    assert readout_fingerprint(model, spec.coda_start) == before_readout


def test_exact_state_teacher_shapes_recurrent_tissue_without_entering_prompt() -> None:
    model = _model()
    controller = _controller(literal_digit_token_ids=tuple(range(10, 20)))
    trace = StructuredTransitionTrace(
        family="boolean",
        depth=3,
        field_names=("pc", "value", "done"),
        states=((0, 0, 0), (1, 1, 0), (2, 0, 0), (3, 1, 1)),
    )
    program = StructuredTransitionProgram(
        state_trace=trace,
        action_field_names=("opcode", "operand", "has_operand"),
        actions=((0, 1, 1), (1, 0, 1), (2, 1, 1)),
    )

    def objective(candidate: UnifiedRecurrentController):
        return unified_intrinsic_training_loss(
            model,
            TOKENS,
            ANSWERS,
            candidate,
            _spec(),
            transition_trace=trace,
            transition_program=program,
        )[0]

    loss, gradients = nn.value_and_grad(controller, objective)(controller)
    flat = dict(tree_flatten(gradients))
    mx.eval(loss, gradients)
    assert float(mx.max(mx.abs(flat["state_transition_output"]))) > 0.0
    assert float(mx.max(mx.abs(flat["state_transition_query"]))) > 0.0
    assert float(mx.max(mx.abs(flat["state_value_embeddings"]))) > 0.0
    assert float(mx.max(mx.abs(flat["action_output"]))) > 0.0
    _loss, receipt = unified_intrinsic_training_loss(
        model,
        TOKENS,
        ANSWERS,
        controller,
        _spec(),
        transition_trace=trace,
        transition_program=program,
        state_teacher_forcing_probability=0.75,
    )
    assert receipt["state_supervision"]["available"] is True
    assert receipt["state_supervision"]["evaluator_only"] is True
    assert receipt["state_supervision"]["serialized_into_model_input"] is False
    assert receipt["state_supervision"]["teacher_forcing_probability"] == 0.75
    assert receipt["state_supervision"]["teacher_available_at_inference"] is False
    assert receipt["state_loss"] > 0.0
    assert receipt["per_depth"]["T1"]["state_loss"] == 0.0
    assert receipt["per_depth"]["T2"]["state_loss"] == 0.0
    assert receipt["per_depth"]["T3"]["state_step_accuracy"]
    assert receipt["per_depth"]["T3"]["initial_state_loss"] > 0.0
    assert receipt["per_depth"]["T3"]["initial_state_accuracy"] is not None
    assert receipt["per_depth"]["T3"]["action_loss"] > 0.0
    assert receipt["per_depth"]["T3"]["action_accuracy"] is not None
    commitment = receipt["state_supervision"]["commitments"]["T3"]
    assert commitment["private_values_exposed"] is False
    assert "values" not in commitment
    assert commitment["action"]["private_values_exposed"] is False


def test_initial_state_loss_ignores_inactive_padding_slots() -> None:
    from core.learning.recurrent_state_schema import state_targets_from_trace

    trace = StructuredTransitionTrace(
        family="boolean",
        depth=1,
        field_names=("pc", "value", "done"),
        states=((0, 1, 0), (1, 0, 1)),
    )
    targets = state_targets_from_trace(trace, 1)
    logits = mx.full((1, 5, 33), -20.0)
    for slot, value in enumerate(targets.initial_values):
        logits[0, slot, value] = 20.0
    # Inactive slots are deliberately wrong; they must not inflate or reduce
    # the scientific state score.
    logits[0, 2, :] = -20.0
    logits[0, 2, 17] = 20.0
    logits[0, 3, :] = -20.0
    logits[0, 3, 18] = 20.0
    loss, accuracy = structured_initial_state_loss(logits, targets)
    assert float(loss.item()) < 1e-6
    assert accuracy == pytest.approx(1.0)
    breakdown = structured_initial_state_accuracy_breakdown(logits, targets)
    assert breakdown == {
        "value_accuracy": 1.0,
        "value_exact_accuracy": 1.0,
        "control_accuracy": 1.0,
    }


def test_action_accuracy_excludes_post_completion_null_padding() -> None:
    trace = StructuredTransitionTrace(
        family="boolean",
        depth=1,
        field_names=("pc", "value", "done"),
        states=((0, 1, 0), (1, 0, 1)),
    )
    program = StructuredTransitionProgram(
        state_trace=trace,
        action_field_names=("opcode", "operand", "has_operand"),
        actions=((2, 1, 1),),
    )
    targets = action_targets_from_program(program, 4)
    action_width = len(targets.values[0])
    logits = [mx.full((1, action_width, 33), -20.0) for _ in range(4)]
    for step in range(4):
        for slot, value in enumerate(targets.values[step]):
            logits[step][0, slot, value] = 20.0
    # The one real operation is wrong while all post-completion null fields are
    # right.  The scientific score must remain zero, not an inflated average.
    for slot, active in enumerate(targets.masks[0]):
        if not active:
            continue
        logits[0][0, slot, :] = -20.0
        wrong = (targets.values[0][slot] + 1) % 32
        logits[0][0, slot, wrong] = 20.0
    _loss, accuracy, step_accuracy = structured_action_loss(logits, targets)
    assert accuracy == pytest.approx(0.0)
    assert step_accuracy == pytest.approx((0.0, 1.0, 1.0, 1.0))
    action_breakdown = structured_action_accuracy_breakdown(logits, targets)
    assert action_breakdown["instruction_exact_accuracy"] == pytest.approx(0.0)


def test_training_receipt_keeps_heldout_depths_unopened() -> None:
    loss, receipt = unified_intrinsic_training_loss(
        _model(),
        TOKENS,
        ANSWERS,
        _controller(),
        _spec(),
    )
    assert float(loss.item()) == pytest.approx(receipt["total"])
    assert set(receipt["per_depth"]) == {"T1", "T2", "T3"}
    assert receipt["heldout_depths_unopened"] == [5, 8]
    assert receipt["readout_frozen_by_training_contract"] is True
