"""The unified recurrence learns semantics without moving its readout."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
optim = pytest.importorskip("mlx.optimizers")
pytest.importorskip("mlx_lm")

from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.learning.recurrence_curriculum import StructuredTransitionTrace  # noqa: E402
from core.learning.unified_intrinsic_objective import (  # noqa: E402
    UnifiedIntrinsicTrainingSpec,
    readout_fingerprint,
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


def _controller() -> UnifiedRecurrentController:
    return UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=32,
            correction_rank=4,
            minimum_iterations=1,
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
    controller = _controller()
    trace = StructuredTransitionTrace(
        family="boolean",
        depth=3,
        field_names=("pc", "value", "done"),
        states=((0, 0, 0), (1, 1, 0), (2, 0, 0), (3, 1, 1)),
    )

    def objective(candidate: UnifiedRecurrentController):
        return unified_intrinsic_training_loss(
            model,
            TOKENS,
            ANSWERS,
            candidate,
            _spec(),
            transition_trace=trace,
        )[0]

    loss, gradients = nn.value_and_grad(controller, objective)(controller)
    flat = dict(tree_flatten(gradients))
    mx.eval(loss, gradients)
    assert float(mx.max(mx.abs(flat["state_transition_output"]))) > 0.0
    assert float(mx.max(mx.abs(flat["state_transition_query"]))) > 0.0
    assert float(mx.max(mx.abs(flat["state_value_embeddings"]))) > 0.0
    _loss, receipt = unified_intrinsic_training_loss(
        model,
        TOKENS,
        ANSWERS,
        controller,
        _spec(),
        transition_trace=trace,
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
    commitment = receipt["state_supervision"]["commitments"]["T3"]
    assert commitment["private_values_exposed"] is False
    assert "values" not in commitment


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
