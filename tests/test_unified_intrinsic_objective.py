"""The unified recurrence learns semantics without moving its readout."""

from __future__ import annotations

import math

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
from core.learning.recurrent_state_schema import (  # noqa: E402
    STATE_INVALID,
    state_targets_from_trace,
)
from core.learning.unified_intrinsic_objective import (  # noqa: E402
    UnifiedIntrinsicTrainingSpec,
    readout_fingerprint,
    structured_action_accuracy_breakdown,
    structured_action_loss,
    structured_initial_state_accuracy_breakdown,
    structured_initial_state_loss,
    structured_state_trajectory_diagnostics,
    unified_answer_and_recurrent_trajectory,
    unified_answer_trajectory,
    unified_intrinsic_training_loss,
    unified_process_training_loss,
    unified_typed_transition_processor_loss,
)
from core.learning.unified_intrinsic_recurrence import (  # noqa: E402
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
)


def _model() -> Model:
    mx.random.seed(7)
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


def test_state_trajectory_diagnostics_separate_execution_recovery_and_padding() -> None:
    controller = _controller(literal_digit_token_ids=tuple(range(10, 20)))
    trace = StructuredTransitionTrace(
        family="diagnostic",
        depth=3,
        field_names=("pc", "value0", "value1", "value2", "done"),
        states=(
            (0, 0, 0, 0, 0),
            (1, 1, 0, 0, 0),
            (2, 2, 0, 0, 0),
            (3, 3, 0, 0, 1),
        ),
    )
    targets = state_targets_from_trace(trace, 5)
    predicted = (
        targets.values[0],
        (2, 9, 0, 0, 0),
        targets.values[2],
        targets.values[3],
        (3, 8, 0, 0, 1),
    )
    logits = tuple(
        controller.exact_probabilities(
            row,
            slots=controller.config.state_slots,
            cardinality=controller.config.state_cardinality,
        )
        for row in predicted
    )

    report = structured_state_trajectory_diagnostics(
        logits,
        targets,
        active_steps=3,
    )

    assert report["active_steps"] == 3
    assert report["padding_steps"] == 2
    assert report["active_state_exact_accuracy"] == pytest.approx(2 / 3)
    assert report["active_value_exact_accuracy"] == pytest.approx(2 / 3)
    assert report["active_trajectory_exact"] is False
    assert report["first_error_step"] == 2
    assert report["first_error_fraction"] == pytest.approx(1 / 3)
    assert report["recovery_observable"] is True
    assert report["recovered_after_first_error"] is True
    assert report["sustained_recovery_after_first_error"] is True
    assert report["p_correct_given_previous_correct"] == 0.0
    assert report["p_correct_given_previous_wrong"] == 1.0
    assert report["terminal_stability_observable"] is True
    assert report["terminal_correct_stable"] is False
    assert report["terminal_self_stable"] is False
    assert report["per_register_accuracy"]["value0"] == pytest.approx(2 / 3)


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
    recurrent, all_hidden, all_losses, _state_logits = unified_answer_and_recurrent_trajectory(
        model,
        TOKENS,
        ANSWERS,
        plan,
        controller,
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


def test_answer_trajectory_can_execute_public_actions_without_microcode() -> None:
    actions = (
        (10, 1, 2, 3, 4, 5, 6, 0),
        (10, 2, 3, 4, 5, 6, 7, 0),
        (10, 3, 4, 5, 6, 7, 8, 1),
    )
    recurrent, hidden, losses, states = unified_answer_and_recurrent_trajectory(
        _model(),
        TOKENS,
        ANSWERS,
        _spec().plan_at(3),
        _controller(literal_digit_token_ids=tuple(range(10))),
        use_state_slots=True,
        public_action_values=actions,
        microcode_lesion=True,
        answer_digit_pointer_enabled=False,
        final_answer_only=True,
    )
    mx.eval(recurrent, hidden, losses, states)

    assert len(recurrent) == len(states) == 3
    assert len(hidden) == len(losses) == 1
    assert float(losses[0].item()) > 0.0


def test_answer_trajectory_exposes_transition_processor_and_history_lesions() -> None:
    model = _model()
    controller = _controller(literal_digit_token_ids=tuple(range(10)))
    actions = (
        (10, 1, 2, 3, 4, 5, 6, 0),
        (10, 2, 3, 4, 5, 6, 7, 0),
        (10, 3, 4, 5, 6, 7, 8, 1),
    )
    _normal_recurrent, _normal_hidden, _normal_losses, normal_states = (
        unified_answer_and_recurrent_trajectory(
            model,
            TOKENS,
            ANSWERS,
            _spec().plan_at(3),
            controller,
            use_state_slots=True,
            public_action_values=actions,
            microcode_lesion=True,
            answer_digit_pointer_enabled=False,
        )
    )
    _lesioned_recurrent, _lesioned_hidden, _lesioned_losses, lesioned_states = (
        unified_answer_and_recurrent_trajectory(
            model,
            TOKENS,
            ANSWERS,
            _spec().plan_at(3),
            controller,
            use_state_slots=True,
            public_action_values=actions,
            microcode_lesion=True,
            transition_processor_lesion=True,
            transition_history_lesion=True,
            answer_digit_pointer_enabled=False,
        )
    )
    mx.eval(*normal_states, *lesioned_states)

    assert len(normal_states) == len(lesioned_states) == 3
    assert not bool(mx.allclose(normal_states[-1], lesioned_states[-1]))


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
    assert float(teacher_losses[-1].item()) != pytest.approx(float(student_losses[-1].item()))
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
    assert float(mx.max(mx.abs(flat["action_workspace_output"]))) > 0.0
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


def test_process_objective_uses_public_prompt_without_answer_or_coda_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.learning.unified_intrinsic_recurrence as recurrence

    model = _model()
    controller = _controller(literal_digit_token_ids=tuple(range(10, 20)))
    layer_calls: list[int] = []
    actual_run = recurrence._run

    def tracked_run(layers, hidden, caches=None):
        materialized = list(layers)
        layer_calls.append(len(materialized))
        return actual_run(materialized, hidden, caches)

    monkeypatch.setattr(recurrence, "_run", tracked_run)
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
        return unified_process_training_loss(
            model,
            TOKENS,
            candidate,
            _spec().plan_at(3),
            transition_trace=trace,
            transition_program=program,
            state_teacher_forcing_probability=1.0,
        )[0]

    loss, gradients = nn.value_and_grad(controller, objective)(controller)
    flat = dict(tree_flatten(gradients))
    mx.eval(loss, gradients)
    assert float(loss.item()) > 0.0
    assert float(mx.max(mx.abs(flat["action_output"]))) > 0.0
    assert float(mx.max(mx.abs(flat["initial_state_output"]))) > 0.0
    _loss, receipt = unified_process_training_loss(
        model,
        TOKENS,
        controller,
        _spec().plan_at(3),
        transition_trace=trace,
        transition_program=program,
        state_teacher_forcing_probability=1.0,
    )
    assert receipt["objective"] == "prompt_only_typed_process"
    assert receipt["answer_tokens_exposed"] is False
    assert receipt["answer_or_coda_graph_constructed"] is False
    assert receipt["problem_evidence_gradient"] == "scoped_transformer_enabled"
    assert layer_calls == [2, 2, 2, 2]


def test_process_objective_can_train_each_causal_component_exclusively() -> None:
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

    for component in (
        "initializer",
        "action",
        "action_workspace",
        "transition",
        "joint",
    ):
        loss, receipt = unified_process_training_loss(
            model,
            TOKENS,
            controller,
            _spec().plan_at(3),
            transition_trace=trace,
            transition_program=program,
            state_teacher_forcing_probability=1.0,
            component=component,
        )
        mx.eval(loss)
        assert receipt["component"] == component
        assert float(loss.item()) == pytest.approx(receipt["component_losses"][component])

    with pytest.raises(ValueError, match="component is invalid"):
        unified_process_training_loss(
            model,
            TOKENS,
            controller,
            _spec().plan_at(3),
            transition_trace=trace,
            transition_program=program,
            state_teacher_forcing_probability=1.0,
            component="answer",
        )


def test_public_actions_train_learned_transition_with_exact_microcode_removed() -> None:
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
    public_actions = action_targets_from_program(program, 3).values

    def objective(candidate: UnifiedRecurrentController):
        return unified_process_training_loss(
            model,
            TOKENS,
            candidate,
            _spec().plan_at(3),
            transition_trace=trace,
            transition_program=program,
            state_teacher_forcing_probability=1.0,
            component="transition",
            public_action_values=public_actions,
            microcode_lesion=True,
        )[0]

    loss, gradients = nn.value_and_grad(controller, objective)(controller)
    flat = dict(tree_flatten(gradients))
    mx.eval(loss, gradients)
    assert float(loss.item()) > 0.0
    assert float(mx.max(mx.abs(flat["state_transition_output"]))) > 0.0
    assert float(mx.max(mx.abs(flat["action_output"]))) == 0.0
    _loss, receipt = unified_process_training_loss(
        model,
        TOKENS,
        controller,
        _spec().plan_at(3),
        transition_trace=trace,
        transition_program=program,
        state_teacher_forcing_probability=1.0,
        component="transition",
        public_action_values=public_actions,
        microcode_lesion=True,
    )
    assert receipt["public_action_program"] is True
    assert receipt["public_actions_are_correctness_authority"] is False
    assert receipt["exact_microcode_available"] is False
    assert receipt["action_accuracy"] is None

    with pytest.raises(ValueError, match="bypass"):
        unified_process_training_loss(
            model,
            TOKENS,
            controller,
            _spec().plan_at(3),
            transition_trace=trace,
            transition_program=program,
            state_teacher_forcing_probability=1.0,
            component="action",
            public_action_values=public_actions,
            microcode_lesion=True,
        )


def test_direct_transition_objective_reaches_only_categorical_processor() -> None:
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
    public_actions = action_targets_from_program(program, 3).values

    def objective(candidate: UnifiedRecurrentController):
        return unified_typed_transition_processor_loss(
            candidate,
            _spec().plan_at(3),
            transition_trace=trace,
            transition_program=program,
            public_action_values=public_actions,
        )[0]

    loss, gradients = nn.value_and_grad(controller, objective)(controller)
    flat = dict(tree_flatten(gradients))
    mx.eval(loss, gradients)

    assert float(loss.item()) > 0.0
    assert float(mx.max(mx.abs(flat["transition_processor_output"]))) > 0.0
    assert (
        float(mx.max(mx.abs(flat["transition_processor_opcode_output"]))) > 0.0
    )
    assert float(mx.max(mx.abs(flat["state_transition_output"]))) == 0.0
    _loss, receipt = unified_typed_transition_processor_loss(
        controller,
        _spec().plan_at(3),
        transition_trace=trace,
        transition_program=program,
        public_action_values=public_actions,
    )
    assert receipt["transformer_graph_constructed"] is False
    assert receipt["readout_graph_constructed"] is False
    assert receipt["answer_tokens_exposed"] is False
    assert receipt["deployed_transition_policy"] == "processor_authoritative"
    assert receipt["legacy_transition_logits_available"] is False
    assert receipt["opcode_expert_routing"] == "opcode"


def test_direct_transition_replay_gets_verified_prefix_only_supervision() -> None:
    controller = _controller(literal_digit_token_ids=tuple(range(10, 20)))
    trace = StructuredTransitionTrace(
        family="boolean",
        depth=2,
        field_names=("pc", "value", "done"),
        states=((0, 0, 0), (1, 1, 0), (2, 0, 1)),
    )
    program = StructuredTransitionProgram(
        state_trace=trace,
        action_field_names=("opcode", "operand", "has_operand"),
        actions=((0, 1, 1), (1, 0, 1)),
    )
    public_actions = action_targets_from_program(program, 2).values

    def objective(candidate: UnifiedRecurrentController):
        return unified_typed_transition_processor_loss(
            candidate,
            _spec().plan_at(2),
            transition_trace=trace,
            transition_program=program,
            public_action_values=public_actions,
            transition_replay_mode="active",
            replay_auxiliary_weight=1.0,
        )[0]

    loss, gradients = nn.value_and_grad(controller, objective)(controller)
    flat = dict(tree_flatten(gradients))
    mx.eval(loss, gradients)

    assert float(loss.item()) > 0.0
    assert float(mx.max(mx.abs(flat["transition_replay_output"]))) > 0.0
    assert (
        float(mx.max(mx.abs(flat["transition_replay_opcode_output"]))) > 0.0
    )
    _loss, receipt = unified_typed_transition_processor_loss(
        controller,
        _spec().plan_at(2),
        transition_trace=trace,
        transition_program=program,
        public_action_values=public_actions,
        transition_replay_mode="active",
        replay_auxiliary_weight=1.0,
    )
    assert receipt["transition_replay"]["state_independent_public_prefix"] is True
    assert receipt["transition_replay"]["auxiliary_loss"] is not None
    assert receipt["answer_tokens_exposed"] is False


def test_direct_transition_objective_rolls_its_own_prediction_forward() -> None:
    controller = _controller(literal_digit_token_ids=tuple(range(10, 20)))
    trace = StructuredTransitionTrace(
        family="boolean",
        depth=2,
        field_names=("pc", "value", "done"),
        states=((0, 0, 0), (1, 1, 0), (2, 0, 1)),
    )
    program = StructuredTransitionProgram(
        state_trace=trace,
        action_field_names=("opcode", "operand", "has_operand"),
        actions=((0, 0, 0), (0, 0, 0)),
    )
    public_actions = action_targets_from_program(program, 3).values
    observed_inputs: list[tuple[int, ...]] = []
    original = controller.typed_transition_processor_logits

    def observe_student_state(
        state_probabilities,
        action_probabilities,
        history_memory,
        *,
        opcode_expert_routing="opcode",
    ):
        assert opcode_expert_routing == "opcode"
        observed_inputs.append(
            tuple(int(value) for value in mx.argmax(state_probabilities, axis=-1)[0].tolist())
        )
        logits = mx.full(
            (
                1,
                controller.config.state_slots,
                controller.config.state_cardinality,
            ),
            -8.0,
        )
        predicted = (
            (7, 9, 0, 0, 0)
            if len(observed_inputs) == 1
            else (8, 0, 0, 0, 1)
        )
        for slot, value in enumerate(predicted):
            logits = logits.at[0, slot, value].add(16.0)
        return logits

    controller.typed_transition_processor_logits = observe_student_state
    try:
        _loss, receipt = unified_typed_transition_processor_loss(
            controller,
            _spec().plan_at(3),
            transition_trace=trace,
            transition_program=program,
            public_action_values=public_actions,
        )
    finally:
        controller.typed_transition_processor_logits = original

    assert observed_inputs[0] == (0, 0, 0, 0, 0)
    assert observed_inputs[1] == (7, 9, 0, 0, 0)
    assert observed_inputs[1] != (1, 1, 0, 0, 0)
    assert len(observed_inputs) == 2
    assert receipt["closed_loop_student_rollout"] is True
    assert receipt["rollout_state_authority"] == "student_prediction_after_initial"
    assert (
        receipt["transition_target_authority"]
        == "exact_transition_from_actual_committed_state"
    )
    assert receipt["gold_trace_used_after_initial"] == "consistency_check_only"
    assert receipt["off_reference_transition_count"] == 1
    assert receipt["dynamic_target_difference_count"] == 1
    assert receipt["reference_inconsistency_count"] == 0
    assert receipt["state_accuracy"] == receipt["local_transition_accuracy"]
    assert receipt["local_transition_accuracy"] > (
        receipt["verified_trace_position_accuracy"]
    )
    assert receipt["verified_trace_position_accuracy_scope"] == (
        "diagnostic_nominal_step_comparison_including_off_reference_states"
    )
    assert receipt["post_terminal_transitions_trained"] == 0
    assert receipt["active_transitions"] == 2


def test_invalid_recurrent_state_is_an_absorbing_structural_latch() -> None:
    controller = _controller(literal_digit_token_ids=tuple(range(10, 20)))
    state = controller.exact_probabilities(
        (0, STATE_INVALID, 0, 0, 0),
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    action = controller.exact_probabilities(
        (0, 1, 32, 32, 32, 32, 32, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )
    history = controller._typed_transition_memory(
        (action,),
        state_probabilities=state,
        action_probabilities=action,
    )

    learned = controller.resolve_transition_processor_logits(
        None,
        state,
        action,
        history,
        transition_processor_mode="authoritative",
    )
    exact, recognized = controller.microcode_transition_logits(
        state,
        action,
        action_probability_history=(action,),
    )
    mx.eval(learned, exact, recognized)

    expected = [STATE_INVALID] * controller.config.state_slots
    assert mx.argmax(learned, axis=-1)[0].tolist() == expected
    assert mx.argmax(exact, axis=-1)[0].tolist() == expected
    assert bool(mx.all(recognized).item()) is True


def test_out_of_vocabulary_transition_result_enters_invalid_latch() -> None:
    controller = _controller(literal_digit_token_ids=tuple(range(10, 20)))
    state = controller.exact_probabilities(
        (0, 0, 0, 0, 0),
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    action = controller.exact_probabilities(
        (15, 1, 31, 31, 0, 0, 32, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )

    exact, recognized = controller.microcode_transition_logits(
        state,
        action,
        action_probability_history=(action,),
    )
    mx.eval(exact, recognized)

    assert mx.argmax(exact, axis=-1)[0].tolist() == [STATE_INVALID] * 5
    assert bool(mx.all(recognized).item()) is True


def test_direct_transition_refuses_gold_that_disagrees_with_public_execution() -> None:
    controller = _controller(literal_digit_token_ids=tuple(range(10, 20)))

    def evidence(value: int) -> tuple[StructuredTransitionTrace, StructuredTransitionProgram]:
        trace = StructuredTransitionTrace(
            family="boolean",
            depth=2,
            field_names=("pc", "value", "done"),
            states=((0, 0, 0), (1, value, 0), (2, 1 - value, 1)),
        )
        return trace, StructuredTransitionProgram(
            state_trace=trace,
            action_field_names=("opcode", "operand", "has_operand"),
            actions=((0, 0, 0), (0, 0, 0)),
        )

    trace_a, program_a = evidence(1)
    trace_b, program_b = evidence(0)
    public_actions = action_targets_from_program(program_a, 2).values
    assert public_actions == action_targets_from_program(program_b, 2).values
    captures: list[list[tuple[tuple[int, ...], tuple[float, ...]]]] = []
    original = controller.typed_transition_processor_logits

    for index, (trace, program) in enumerate(
        ((trace_a, program_a), (trace_b, program_b))
    ):
        run: list[tuple[tuple[int, ...], tuple[float, ...]]] = []

        def capture(
            state_probabilities,
            action_probabilities,
            history_memory,
            *,
            opcode_expert_routing="opcode",
            _run=run,
        ):
            logits = original(
                state_probabilities,
                action_probabilities,
                history_memory,
                opcode_expert_routing=opcode_expert_routing,
            )
            mx.eval(state_probabilities, logits)
            _run.append(
                (
                    tuple(
                        int(value)
                        for value in mx.argmax(state_probabilities[0], axis=-1).tolist()
                    ),
                    tuple(float(value) for value in logits.flatten().tolist()),
                )
            )
            return logits

        controller.typed_transition_processor_logits = capture
        try:
            if index == 0:
                unified_typed_transition_processor_loss(
                    controller,
                    _spec().plan_at(2),
                    transition_trace=trace,
                    transition_program=program,
                    public_action_values=public_actions,
                )
            else:
                with pytest.raises(
                    ValueError,
                    match="exact transition authority differs",
                ):
                    unified_typed_transition_processor_loss(
                        controller,
                        _spec().plan_at(2),
                        transition_trace=trace,
                        transition_program=program,
                        public_action_values=public_actions,
                    )
        finally:
            controller.typed_transition_processor_logits = original
        captures.append(run)

    assert captures[0][0] == captures[1][0]


def test_direct_transition_tape_never_reads_future_public_actions() -> None:
    controller = _controller(literal_digit_token_ids=tuple(range(10, 20)))

    def evidence(
        final_value: int,
        future_operand: int,
    ) -> tuple[StructuredTransitionTrace, StructuredTransitionProgram]:
        trace = StructuredTransitionTrace(
            family="boolean",
            depth=2,
            field_names=("pc", "value", "done"),
            states=((0, 0, 0), (1, 1, 0), (2, final_value, 1)),
        )
        return trace, StructuredTransitionProgram(
            state_trace=trace,
            action_field_names=("opcode", "operand", "has_operand"),
            actions=((0, 1, 1), (1, future_operand, 1)),
        )

    first_trace, first_program = evidence(0, 0)
    second_trace, second_program = evidence(1, 1)
    first_reads: list[tuple[float, ...]] = []
    original = controller.typed_transition_processor_logits

    for trace, program in (
        (first_trace, first_program),
        (second_trace, second_program),
    ):
        reads: list[tuple[float, ...]] = []

        def capture(
            state_probabilities,
            action_probabilities,
            history_memory,
            *,
            opcode_expert_routing="opcode",
            _reads=reads,
        ):
            mx.eval(history_memory)
            _reads.append(tuple(float(value) for value in history_memory.flatten().tolist()))
            return original(
                state_probabilities,
                action_probabilities,
                history_memory,
                opcode_expert_routing=opcode_expert_routing,
            )

        controller.typed_transition_processor_logits = capture
        try:
            public_actions = action_targets_from_program(program, 2).values
            unified_typed_transition_processor_loss(
                controller,
                _spec().plan_at(2),
                transition_trace=trace,
                transition_program=program,
                public_action_values=public_actions,
            )
        finally:
            controller.typed_transition_processor_logits = original
        first_reads.append(reads[0])

    assert first_reads[0] == first_reads[1]


def test_direct_transition_curriculum_keeps_prior_public_tape_for_midtrace_window() -> None:
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
    observed_states: list[tuple[int, ...]] = []
    observed_prefix_reads: list[tuple[float, ...]] = []
    original = controller.typed_transition_processor_logits

    def capture(
        state_probabilities,
        action_probabilities,
        history_memory,
        *,
        opcode_expert_routing="opcode",
    ):
        mx.eval(state_probabilities, history_memory)
        observed_states.append(
            tuple(
                int(value)
                for value in mx.argmax(state_probabilities[0], axis=-1).tolist()
            )
        )
        observed_prefix_reads.append(
            tuple(float(value) for value in history_memory.flatten().tolist())
        )
        return original(
            state_probabilities,
            action_probabilities,
            history_memory,
            opcode_expert_routing=opcode_expert_routing,
        )

    controller.typed_transition_processor_logits = capture
    try:
        _loss, receipt = unified_typed_transition_processor_loss(
            controller,
            _spec().plan_at(3),
            transition_trace=trace,
            transition_program=program,
            public_action_values=action_targets_from_program(program, 3).values,
            transition_start=1,
            transition_count=1,
        )
    finally:
        controller.typed_transition_processor_logits = original

    assert observed_states == [(1, 1, 0, 0, 0)]
    assert len(observed_prefix_reads) == 1
    assert receipt["initial_state_authority"] == "training_only_verified_midtrace_state"
    assert receipt["transition_start"] == 1
    assert receipt["transition_stop"] == 2
    assert receipt["complete_public_prefix_visible"] is True


def test_direct_transition_recovery_curriculum_injects_no_runtime_oracle() -> None:
    controller = _controller(literal_digit_token_ids=tuple(range(10, 20)))
    trace = StructuredTransitionTrace(
        family="boolean",
        depth=2,
        field_names=("pc", "value", "done"),
        states=((0, 0, 0), (1, 1, 0), (2, 0, 1)),
    )
    program = StructuredTransitionProgram(
        state_trace=trace,
        action_field_names=("opcode", "operand", "has_operand"),
        actions=((0, 1, 1), (1, 0, 1)),
    )
    observed: list[tuple[int, ...]] = []
    original = controller.typed_transition_processor_logits

    def capture(
        state_probabilities,
        action_probabilities,
        history_memory,
        *,
        opcode_expert_routing="opcode",
    ):
        mx.eval(state_probabilities)
        observed.append(
            tuple(
                int(value)
                for value in mx.argmax(state_probabilities[0], axis=-1).tolist()
            )
        )
        return original(
            state_probabilities,
            action_probabilities,
            history_memory,
            opcode_expert_routing=opcode_expert_routing,
        )

    controller.typed_transition_processor_logits = capture
    try:
        _loss, receipt = unified_typed_transition_processor_loss(
            controller,
            _spec().plan_at(2),
            transition_trace=trace,
            transition_program=program,
            public_action_values=action_targets_from_program(program, 2).values,
            corrupt_transition=0,
            corrupt_state_slot=1,
            corrupt_state_offset=2,
        )
    finally:
        controller.typed_transition_processor_logits = original

    assert observed[0] == (0, 2, 0, 0, 0)
    assert receipt["controlled_state_corruption"] == {
        "enabled": True,
        "transition": 0,
        "mode": "single_slot_offset",
        "slot": 1,
        "offset": 2,
        "coherent_trace_source_index": None,
        "runtime_correctness_oracle_available": False,
        "target_authority": "true_transition_from_corrupted_state",
    }


def test_recovery_curriculum_transplants_a_coherent_off_path_state() -> None:
    controller = _controller(literal_digit_token_ids=tuple(range(10, 20)))
    trace = StructuredTransitionTrace(
        family="boolean",
        depth=2,
        field_names=("pc", "value", "done"),
        states=((0, 0, 0), (1, 1, 0), (2, 0, 1)),
    )
    program = StructuredTransitionProgram(
        state_trace=trace,
        action_field_names=("opcode", "operand", "has_operand"),
        actions=((0, 1, 1), (1, 0, 1)),
    )
    observed: list[tuple[int, ...]] = []
    original = controller.typed_transition_processor_logits

    def capture(
        state_probabilities,
        action_probabilities,
        history_memory,
        *,
        opcode_expert_routing="opcode",
    ):
        mx.eval(state_probabilities)
        observed.append(
            tuple(int(value) for value in mx.argmax(state_probabilities[0], axis=-1).tolist())
        )
        return original(
            state_probabilities,
            action_probabilities,
            history_memory,
            opcode_expert_routing=opcode_expert_routing,
        )

    controller.typed_transition_processor_logits = capture
    try:
        _loss, receipt = unified_typed_transition_processor_loss(
            controller,
            _spec().plan_at(2),
            transition_trace=trace,
            transition_program=program,
            public_action_values=action_targets_from_program(program, 2).values,
            corrupt_transition=0,
            corrupt_state_mode="coherent_trace_state",
            corrupt_state_offset=1,
        )
    finally:
        controller.typed_transition_processor_logits = original

    assert observed[0] == (1, 1, 0, 0, 0)
    assert receipt["controlled_state_corruption"]["mode"] == "coherent_trace_state"
    assert receipt["controlled_state_corruption"]["slot"] is None
    assert receipt["controlled_state_corruption"]["coherent_trace_source_index"] == 1


def test_direct_transition_loss_weights_value_registers_and_weakest_term() -> None:
    controller = _controller(literal_digit_token_ids=tuple(range(10, 20)))
    trace = StructuredTransitionTrace(
        family="boolean",
        depth=1,
        field_names=("pc", "value", "done"),
        states=((0, 0, 0), (1, 1, 1)),
    )
    program = StructuredTransitionProgram(
        state_trace=trace,
        action_field_names=("opcode", "operand", "has_operand"),
        actions=((0, 1, 1),),
    )
    public_actions = action_targets_from_program(program, 1).values

    base, base_receipt = unified_typed_transition_processor_loss(
        controller,
        _spec().plan_at(1),
        transition_trace=trace,
        transition_program=program,
        public_action_values=public_actions,
    )
    strongest, strongest_receipt = unified_typed_transition_processor_loss(
        controller,
        _spec().plan_at(1),
        transition_trace=trace,
        transition_program=program,
        public_action_values=public_actions,
        weakest_register_weight=0.25,
    )

    mx.eval(base, strongest)
    assert base_receipt["register_loss_weights"] == [1.0, 4.0, 4.0, 4.0, 1.0]
    assert strongest_receipt["weakest_register_weight"] == pytest.approx(0.25)
    assert float(strongest.item() - base.item()) == pytest.approx(
        0.25 * math.log(controller.config.state_cardinality),
        rel=1e-5,
    )


@pytest.mark.parametrize(
    "transition_processor_mode",
    ["authoritative", "copy_write", "masked_copy_write"],
)
def test_direct_transition_objective_learns_exact_trace(
    transition_processor_mode: str,
) -> None:
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
    public_actions = action_targets_from_program(program, 3).values

    def objective(candidate: UnifiedRecurrentController):
        return unified_typed_transition_processor_loss(
            candidate,
            _spec().plan_at(3),
            transition_trace=trace,
            transition_program=program,
            public_action_values=public_actions,
            transition_processor_mode=transition_processor_mode,
            transition_copy_prior_logit_bias=(
                0.01
                if transition_processor_mode in {"copy_write", "masked_copy_write"}
                else 2.0
            ),
        )[0]

    initial_loss, initial_receipt = unified_typed_transition_processor_loss(
        controller,
        _spec().plan_at(3),
        transition_trace=trace,
        transition_program=program,
        public_action_values=public_actions,
        transition_processor_mode=transition_processor_mode,
        transition_copy_prior_logit_bias=(
            0.01
            if transition_processor_mode in {"copy_write", "masked_copy_write"}
            else 2.0
        ),
    )
    legacy_before = mx.array(controller.state_transition_output)
    optimizer = optim.Adam(learning_rate=0.01)
    for _step in range(120):
        loss, gradients = nn.value_and_grad(controller, objective)(controller)
        optimizer.update(controller, gradients)
        mx.eval(loss, controller.parameters(), optimizer.state)
    final_loss, final_receipt = unified_typed_transition_processor_loss(
        controller,
        _spec().plan_at(3),
        transition_trace=trace,
        transition_program=program,
        public_action_values=public_actions,
        transition_processor_mode=transition_processor_mode,
        transition_copy_prior_logit_bias=(
            0.01
            if transition_processor_mode in {"copy_write", "masked_copy_write"}
            else 2.0
        ),
    )
    mx.eval(initial_loss, final_loss, legacy_before, controller.state_transition_output)

    assert float(final_loss.item()) < float(initial_loss.item()) * 0.05
    assert final_receipt["state_accuracy"] > initial_receipt["state_accuracy"]
    assert final_receipt["state_accuracy"] == 1.0
    assert final_receipt["transition_processor_mode"] == transition_processor_mode
    assert final_receipt["transition_copy_prior_logit_bias"] == (
        0.01
        if transition_processor_mode in {"copy_write", "masked_copy_write"}
        else 2.0
    )
    assert mx.array_equal(controller.state_transition_output, legacy_before).item()


def test_process_objective_reaches_scoped_transformer_tissue() -> None:
    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_scope,
    )
    from tools.train_unified_intrinsic_recurrence import _configure_window_tissue

    model = _model()
    _configure_window_tissue(
        model,
        _spec(),
        mode="scoped_lora",
        rank=2,
        targets=("o_proj",),
        depth_basis_size=2,
    )
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

    def objective(candidate: Model):
        with recurrence_adapter_scope(start=None, stop=None):
            return unified_process_training_loss(
                candidate,
                TOKENS,
                controller,
                _spec().plan_at(3),
                transition_trace=trace,
                transition_program=program,
                state_teacher_forcing_probability=1.0,
            )[0]

    loss, gradients = nn.value_and_grad(model, objective)(model)
    flat = dict(tree_flatten(gradients))
    adapter_gradients = [
        value
        for name, value in flat.items()
        if name.endswith((".lora_a", ".lora_b")) and "continuous_depth_" not in name
    ]
    mx.eval(loss, gradients)
    assert adapter_gradients
    assert any(float(mx.max(mx.abs(value))) > 0.0 for value in adapter_gradients)


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


def test_action_loss_places_extra_pressure_on_the_weakest_active_field() -> None:
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
    targets = action_targets_from_program(program, 1)
    width = len(targets.values[0])
    decision = mx.full((1, width, 33), -8.0)
    for slot, value in enumerate(targets.values[0]):
        decision[0, slot, value] = 8.0
    weakest = next(index for index, active in enumerate(targets.masks[0]) if active)
    target = targets.values[0][weakest]
    decision[0, weakest, target] = -8.0
    decision[0, weakest, (target + 1) % 33] = 8.0

    loss, _accuracy, _steps = structured_action_loss([decision], targets)
    labels = mx.array(targets.values[0], dtype=mx.int32)
    per_slot = nn.losses.cross_entropy(decision[0], labels, reduction="none")
    mask = mx.array(targets.masks[0], dtype=mx.float32)
    flat_active_mean = mx.sum(per_slot * mask) / mx.sum(mask)
    inactive = 1.0 - mask
    flat_null_mean = mx.sum(per_slot * inactive) / mx.maximum(mx.sum(inactive), 1.0)

    assert float(loss.item()) > float((flat_active_mean + 0.1 * flat_null_mean).item())


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
