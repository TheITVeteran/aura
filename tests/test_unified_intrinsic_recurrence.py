"""Contracts for the single-path intrinsic recurrent controller."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.learning.intrinsic_recurrence import RecurrentDepthPlan  # noqa: E402
from core.learning.protected_memory import MemoryLayout  # noqa: E402
from core.learning.recurrent_answer_emission import (  # noqa: E402
    RecurrentAnswerEmissionContract,
)
from core.learning.unified_intrinsic_recurrence import (  # noqa: E402
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
    apply_terminal_answer_grammar,
    unified_recurrent_hidden_states,
    unified_recurrent_logits,
)


def _model() -> Model:
    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=64,
            num_hidden_layers=8,
            intermediate_size=128,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=128,
            num_key_value_heads=2,
            max_position_embeddings=256,
            rope_theta=10_000.0,
        )
    )
    mx.eval(model.parameters())
    return model


TOKENS = mx.array([[3, 11, 42, 7, 19, 23]])


def _controller() -> UnifiedRecurrentController:
    return UnifiedRecurrentController(
        UnifiedRecurrenceConfig(hidden_size=64, correction_rank=8)
    )


def test_identity_controller_preserves_base_forward_at_one_iteration() -> None:
    model = _model()
    controller = _controller()
    logits, telemetry = unified_recurrent_logits(
        model,
        TOKENS,
        RecurrentDepthPlan(2, 6, iterations=1),
        controller,
    )
    assert controller.identity_initialized()
    assert bool(mx.allclose(logits, model(TOKENS), atol=1e-5))
    assert telemetry.executed_iterations == 1
    assert telemetry.halted is False


def test_continuous_depth_basis_is_defined_and_distinct_beyond_train_depth() -> None:
    controller = _controller()
    at_four = controller.depth_features(4)
    at_sixteen = controller.depth_features(16)
    at_thousand = controller.depth_features(1_000)
    assert not bool(mx.array_equal(at_four, at_sixteen))
    assert not bool(mx.array_equal(at_sixteen, at_thousand))
    assert bool(mx.all(at_thousand <= 1.0))
    assert bool(mx.all(at_thousand >= 0.0))


def test_controller_initialization_seed_replays_and_varies() -> None:
    config = UnifiedRecurrenceConfig(hidden_size=64, initialization_seed=7)
    first = UnifiedRecurrentController(config)
    replay = UnifiedRecurrentController(config)
    other = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(hidden_size=64, initialization_seed=8)
    )
    assert first.parameter_sha256() == replay.parameter_sha256()
    assert first.parameter_sha256() != other.parameter_sha256()


def test_state_readout_uses_only_the_declared_public_prompt_position() -> None:
    controller = _controller()
    hidden = mx.zeros((1, 6, 64), dtype=mx.float32)
    baseline = controller.state_logits(hidden, public_token_count=4)
    private_changed = hidden.at[:, 4:, :].add(100.0)
    unchanged = controller.state_logits(private_changed, public_token_count=4)
    public_changed = hidden.at[:, 3, :].add(1.0)
    changed = controller.state_logits(public_changed, public_token_count=4)
    assert baseline.shape == (1, 5, 33)
    assert bool(mx.array_equal(baseline, unchanged))
    assert not bool(mx.array_equal(baseline, changed))


def test_learned_state_slots_enter_the_real_recurrent_sequence() -> None:
    model = _model()
    controller = _controller()
    _final, trajectory, _telemetry = unified_recurrent_hidden_states(
        model,
        TOKENS,
        RecurrentDepthPlan(2, 6, iterations=2),
        controller,
        state_slot_start=4,
    )
    assert trajectory[0].shape[1] == TOKENS.shape[1] + controller.config.state_slots
    logits = controller.state_logits(trajectory[-1], state_slot_start=4)
    assert logits.shape == (1, 5, 33)


def test_typed_state_decision_is_committed_as_next_step_input() -> None:
    model = _model()
    controller = _controller()
    decisions: list[object] = []
    decode_states: list[object] = []
    _final, trajectory, telemetry = unified_recurrent_hidden_states(
        model,
        TOKENS,
        RecurrentDepthPlan(2, 6, iterations=3),
        controller,
        state_slot_start=4,
        state_logit_trajectory=decisions,
        decode_state_trajectory=decode_states,
    )
    assert len(decisions) == len(decode_states) == len(trajectory) == 3
    assert all(decision.shape == (1, 5, 33) for decision in decisions)
    assert not bool(mx.array_equal(decode_states[-1], trajectory[-1]))
    assert telemetry.receipt()["teacher_available"] is False
    receipt = controller.receipt()
    assert receipt["typed_state_bottleneck"] == "straight_through_categorical"
    assert receipt["predicted_state_is_next_step_input"] is True
    assert receipt["state_processor"] == "shared_evidence_attention_transition"
    assert (
        receipt["state_problem_evidence"]
        == "frozen_deep_prefix_no_decoder_suffix"
    )
    assert receipt["transformer_answer_passes_per_state"] == 1
    assert (
        receipt["state_to_answer_bridge"]
        == "masked_action_state_process_tape_attention_over_frozen_readout"
    )
    assert receipt["terminal_decode_semantics"] == "first_terminal_state_preserved"


def test_typed_action_lesion_removes_selected_process_channel() -> None:
    model = _model()
    controller = _controller()
    controller.action_bias = mx.full((8, 33), -100.0)
    for slot, value in enumerate((9, 1, 2, 3, 4, 5, 6, 0)):
        controller.action_bias = controller.action_bias.at[slot, value].add(200.0)
    normal_states: list[object] = []
    lesioned_states: list[object] = []
    unified_recurrent_hidden_states(
        model,
        TOKENS,
        RecurrentDepthPlan(2, 6, iterations=3),
        controller,
        state_slot_start=4,
        state_probability_trajectory=normal_states,
        initial_state_teacher_values=(0, 0, 0, 0, 0),
        state_teacher_forcing_probability=1.0,
    )
    unified_recurrent_hidden_states(
        model,
        TOKENS,
        RecurrentDepthPlan(2, 6, iterations=3),
        controller,
        state_slot_start=4,
        state_probability_trajectory=lesioned_states,
        initial_state_teacher_values=(0, 0, 0, 0, 0),
        state_teacher_forcing_probability=1.0,
        typed_action_lesion=True,
    )

    assert len(normal_states) == len(lesioned_states) == 3
    assert any(
        not bool(mx.array_equal(normal, lesioned))
        for normal, lesioned in zip(normal_states, lesioned_states, strict=True)
    )


def test_typed_action_lesion_rejects_teacher_contamination() -> None:
    with pytest.raises(ValueError, match="cannot accompany an action teacher"):
        unified_recurrent_hidden_states(
            _model(),
            TOKENS,
            RecurrentDepthPlan(2, 6, iterations=1),
            _controller(),
            state_slot_start=4,
            typed_action_lesion=True,
            action_teacher_values=[(0,) * 8],
        )


def test_neural_answer_bridge_reads_state_without_rewriting_public_prefix() -> None:
    controller = _controller()
    candidate = mx.zeros((1, 12, 64), dtype=mx.float32)
    committed = mx.zeros((1, 12, 64), dtype=mx.float32)
    baseline = controller.attend_answer_to_state(
        candidate,
        committed,
        state_slot_start=4,
    )
    changed = controller.attend_answer_to_state(
        candidate,
        committed.at[:, 5, :].add(1.0),
        state_slot_start=4,
    )
    assert bool(mx.array_equal(baseline[:, :8, :], changed[:, :8, :]))
    assert not bool(mx.array_equal(baseline[:, 8:, :], changed[:, 8:, :]))
    assert controller.answer_gate_query.shape == (64, 1)


def test_neural_answer_bridge_reads_only_active_process_tape_entries() -> None:
    controller = _controller()
    candidate = mx.zeros((1, 12, 64), dtype=mx.float32)
    committed = mx.zeros((1, 12, 64), dtype=mx.float32)
    baseline_memory = mx.zeros((1, 10, 64), dtype=mx.float32)
    changed_memory = baseline_memory.at[:, 5:, :].add(100.0)
    mask = mx.array([[True] * 5 + [False] * 5])

    baseline = controller.attend_answer_to_state(
        candidate,
        committed,
        state_slot_start=4,
        process_memory=baseline_memory,
        process_memory_mask=mask,
    )
    masked = controller.attend_answer_to_state(
        candidate,
        committed,
        state_slot_start=4,
        process_memory=changed_memory,
        process_memory_mask=mask,
    )
    unmasked = controller.attend_answer_to_state(
        candidate,
        committed,
        state_slot_start=4,
        process_memory=changed_memory,
        process_memory_mask=mx.ones((1, 10), dtype=mx.bool_),
    )

    assert bool(mx.array_equal(baseline, masked))
    assert not bool(mx.array_equal(baseline[:, 8:, :], unmasked[:, 8:, :]))


def test_process_tape_records_every_live_action_and_state_and_can_be_lesioned() -> None:
    model = _model()
    controller = _controller()
    controller.state_transition_bias = mx.full((5, 33), -100.0)
    for slot in range(5):
        controller.state_transition_bias = (
            controller.state_transition_bias.at[slot, 0].add(200.0)
        )
    plan = RecurrentDepthPlan(2, 6, iterations=3)

    intact_final, _trajectory, intact = unified_recurrent_hidden_states(
        model,
        TOKENS,
        plan,
        controller,
        state_slot_start=4,
        initial_state_teacher_values=(0, 0, 0, 0, 0),
        state_teacher_forcing_probability=1.0,
    )
    lesioned_final, _trajectory, lesioned = unified_recurrent_hidden_states(
        model,
        TOKENS,
        plan,
        controller,
        state_slot_start=4,
        initial_state_teacher_values=(0, 0, 0, 0, 0),
        state_teacher_forcing_probability=1.0,
        process_tape_lesion=True,
    )

    entries_per_step = controller.config.action_slots + controller.config.state_slots
    assert intact.process_tape_entries == entries_per_step * plan.iterations
    assert intact.process_tape_active_entries == intact.process_tape_entries
    assert lesioned.process_tape_entries == 0
    assert lesioned.process_tape_active_entries == 0
    assert not bool(mx.array_equal(intact_final, lesioned_final))


def test_answer_digit_place_reads_the_selected_terminal_register() -> None:
    controller = _controller()
    controller.answer_role_bias = mx.full((6,), -100.0).at[2].add(200.0)
    controller.answer_place_bias = mx.zeros((3,))
    controller.answer_place_state_projection = (
        mx.zeros((64, 3)).at[0, 2].add(10.0)
    )
    candidate = mx.zeros((1, 12, 64), dtype=mx.float32)
    committed = mx.zeros((1, 12, 64), dtype=mx.float32)
    committed = committed.at[:, 5, 0].add(1.0)
    places: list[object] = []

    controller.attend_answer_to_state(
        candidate,
        committed,
        state_slot_start=4,
        place_logit_trajectory=places,
    )

    assert int(mx.argmax(places[-1][0, -1]).item()) == 2


def test_answer_digit_place_receives_exact_selected_value_width() -> None:
    controller = _controller()
    controller.answer_role_bias = mx.full((6,), -100.0).at[2].add(200.0)
    controller.answer_place_bias = mx.zeros((3,))
    controller.answer_place_width_projection = mx.array(
        ((0.0, 0.0, 10.0), (0.0, 10.0, 0.0))
    )
    candidate = mx.zeros((1, 12, 64), dtype=mx.float32)
    committed = mx.zeros((1, 12, 64), dtype=mx.float32)
    one_digit_places: list[object] = []
    two_digit_places: list[object] = []

    controller.attend_answer_to_state(
        candidate,
        committed,
        state_slot_start=4,
        state_probabilities=controller.exact_probabilities(
            (0, 3, 0, 0, 1), slots=5, cardinality=33
        ),
        place_logit_trajectory=one_digit_places,
    )
    controller.attend_answer_to_state(
        candidate,
        committed,
        state_slot_start=4,
        state_probabilities=controller.exact_probabilities(
            (0, 13, 0, 0, 1), slots=5, cardinality=33
        ),
        place_logit_trajectory=two_digit_places,
    )

    assert int(mx.argmax(one_digit_places[-1][0, -1]).item()) == 2
    assert int(mx.argmax(two_digit_places[-1][0, -1]).item()) == 1


def test_answer_digit_pointer_copies_selected_terminal_register_only() -> None:
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=64,
            correction_rank=8,
            literal_digit_token_ids=tuple(range(10, 20)),
        )
    )
    logits = mx.zeros((1, 1, 32), dtype=mx.float32)
    role_logits = mx.full((1, 1, 6), -100.0)
    role_logits = role_logits.at[:, :, 2].add(200.0)
    place_logits = mx.full((1, 1, 3), -100.0)
    place_logits = place_logits.at[:, :, 2].add(200.0)
    state = controller.exact_probabilities(
        (0, 7, 0, 0, 1),
        slots=5,
        cardinality=33,
    )
    pointed = controller.apply_answer_digit_pointer(
        logits,
        role_logits,
        place_logits,
        state,
    )
    assert int(mx.argmax(pointed[0, 0]).item()) == 17

    syntax_roles = mx.full((1, 1, 6), -100.0)
    syntax_roles = syntax_roles.at[:, :, 0].add(200.0)
    syntax_places = mx.full((1, 1, 3), -100.0)
    syntax_places = syntax_places.at[:, :, 0].add(200.0)
    unchanged = controller.apply_answer_digit_pointer(
        logits,
        syntax_roles,
        syntax_places,
        state,
    )
    assert bool(mx.allclose(mx.softmax(unchanged, axis=-1), mx.softmax(logits, axis=-1)))


def test_answer_digit_pointer_preserves_role_across_two_digit_value() -> None:
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=64,
            correction_rank=8,
            literal_digit_token_ids=tuple(range(10, 20)),
        )
    )
    logits = mx.zeros((1, 2, 32), dtype=mx.float32)
    roles = mx.full((1, 2, 6), -100.0)
    roles = roles.at[:, 0, 2].add(200.0)
    roles = roles.at[:, 1, 3].add(200.0)
    places = mx.full((1, 2, 3), -100.0)
    places = places.at[:, 0, 2].add(200.0)
    places = places.at[:, 1, 0].add(200.0)
    state = controller.exact_probabilities(
        (0, 17, 24, 0, 1),
        slots=5,
        cardinality=33,
    )

    pointed = controller.apply_answer_digit_pointer(logits, roles, places, state)

    assert mx.argmax(pointed[0], axis=-1).tolist() == [11, 17]


def test_answer_digit_pointer_closes_one_digit_value_after_first_token() -> None:
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=64,
            correction_rank=8,
            literal_digit_token_ids=tuple(range(10, 20)),
        )
    )
    logits = mx.zeros((1, 2, 32), dtype=mx.float32)
    roles = mx.full((1, 2, 6), -100.0).at[:, :, 2].add(200.0)
    places = mx.full((1, 2, 3), -100.0).at[:, :, 2].add(200.0)
    state = controller.exact_probabilities(
        (0, 7, 0, 0, 1),
        slots=5,
        cardinality=33,
    )

    pointed = controller.apply_answer_digit_pointer(logits, roles, places, state)

    assert mx.argmax(pointed[0], axis=-1).tolist() == [17, 0]


def test_answer_digit_pointer_does_not_treat_one_digit_mass_as_leading_zero() -> None:
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=64,
            correction_rank=8,
            literal_digit_token_ids=tuple(range(10, 20)),
        )
    )
    logits = mx.zeros((1, 1, 32), dtype=mx.float32)
    roles = mx.full((1, 1, 6), -100.0)
    roles = roles.at[:, :, 3].add(104.0).at[:, :, 4].add(100.0)
    places = mx.full((1, 1, 3), -100.0).at[:, :, 2].add(200.0)
    state = controller.exact_probabilities(
        (0, 20, 4, 13, 1), slots=5, cardinality=33
    )

    pointed = controller.apply_answer_digit_pointer(logits, roles, places, state)

    assert int(mx.argmax(pointed[0, 0]).item()) == 14


def test_terminal_answer_grammar_forces_only_syntax_around_neural_digits() -> None:
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=64,
            correction_rank=8,
            literal_digit_token_ids=tuple(range(10, 20)),
        )
    )
    contract = RecurrentAnswerEmissionContract(
        digit_token_ids=tuple(range(10, 20)),
        eos_token_id=39,
        family_markers=(
            ("khop", (1,)),
            ("modular", (2,)),
            ("register_trace", (3,)),
        ),
        syntax=(
            ("khop", (30,)),
            ("modular", (31,)),
            ("register_head", (32,)),
            ("register_mid_r1", (33,)),
            ("register_mid_r2", (34,)),
            ("close", (35,)),
        ),
    )
    state = controller.exact_probabilities(
        (0, 7, 0, 0, 1), slots=5, cardinality=33
    )
    logits = (
        mx.zeros((1, 1, 40), dtype=mx.float32)
        .at[:, :, 5]
        .add(10.0)
        .at[:, :, 17]
        .add(9.0)
    )

    opening = apply_terminal_answer_grammar(
        logits,
        mx.array([[2]]),
        state_slot_start=1,
        state_probabilities=state,
        contract=contract,
    )
    digit = apply_terminal_answer_grammar(
        logits,
        mx.array([[2, 31]]),
        state_slot_start=1,
        state_probabilities=state,
        contract=contract,
    )
    closing = apply_terminal_answer_grammar(
        logits,
        mx.array([[2, 31, 17]]),
        state_slot_start=1,
        state_probabilities=state,
        contract=contract,
    )

    assert int(mx.argmax(opening[0, -1]).item()) == 31
    assert int(mx.argmax(digit[0, -1]).item()) == 17
    assert int(mx.argmax(closing[0, -1]).item()) == 35


def test_terminal_typed_state_preserves_first_terminal_decode_state() -> None:
    model = _model()
    controller = _controller()
    controller.answer_role_projection = mx.zeros((64, 6)).at[:6, :].add(mx.eye(6))
    controller.answer_place_projection = mx.zeros((64, 3)).at[:3, :].add(mx.eye(3))
    decode_states: list[object] = []
    role_logits: list[object] = []
    place_logits: list[object] = []
    _final, _trajectory, _telemetry = unified_recurrent_hidden_states(
        model,
        TOKENS,
        RecurrentDepthPlan(2, 6, iterations=3),
        controller,
        state_slot_start=4,
        decode_state_trajectory=decode_states,
        answer_role_logit_trajectory=role_logits,
        answer_place_logit_trajectory=place_logits,
        initial_state_teacher_values=(0, 0, 0, 0, 0),
        state_teacher_values=(
            (1, 2, 3, 4, 1),
            (1, 2, 3, 4, 1),
            (1, 2, 3, 4, 1),
        ),
        state_teacher_forcing_probability=1.0,
    )
    mx.eval(decode_states, role_logits, place_logits)
    assert len(decode_states) == len(role_logits) == len(place_logits) == 3
    assert bool(mx.array_equal(decode_states[0], decode_states[1]))
    assert bool(mx.array_equal(decode_states[1], decode_states[2]))
    assert bool(mx.array_equal(role_logits[0], role_logits[1]))
    assert bool(mx.array_equal(role_logits[1], role_logits[2]))
    assert bool(mx.array_equal(place_logits[0], place_logits[1]))
    assert bool(mx.array_equal(place_logits[1], place_logits[2]))


def test_state_processor_reads_problem_and_state_but_not_answer_suffix() -> None:
    controller = _controller()
    evidence = mx.zeros((1, 4, 64), dtype=mx.float32)
    hidden = mx.zeros((1, 12, 64), dtype=mx.float32).at[:, 4:9, :].add(1.0)
    baseline = controller.state_transition_logits(
        evidence,
        hidden,
        state_slot_start=4,
        step=2,
    )
    changed_problem = controller.state_transition_logits(
        evidence + 2.0,
        hidden,
        state_slot_start=4,
        step=2,
    )
    changed_state = controller.state_transition_logits(
        evidence,
        hidden.at[:, 4:9, :].add(2.0),
        state_slot_start=4,
        step=2,
    )
    changed_answer_suffix = controller.state_transition_logits(
        evidence,
        hidden.at[:, 9:, :].add(100.0),
        state_slot_start=4,
        step=2,
    )
    assert baseline.shape == (1, 5, 33)
    assert not bool(mx.array_equal(baseline, changed_problem))
    assert not bool(mx.array_equal(baseline, changed_state))
    assert bool(mx.array_equal(baseline, changed_answer_suffix))


def test_exact_state_rollin_is_training_only_and_changes_typed_slots() -> None:
    controller = _controller()
    hidden = mx.ones((1, 9, 64), dtype=mx.float32)
    teacher = controller.teacher_state_transition(
        hidden,
        state_slot_start=2,
        values=(1, 2, 3, 4, 0),
        probability=1.0,
    )
    assert bool(mx.array_equal(teacher[:, :2, :], hidden[:, :2, :]))
    assert bool(mx.array_equal(teacher[:, 7:, :], hidden[:, 7:, :]))
    assert not bool(mx.array_equal(teacher[:, 2:7, :], hidden[:, 2:7, :]))
    with pytest.raises(ValueError, match="probability"):
        controller.teacher_state_transition(
            hidden,
            state_slot_start=2,
            values=(1, 2, 3, 4, 0),
            probability=1.1,
        )


def test_typed_recurrence_preserves_public_input_lane_between_passes() -> None:
    model = _model()
    controller = _controller()
    decode_states: list[object] = []
    recurrent_inputs: list[object] = []
    _final, committed, _telemetry = unified_recurrent_hidden_states(
        model,
        TOKENS,
        RecurrentDepthPlan(2, 6, iterations=3),
        controller,
        state_slot_start=4,
        decode_state_trajectory=decode_states,
        recurrent_input_trajectory=recurrent_inputs,
    )
    assert len(committed) == len(decode_states) == len(recurrent_inputs) == 3
    assert bool(mx.array_equal(recurrent_inputs[0][:, :4, :], recurrent_inputs[1][:, :4, :]))
    assert bool(mx.array_equal(recurrent_inputs[1][:, :4, :], recurrent_inputs[2][:, :4, :]))
    assert bool(mx.array_equal(recurrent_inputs[0][:, 9:, :], recurrent_inputs[1][:, 9:, :]))
    assert bool(mx.array_equal(recurrent_inputs[1][:, 9:, :], recurrent_inputs[2][:, 9:, :]))
    assert not bool(
        mx.array_equal(committed[0][:, 4:9, :], committed[1][:, 4:9, :])
    )


def test_protected_memory_survives_while_semantic_lane_keeps_moving() -> None:
    model = _model()
    controller = _controller()
    _final, trajectory, telemetry = unified_recurrent_hidden_states(
        model,
        TOKENS,
        RecurrentDepthPlan(2, 6, iterations=5, renormalize=True),
        controller,
        memory_layout=MemoryLayout(
            n_slots=TOKENS.shape[1],
            memory_slots=(0, 1),
            control_slots=(2,),
        ),
    )
    assert len(trajectory) == 5
    assert telemetry.memory_retention is not None
    assert telemetry.memory_retention["cosine"] > 0.99999
    assert telemetry.memory_retention["relative_drift"] < 1e-6
    assert telemetry.memory_retention["slots"] == 3
    assert telemetry.semantic_residuals
    assert any(value > 0.01 for value in telemetry.semantic_residuals)
    assert telemetry.memory_write_means == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_learned_halt_is_causal_but_off_until_explicitly_enabled() -> None:
    model = _model()
    controller = _controller()
    controller.halt_state_weight = mx.zeros_like(controller.halt_state_weight)
    controller.halt_motion_weight = mx.array(0.0)
    controller.halt_bias = mx.array(20.0)
    plan = RecurrentDepthPlan(2, 6, iterations=8)
    _final, full, full_telemetry = unified_recurrent_hidden_states(
        model,
        TOKENS,
        plan,
        controller,
        adaptive_halt=False,
    )
    _final, halted, halted_telemetry = unified_recurrent_hidden_states(
        model,
        TOKENS,
        plan,
        controller,
        adaptive_halt=True,
    )
    assert len(full) == 8
    assert full_telemetry.halted is False
    assert len(halted) == controller.config.minimum_iterations
    assert halted_telemetry.halted is True
    assert halted_telemetry.halt_reason == "learned_threshold"


def test_trained_correction_changes_the_real_answer_path() -> None:
    model = _model()
    controller = _controller()
    baseline, _ = unified_recurrent_logits(
        model,
        TOKENS,
        RecurrentDepthPlan(2, 6, iterations=3),
        controller,
    )
    controller.correction_b = mx.ones_like(controller.correction_b) * 0.01
    changed, telemetry = unified_recurrent_logits(
        model,
        TOKENS,
        RecurrentDepthPlan(2, 6, iterations=3),
        controller,
    )
    assert not bool(mx.allclose(baseline, changed, atol=1e-5))
    assert telemetry.receipt()["teacher_available"] is False
    assert telemetry.receipt()["solver_available"] is False


def test_cached_unified_decode_preserves_the_selected_token() -> None:
    from core.learning.intrinsic_recurrence import make_recurrent_caches

    model = _model()
    controller = _controller()
    plan = RecurrentDepthPlan(2, 6, iterations=3, renormalize=True)
    reference, _ = unified_recurrent_logits(model, TOKENS, plan, controller)
    caches = make_recurrent_caches(model, plan)
    stepwise = None
    for index in range(TOKENS.shape[1]):
        stepwise, _ = unified_recurrent_logits(
            model,
            TOKENS[:, index : index + 1],
            plan,
            controller,
            caches=caches,
        )

    assert stepwise is not None
    assert int(mx.argmax(reference[0, -1])) == int(mx.argmax(stepwise[0, -1]))


def test_cached_unified_decode_preserves_a_generated_sequence() -> None:
    from core.learning.intrinsic_recurrence import make_recurrent_caches

    model = _model()
    controller = _controller()
    plan = RecurrentDepthPlan(2, 6, iterations=3, renormalize=True)

    def decode(*, incremental: bool) -> tuple[int, ...]:
        prefix = TOKENS
        next_tokens = TOKENS
        caches = make_recurrent_caches(model, plan) if incremental else None
        generated: list[int] = []
        for _index in range(4):
            logits, _ = unified_recurrent_logits(
                model,
                next_tokens if incremental else prefix,
                plan,
                controller,
                caches=caches,
            )
            token = int(mx.argmax(logits[0, -1]))
            generated.append(token)
            next_tokens = mx.array([[token]], dtype=TOKENS.dtype)
            prefix = mx.concatenate([prefix, next_tokens], axis=1)
        return tuple(generated)

    assert decode(incremental=True) == decode(incremental=False)


def test_cached_unified_decode_refuses_stateful_or_mismatched_modes() -> None:
    from core.learning.intrinsic_recurrence import make_recurrent_caches

    model = _model()
    controller = _controller()
    plan = RecurrentDepthPlan(2, 6, iterations=3)
    caches = make_recurrent_caches(model, plan)

    with pytest.raises(ValueError, match="untyped fixed-depth"):
        unified_recurrent_logits(
            model,
            TOKENS,
            plan,
            controller,
            caches=caches,
            adaptive_halt=True,
        )
    wrong = make_recurrent_caches(model, RecurrentDepthPlan(2, 6, iterations=2))
    with pytest.raises(ValueError, match="cache iteration count"):
        unified_recurrent_logits(model, TOKENS, plan, controller, caches=wrong)
    malformed = make_recurrent_caches(model, plan)
    malformed["window"][0].pop()
    with pytest.raises(ValueError, match="cache layer topology"):
        unified_recurrent_logits(model, TOKENS, plan, controller, caches=malformed)


def test_controller_cannot_rewrite_the_t1_semantic_anchor() -> None:
    model = _model()
    controller = _controller()
    plan = RecurrentDepthPlan(2, 6, iterations=1)
    baseline, _ = unified_recurrent_logits(model, TOKENS, plan, controller)
    controller.correction_b = mx.ones_like(controller.correction_b) * 10.0
    controller.depth_scale = mx.ones_like(controller.depth_scale) * 10.0
    changed, _ = unified_recurrent_logits(model, TOKENS, plan, controller)
    assert bool(mx.array_equal(baseline, changed))


def test_bounded_transport_preserves_t1_and_controls_deep_reentry() -> None:
    controller = _controller()
    previous = mx.ones((1, 4, 64))
    candidate = mx.ones((1, 4, 64)) * 100.0
    first, first_gate = controller.transport(previous, candidate, 0)
    deep, deep_gate = controller.transport(previous, candidate, 8)
    assert bool(mx.array_equal(first, candidate))
    assert float(first_gate.item()) == 1.0
    assert 0.0 < float(mx.mean(deep_gate).item()) < 1.0
    assert float(mx.mean(mx.abs(deep))) < float(mx.mean(mx.abs(candidate)))
    assert float(mx.mean(mx.abs(deep))) >= float(mx.mean(mx.abs(previous)))
    assert float(controller.transport_gate(8).item()) < float(
        controller.transport_gate(1).item()
    )
    exponent = float(controller.transport_decay_exponent().item())
    assert 0.5 < exponent < 1.0


def test_adaptive_transport_starts_at_depth_prior_and_discriminates_state() -> None:
    controller = _controller()
    previous = mx.ones((1, 3, 64), dtype=mx.float32)
    candidate = previous + 0.25
    prior = controller.transport_gate(2)
    initial = controller.transport_gate(2, previous, candidate)
    assert initial.shape == (1, 3, 1)
    assert bool(mx.allclose(initial, prior))

    controller.transport_state_weight = mx.ones((64,), dtype=mx.float32)
    accepted = controller.transport_gate(2, previous, candidate)
    rejected = controller.transport_gate(2, -previous, -candidate)
    assert bool(mx.all(accepted > initial))
    assert bool(mx.all(rejected < initial))
    assert bool(mx.all(accepted > 0.0)) and bool(mx.all(accepted < 1.0))


def test_invalid_unified_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="correction rank"):
        UnifiedRecurrenceConfig(hidden_size=4, correction_rank=8)
    with pytest.raises(ValueError, match="state slot"):
        UnifiedRecurrenceConfig(hidden_size=8, correction_rank=4, state_slots=4)
    with pytest.raises(ValueError, match="minimum iterations"):
        unified_recurrent_hidden_states(
            _model(),
            TOKENS,
            RecurrentDepthPlan(2, 6, iterations=1),
            _controller(),
            adaptive_halt=True,
        )
    with pytest.raises(ValueError, match="token positions"):
        unified_recurrent_hidden_states(
            _model(),
            TOKENS,
            RecurrentDepthPlan(2, 6, iterations=3),
            _controller(),
            memory_layout=MemoryLayout(n_slots=5, memory_slots=(0,)),
        )
    with pytest.raises(ValueError, match="supplied together"):
        _controller().transport_gate(1, mx.zeros((1, 1, 64)))
    with pytest.raises(ValueError, match="shapes differ"):
        _controller().transport_gate(
            1,
            mx.zeros((1, 1, 64)),
            mx.zeros((1, 2, 64)),
        )
