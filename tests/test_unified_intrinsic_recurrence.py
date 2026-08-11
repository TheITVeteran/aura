"""Contracts for the single-path intrinsic recurrent controller."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.learning.intrinsic_recurrence import RecurrentDepthPlan  # noqa: E402
from core.learning.protected_memory import MemoryLayout  # noqa: E402
from core.learning.unified_intrinsic_recurrence import (  # noqa: E402
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
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
        == "token_conditioned_norm_bounded_cross_attention_before_frozen_coda"
    )
    assert receipt["terminal_decode_semantics"] == "first_terminal_state_preserved"


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


def test_terminal_typed_state_preserves_first_terminal_decode_state() -> None:
    model = _model()
    controller = _controller()
    decode_states: list[object] = []
    _final, _trajectory, _telemetry = unified_recurrent_hidden_states(
        model,
        TOKENS,
        RecurrentDepthPlan(2, 6, iterations=3),
        controller,
        state_slot_start=4,
        decode_state_trajectory=decode_states,
        initial_state_teacher_values=(0, 0, 0, 0, 0),
        state_teacher_values=(
            (1, 2, 3, 4, 1),
            (1, 2, 3, 4, 1),
            (1, 2, 3, 4, 1),
        ),
        state_teacher_forcing_probability=1.0,
    )
    mx.eval(decode_states)
    assert len(decode_states) == 3
    assert bool(mx.array_equal(decode_states[0], decode_states[1]))
    assert bool(mx.array_equal(decode_states[1], decode_states[2]))


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
