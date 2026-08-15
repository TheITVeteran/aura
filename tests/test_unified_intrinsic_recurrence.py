"""Contracts for the single-path intrinsic recurrent controller."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.learning.intrinsic_recurrence import RecurrentDepthPlan  # noqa: E402
from core.learning.protected_memory import MemoryLayout  # noqa: E402
from core.learning.recurrent_action_schema import (  # noqa: E402
    OP_FRONTIER_AUDIT,
    OP_FRONTIER_TRAVERSE,
)
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


def test_action_workspace_attaches_as_exact_noop_then_changes_logits() -> None:
    baseline = _controller()
    altered = _controller()
    evidence = mx.random.normal((1, 9, 64), key=mx.random.key(91))
    hidden = mx.random.normal((1, 12, 64), key=mx.random.key(92))

    expected = baseline.action_logits(
        evidence,
        hidden,
        state_slot_start=4,
        step=3,
    )
    altered.action_workspace_cross_query = mx.full_like(
        altered.action_workspace_cross_query,
        17.0,
    )
    no_op = altered.action_logits(
        evidence,
        hidden,
        state_slot_start=4,
        step=3,
    )
    assert bool(mx.array_equal(expected, no_op))

    altered.action_workspace_output = mx.full_like(
        altered.action_workspace_output,
        0.01,
    )
    active = altered.action_logits(
        evidence,
        hidden,
        state_slot_start=4,
        step=3,
    )
    assert not bool(mx.array_equal(expected, active))
    assert (
        altered.receipt()["action_processor"]
        == "public_evidence_bounded_autoregressive_typed_action_workspace"
    )


def test_action_workspace_can_be_captured_for_training_only_readout_fit() -> None:
    controller = _controller()
    evidence = mx.random.normal((1, 9, 64), key=mx.random.key(197))
    hidden = mx.random.normal((1, 12, 64), key=mx.random.key(198))
    trajectory: list = []
    controller.action_logits(
        evidence,
        hidden,
        state_slot_start=4,
        step=3,
        action_workspace_trajectory=trajectory,
    )
    assert len(trajectory) == 1
    assert trajectory[0].shape[:2] == (1, controller.config.action_slots)


def test_public_action_signature_preserves_literals_state_depth_and_family() -> None:
    patterns = tuple(
        (opcode, (100 + opcode - OP_FRONTIER_TRAVERSE,))
        for opcode in range(OP_FRONTIER_TRAVERSE, OP_FRONTIER_AUDIT + 1)
    )
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=64,
            correction_rank=8,
            literal_digit_token_ids=tuple(range(10, 20)),
            frontier_family_token_patterns=patterns,
        )
    )
    tokens = mx.array(
        [
            [100, 12, 19, 99, 14],
            [101, 14, 99, 12, 19],
        ]
    )
    state = mx.zeros(
        (2, controller.config.state_slots, controller.config.state_cardinality),
        dtype=mx.float32,
    ).at[:, :, 0].add(1.0)

    signature = controller._public_action_signature(
        tokens,
        state,
        step=3,
        width=64,
    )
    assert signature is not None
    assert signature.shape == (2, controller.config.action_slots, 64)
    assert not bool(mx.array_equal(signature[0], signature[1]))

    changed_state = state.at[0, 0, 0].add(-1.0).at[0, 0, 3].add(1.0)
    state_signature = controller._public_action_signature(
        tokens,
        changed_state,
        step=3,
        width=64,
    )
    next_depth_signature = controller._public_action_signature(
        tokens,
        state,
        step=4,
        width=64,
    )
    assert state_signature is not None
    assert next_depth_signature is not None
    assert not bool(mx.array_equal(signature[0], state_signature[0]))
    assert not bool(mx.array_equal(signature, next_depth_signature))


def test_semantic_controller_initial_state_uses_registered_public_topology() -> None:
    patterns = tuple(
        (opcode, (100 + opcode - OP_FRONTIER_TRAVERSE,))
        for opcode in range(OP_FRONTIER_TRAVERSE, OP_FRONTIER_AUDIT + 1)
    )
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=64,
            correction_rank=8,
            state_slots=11,
            literal_digit_token_ids=tuple(range(10, 20)),
            frontier_family_token_patterns=patterns,
        )
    )
    token_ids = mx.array([[100, 12, 19]])
    logits = controller.initial_state_logits(
        mx.zeros((1, 3, 64), dtype=mx.float32),
        token_ids,
    )
    mx.eval(logits)

    assert logits.shape == (1, 11, controller.config.state_cardinality)
    assert tuple(int(value) for value in mx.argmax(logits[0], axis=-1).tolist()) == (
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def test_action_kernel_capture_matches_runtime_public_signature() -> None:
    patterns = tuple(
        (opcode, (100 + opcode - OP_FRONTIER_TRAVERSE,))
        for opcode in range(OP_FRONTIER_TRAVERSE, OP_FRONTIER_AUDIT + 1)
    )
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=64,
            correction_rank=8,
            literal_digit_token_ids=tuple(range(10, 20)),
            frontier_family_token_patterns=patterns,
        )
    )
    evidence = mx.random.normal((1, 5, 64), key=mx.random.key(19_701))
    hidden = mx.random.normal((1, 12, 64), key=mx.random.key(19_702))
    tokens = mx.array([[100, 12, 19, 99, 14]])
    state = mx.zeros(
        (1, controller.config.state_slots, controller.config.state_cardinality),
        dtype=mx.float32,
    ).at[:, :, 0].add(1.0)
    captured: list = []

    controller.action_logits(
        evidence,
        hidden,
        state_slot_start=5,
        step=3,
        token_ids=tokens,
        state_probabilities=state,
        action_kernel_feature_trajectory=captured,
    )
    expected = controller._public_action_signature(
        tokens,
        state,
        step=3,
        width=int(controller.action_family_output.shape[2]),
    )
    assert expected is not None
    assert len(captured) == 1
    assert bool(mx.array_equal(captured[0], expected))


def test_action_workspace_causally_reads_complete_prior_process_memory() -> None:
    controller = _controller()
    controller.action_workspace_output = mx.random.normal(
        controller.action_workspace_output.shape,
        key=mx.random.key(901),
    )
    evidence = mx.random.normal((1, 9, 64), key=mx.random.key(902))
    hidden = mx.random.normal((1, 12, 64), key=mx.random.key(903))
    left = mx.zeros((1, 13, 64), dtype=mx.float32)
    right = left.at[:, -1, :].add(5.0)

    left_logits = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=4,
        step=3,
        process_memory=left,
    )
    right_logits = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=4,
        step=3,
        process_memory=right,
    )
    mx.eval(left_logits, right_logits)

    assert not bool(mx.array_equal(left_logits, right_logits))
    assert controller.receipt()["action_process_memory"] == {
        "source": "complete_prior_typed_process_tape",
        "future_steps_visible": False,
        "private_answer_exposed": False,
    }
    with pytest.raises(ValueError, match="action process memory"):
        controller.action_logits(
            evidence,
            hidden,
            state_slot_start=4,
            step=3,
            process_memory=mx.zeros((2, 13, 64)),
        )


def test_action_literal_binding_is_exact_noop_then_tracks_public_numbers() -> None:
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=64,
            correction_rank=8,
            literal_digit_token_ids=tuple(range(10, 20)),
        )
    )
    evidence = mx.random.normal((1, 2, 64), key=mx.random.key(904))
    hidden = mx.random.normal((1, 9, 64), key=mx.random.key(905))
    tokens = mx.array([[12, 19]])
    baseline = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=2,
        step=0,
        token_ids=tokens,
        action_literal_binding_lesion=True,
    )
    no_op = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=2,
        step=0,
        token_ids=tokens,
    )
    assert bool(mx.array_equal(baseline, no_op))

    controller.action_literal_binding_output = (
        controller.action_literal_binding_output.at[1, 0].add(1.0)
    )
    active = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=2,
        step=0,
        token_ids=tokens,
    )
    changed_literal = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=2,
        step=0,
        token_ids=mx.array([[13, 18]]),
    )
    assert not bool(mx.array_equal(active, baseline))
    assert not bool(mx.array_equal(active[:, 1, :], changed_literal[:, 1, :]))
    receipt = controller.receipt()["action_literal_binding"]
    assert receipt["source"] == "ordered_public_prompt_literals"
    assert receipt["private_transition_program_visible"] is False


def test_action_literal_binding_gradient_reaches_zero_output_attachment() -> None:
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=64,
            correction_rank=8,
            literal_digit_token_ids=tuple(range(10, 20)),
        )
    )
    evidence = mx.random.normal((1, 2, 64), key=mx.random.key(906))
    hidden = mx.random.normal((1, 9, 64), key=mx.random.key(907))

    def objective(candidate: UnifiedRecurrentController):
        logits = candidate.action_logits(
            evidence,
            hidden,
            state_slot_start=2,
            step=0,
            token_ids=mx.array([[12, 19]]),
        )
        return -mx.mean(nn.log_softmax(logits, axis=-1)[:, 1, 2])

    loss, gradients = nn.value_and_grad(controller, objective)(controller)
    mx.eval(loss, gradients)
    assert float(mx.max(mx.abs(gradients["action_literal_binding_output"]))) > 0.0


def test_later_literal_pointer_is_conditioned_on_earlier_action_fields() -> None:
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=64,
            correction_rank=8,
            literal_digit_token_ids=tuple(range(10, 20)),
        )
    )
    controller.action_literal_binding_output = (
        controller.action_literal_binding_output.at[2, 0].add(1.0)
    )
    evidence = mx.random.normal((1, 7, 64), key=mx.random.key(90_811))
    hidden = mx.random.normal((1, 9, 64), key=mx.random.key(90_812))
    tokens = mx.array([[12, 99, 19, 99, 14, 99, 17]])
    left = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=2,
        step=0,
        token_ids=tokens,
        teacher_values=(1, 2, 3, 4, 5, 6, 7, 0),
        teacher_forcing_probability=1.0,
    )
    right = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=2,
        step=0,
        token_ids=tokens,
        teacher_values=(8, 2, 3, 4, 5, 6, 7, 0),
        teacher_forcing_probability=1.0,
    )
    mx.eval(left, right)
    assert bool(mx.array_equal(left[:, 0, :], right[:, 0, :]))
    assert not bool(mx.array_equal(left[:, 2, :], right[:, 2, :]))


def test_action_literal_binding_selects_transforms_by_public_family() -> None:
    patterns = tuple(
        (opcode, (100 + opcode - OP_FRONTIER_TRAVERSE,))
        for opcode in range(OP_FRONTIER_TRAVERSE, OP_FRONTIER_AUDIT + 1)
    )
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=64,
            correction_rank=8,
            literal_digit_token_ids=tuple(range(10, 20)),
            frontier_family_token_patterns=patterns,
        )
    )
    evidence = mx.repeat(
        mx.random.normal((1, 3, 64), key=mx.random.key(908)),
        2,
        axis=0,
    )
    hidden = mx.repeat(
        mx.random.normal((1, 10, 64), key=mx.random.key(909)),
        2,
        axis=0,
    )
    tokens = mx.array([[100, 12, 19], [101, 12, 19]])
    baseline = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=3,
        step=0,
        token_ids=tokens,
    )
    controller.action_literal_binding_family_output = (
        controller.action_literal_binding_family_output.at[0, 1, 0].add(1.0)
    )
    active = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=3,
        step=0,
        token_ids=tokens,
    )
    mx.eval(baseline, active)

    assert not bool(mx.array_equal(active[0, 1], baseline[0, 1]))
    assert bool(mx.array_equal(active[1], baseline[1]))


def test_action_literal_binding_family_gradient_is_route_isolated() -> None:
    patterns = tuple(
        (opcode, (100 + opcode - OP_FRONTIER_TRAVERSE,))
        for opcode in range(OP_FRONTIER_TRAVERSE, OP_FRONTIER_AUDIT + 1)
    )
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=64,
            correction_rank=8,
            literal_digit_token_ids=tuple(range(10, 20)),
            frontier_family_token_patterns=patterns,
        )
    )
    evidence = mx.random.normal((1, 3, 64), key=mx.random.key(910))
    hidden = mx.random.normal((1, 10, 64), key=mx.random.key(911))

    def objective(candidate: UnifiedRecurrentController):
        logits = candidate.action_logits(
            evidence,
            hidden,
            state_slot_start=3,
            step=0,
            token_ids=mx.array([[100, 12, 19]]),
        )
        return -mx.mean(nn.log_softmax(logits, axis=-1)[:, 1, 2])

    loss, gradients = nn.value_and_grad(controller, objective)(controller)
    family_gradient = gradients["action_literal_binding_family_output"]
    mx.eval(loss, family_gradient)

    assert float(mx.max(mx.abs(family_gradient[0]))) > 0.0
    assert bool(mx.all(family_gradient[1:] == 0))


def test_public_family_experts_attach_as_exact_noop_and_route_independently() -> None:
    patterns = tuple(
        (opcode, (100 + opcode - OP_FRONTIER_TRAVERSE,))
        for opcode in range(OP_FRONTIER_TRAVERSE, OP_FRONTIER_AUDIT + 1)
    )
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=64,
            correction_rank=8,
            frontier_family_token_patterns=patterns,
        )
    )
    evidence = mx.repeat(
        mx.random.normal((1, 9, 64), key=mx.random.key(191)),
        2,
        axis=0,
    )
    hidden = mx.repeat(
        mx.random.normal((1, 12, 64), key=mx.random.key(192)),
        2,
        axis=0,
    )
    tokens = mx.array([[100, 1], [101, 1]])
    baseline = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=4,
        step=3,
        token_ids=tokens,
        family_action_lesion=True,
    )
    no_op = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=4,
        step=3,
        token_ids=tokens,
    )
    assert bool(mx.array_equal(baseline, no_op))
    assert bool(mx.all(controller.action_family_output == 0))

    controller.action_family_output = mx.random.normal(
        controller.action_family_output.shape,
        key=mx.random.key(193),
    )
    active = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=4,
        step=3,
        token_ids=tokens,
    )
    lesioned = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=4,
        step=3,
        token_ids=tokens,
        family_action_lesion=True,
    )
    assert not bool(mx.array_equal(active, lesioned))
    assert not bool(mx.array_equal(active[0], active[1]))


def test_public_family_expert_gradient_isolated_to_selected_route() -> None:
    patterns = tuple(
        (opcode, (100 + opcode - OP_FRONTIER_TRAVERSE,))
        for opcode in range(OP_FRONTIER_TRAVERSE, OP_FRONTIER_AUDIT + 1)
    )
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=64,
            correction_rank=8,
            frontier_family_token_patterns=patterns,
        )
    )
    evidence = mx.random.normal((1, 9, 64), key=mx.random.key(194))
    hidden = mx.random.normal((1, 12, 64), key=mx.random.key(195))

    def objective(candidate: UnifiedRecurrentController):
        logits = candidate.action_logits(
            evidence,
            hidden,
            state_slot_start=4,
            step=3,
            token_ids=mx.array([[100, 1]]),
        )
        return -mx.mean(nn.log_softmax(logits, axis=-1)[:, :, 7])

    loss, gradients = nn.value_and_grad(controller, objective)(controller)
    mx.eval(loss, gradients)
    expert_gradients = gradients["action_family_output"]
    bias_gradients = gradients["action_family_bias"]
    assert float(mx.max(mx.abs(expert_gradients[0]))) > 0.0
    assert float(mx.max(mx.abs(expert_gradients[1:]))) == 0.0
    assert float(mx.max(mx.abs(bias_gradients[0]))) > 0.0
    assert float(mx.max(mx.abs(bias_gradients[1:]))) == 0.0


def test_public_family_kernel_is_noop_until_training_prototypes_attach() -> None:
    patterns = tuple(
        (opcode, (100 + opcode - OP_FRONTIER_TRAVERSE,))
        for opcode in range(OP_FRONTIER_TRAVERSE, OP_FRONTIER_AUDIT + 1)
    )
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=64,
            correction_rank=8,
            frontier_family_token_patterns=patterns,
        )
    )
    evidence = mx.random.normal((1, 9, 64), key=mx.random.key(199))
    hidden = mx.random.normal((1, 12, 64), key=mx.random.key(200))
    tokens = mx.array([[100, 1]])
    baseline = controller.action_logits(
        evidence, hidden, state_slot_start=4, step=3, token_ids=tokens
    )
    controller.action_family_kernel_inv_scale = mx.ones_like(
        controller.action_family_kernel_inv_scale
    )
    controller.action_family_kernel_mask = controller.action_family_kernel_mask.at[
        0, :, 0
    ].add(1.0)
    controller.action_family_kernel_gamma = controller.action_family_kernel_gamma.at[
        0, :
    ].add(1.0)
    controller.action_family_kernel_coefficients = (
        controller.action_family_kernel_coefficients.at[0, :, 0, 7].add(5.0)
    )
    active = controller.action_logits(
        evidence, hidden, state_slot_start=4, step=3, token_ids=tokens
    )
    assert not bool(mx.array_equal(baseline, active))


def test_causal_action_decoder_is_exact_noop_until_its_output_is_trained() -> None:
    controller = _controller()
    evidence = mx.random.normal((1, 9, 64), key=mx.random.key(95))
    hidden = mx.random.normal((1, 12, 64), key=mx.random.key(96))

    intact = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=4,
        step=2,
    )
    lesioned = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=4,
        step=2,
        causal_action_lesion=True,
    )

    assert bool(mx.array_equal(intact, lesioned))
    assert bool(mx.all(controller.action_causal_output == 0))


def test_causal_action_prefix_changes_only_later_fields() -> None:
    controller = _controller()
    controller.action_causal_output = mx.random.normal(
        controller.action_causal_output.shape,
        key=mx.random.key(97),
    )
    evidence = mx.random.normal((1, 9, 64), key=mx.random.key(98))
    hidden = mx.random.normal((1, 12, 64), key=mx.random.key(99))
    left = (1, 2, 3, 4, 5, 6, 7, 0)
    changed_first = (9, 2, 3, 4, 5, 6, 7, 0)
    changed_future = (1, 2, 3, 4, 5, 6, 7, 1)

    baseline = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=4,
        step=2,
        teacher_values=left,
        teacher_forcing_probability=1.0,
    )
    earlier_changed = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=4,
        step=2,
        teacher_values=changed_first,
        teacher_forcing_probability=1.0,
    )
    future_changed = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=4,
        step=2,
        teacher_values=changed_future,
        teacher_forcing_probability=1.0,
    )

    assert bool(mx.array_equal(baseline[:, 0], earlier_changed[:, 0]))
    assert not bool(mx.array_equal(baseline[:, 1:], earlier_changed[:, 1:]))
    assert bool(mx.array_equal(baseline, future_changed))


def test_causal_action_teacher_matches_autonomous_prefix_when_predictions_match() -> None:
    controller = _controller()
    forced = (1, 2, 3, 4, 5, 6, 7, 0)
    controller.action_bias = mx.full_like(controller.action_bias, -100.0)
    for slot, value in enumerate(forced):
        controller.action_bias = controller.action_bias.at[slot, value].add(200.0)
    controller.action_causal_output = mx.random.normal(
        controller.action_causal_output.shape,
        key=mx.random.key(100),
    )
    evidence = mx.random.normal((1, 9, 64), key=mx.random.key(101))
    hidden = mx.random.normal((1, 12, 64), key=mx.random.key(102))

    autonomous = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=4,
        step=2,
    )
    taught = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=4,
        step=2,
        teacher_values=forced,
        teacher_forcing_probability=1.0,
    )

    assert bool(mx.array_equal(autonomous, taught))


def test_previous_action_feedback_is_causal_and_lesionable() -> None:
    controller = _controller()
    controller.action_causal_output = mx.random.normal(
        controller.action_causal_output.shape,
        key=mx.random.key(103),
    )
    evidence = mx.random.normal((1, 9, 64), key=mx.random.key(104))
    hidden = mx.random.normal((1, 12, 64), key=mx.random.key(105))
    left = controller.exact_probabilities(
        (1, 2, 3, 4, 5, 6, 7, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )
    right = controller.exact_probabilities(
        (9, 8, 7, 6, 5, 4, 3, 1),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )

    left_logits = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=4,
        step=2,
        prior_action_probabilities=left,
    )
    right_logits = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=4,
        step=2,
        prior_action_probabilities=right,
    )
    left_lesioned = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=4,
        step=2,
        prior_action_probabilities=left,
        action_feedback_lesion=True,
    )
    right_lesioned = controller.action_logits(
        evidence,
        hidden,
        state_slot_start=4,
        step=2,
        prior_action_probabilities=right,
        action_feedback_lesion=True,
    )

    assert not bool(mx.array_equal(left_logits, right_logits))
    assert bool(mx.array_equal(left_lesioned, right_lesioned))


def test_action_loss_reaches_workspace_without_needing_nonzero_attachment() -> None:
    controller = _controller()
    evidence = mx.random.normal((1, 9, 64), key=mx.random.key(93))
    hidden = mx.random.normal((1, 12, 64), key=mx.random.key(94))

    def objective(candidate: UnifiedRecurrentController):
        logits = candidate.action_logits(
            evidence,
            hidden,
            state_slot_start=4,
            step=3,
        )
        return -mx.mean(nn.log_softmax(logits, axis=-1)[:, :, 7])

    loss, gradients = nn.value_and_grad(controller, objective)(controller)
    mx.eval(loss, gradients)
    assert float(mx.max(mx.abs(gradients["action_workspace_output"]))) > 0.0
    assert float(mx.max(mx.abs(gradients["action_causal_output"]))) > 0.0
    assert float(mx.max(mx.abs(gradients["initial_state_output"]))) == 0.0


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
    assert receipt["state_processor"] == (
        "separate_initial_parser_and_recurrent_transition"
    )
    assert (
        receipt["state_problem_evidence"]
        == "frozen_deep_prefix_no_decoder_suffix"
    )
    assert receipt["transformer_answer_passes_per_state"] == 1
    assert (
        receipt["state_to_answer_bridge"]
        == "causally_contextualized_ordered_typed_masked_action_state_"
        "process_tape_attention_over_frozen_readout"
    )
    assert receipt["process_tape"] == {
        "schema": "aura.unified_intrinsic.process_tape.v4",
        "ordering": "bounded_sinusoidal_step_and_transition_role",
        "reader": "two_independent_rank_expanded_causal_prefix_blocks",
        "reader_rank": 32,
        "contents": [
            "pre_state",
            "typed_action",
            "post_state",
            "state_delta",
        ],
        "terminal_stutter_entries_masked": True,
    }
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


def test_public_action_program_executes_without_becoming_teacher_forcing() -> None:
    model = _model()
    controller = _controller()
    public_actions = (
        (0, 7, 32, 32, 32, 32, 32, 0),
        (1, 5, 13, 32, 32, 32, 32, 1),
    )
    intact_states: list[object] = []
    lesioned_states: list[object] = []
    _final, _trajectory, telemetry = unified_recurrent_hidden_states(
        model,
        TOKENS,
        RecurrentDepthPlan(2, 6, iterations=2),
        controller,
        state_slot_start=4,
        state_probability_trajectory=intact_states,
        initial_state_teacher_values=(0, 0, 0, 0, 0),
        public_action_values=public_actions,
        state_teacher_forcing_probability=1.0,
    )
    unified_recurrent_hidden_states(
        model,
        TOKENS,
        RecurrentDepthPlan(2, 6, iterations=2),
        controller,
        state_slot_start=4,
        state_probability_trajectory=lesioned_states,
        initial_state_teacher_values=(0, 0, 0, 0, 0),
        public_action_values=public_actions,
        state_teacher_forcing_probability=1.0,
        typed_action_lesion=True,
    )

    intact = tuple(
        int(value) for value in mx.argmax(intact_states[-1][0], axis=-1).tolist()
    )
    lesioned = tuple(
        int(value) for value in mx.argmax(lesioned_states[-1][0], axis=-1).tolist()
    )
    assert intact == (2, 12, 0, 0, 1)
    assert lesioned != intact
    assert telemetry.receipt()["teacher_available"] is False


def test_public_actions_cannot_alias_private_action_teacher() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        unified_recurrent_hidden_states(
            _model(),
            TOKENS,
            RecurrentDepthPlan(2, 6, iterations=1),
            _controller(),
            state_slot_start=4,
            public_action_values=((0, 1, 32, 32, 32, 32, 32, 1),),
            action_teacher_values=((0, 1, 32, 32, 32, 32, 32, 1),),
        )


def test_microcode_lesion_forces_the_learned_transition_surface() -> None:
    controller = _controller()
    problem = mx.random.normal((1, 7, 64), key=mx.random.key(114))
    hidden = mx.random.normal((1, 10, 64), key=mx.random.key(115))
    state = controller.exact_probabilities(
        (0, 0, 0, 0, 0),
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    action = controller.exact_probabilities(
        (0, 9, 32, 32, 32, 32, 32, 1),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )
    action_state = controller.commit_action_probabilities(action)
    exact = controller.state_transition_logits(
        problem,
        hidden,
        state_slot_start=3,
        step=0,
        action_state=action_state,
        state_probabilities=state,
        action_probabilities=action,
    )
    learned = controller.state_transition_logits(
        problem,
        hidden,
        state_slot_start=3,
        step=0,
        action_state=action_state,
        state_probabilities=state,
        action_probabilities=action,
        microcode_lesion=True,
    )
    mx.eval(exact, learned)
    assert tuple(int(value) for value in mx.argmax(exact[0], axis=-1).tolist()) == (
        1,
        9,
        0,
        0,
        1,
    )
    assert not bool(mx.array_equal(exact, learned))


def test_learned_transition_reads_prior_actions_in_causal_order() -> None:
    controller = _controller()
    problem = mx.random.normal((1, 7, 64), key=mx.random.key(116))
    hidden = mx.random.normal((1, 10, 64), key=mx.random.key(117))
    state = controller.exact_probabilities(
        (0, 0, 0, 0, 0),
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    first = controller.exact_probabilities(
        (0, 7, 32, 32, 32, 32, 32, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )
    second = controller.exact_probabilities(
        (1, 5, 13, 32, 32, 32, 32, 1),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )
    current = controller.exact_probabilities(
        (4, 3, 2, 32, 32, 32, 32, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )
    current_state = controller.commit_action_probabilities(current)
    forward = controller.state_transition_logits(
        problem,
        hidden,
        state_slot_start=3,
        step=2,
        action_state=current_state,
        state_probabilities=state,
        action_probabilities=current,
        action_probability_history=(first, second, current),
        microcode_lesion=True,
    )
    reversed_history = controller.state_transition_logits(
        problem,
        hidden,
        state_slot_start=3,
        step=2,
        action_state=current_state,
        state_probabilities=state,
        action_probabilities=current,
        action_probability_history=(second, first, current),
        microcode_lesion=True,
    )

    mx.eval(forward, reversed_history)
    assert not bool(mx.allclose(forward, reversed_history))


def test_typed_transition_memory_attaches_as_exact_noop() -> None:
    controller = _controller()
    action = controller.exact_probabilities(
        (1, 7, 5, 32, 32, 32, 32, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )

    memory = controller._typed_transition_memory((action,))

    mx.eval(memory)
    assert tuple(memory.shape) == (
        1,
        controller.config.state_slots,
        controller.config.correction_rank,
    )
    assert bool(mx.all(memory == 0))


def test_typed_transition_memory_preserves_field_identity_and_order() -> None:
    controller = _controller()
    controller.state_action_projection = mx.zeros_like(
        controller.state_action_projection
    )
    controller.transition_memory_output = mx.random.normal(
        controller.transition_memory_output.shape,
        key=mx.random.key(118),
    )
    first = controller.exact_probabilities(
        (1, 7, 5, 32, 32, 32, 32, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )
    swapped_fields = controller.exact_probabilities(
        (1, 5, 7, 32, 32, 32, 32, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )
    second = controller.exact_probabilities(
        (3, 11, 13, 32, 32, 32, 32, 1),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )

    forward = controller._typed_transition_memory((first, second))
    reversed_history = controller._typed_transition_memory((second, first))
    field_swapped = controller._typed_transition_memory((swapped_fields, second))

    mx.eval(forward, reversed_history, field_swapped)
    assert not bool(mx.allclose(forward, reversed_history))
    assert not bool(mx.allclose(forward, field_swapped))


def test_public_transition_tape_reader_is_zero_attached() -> None:
    controller = _controller()
    state = controller.exact_probabilities(
        (0, 3, 5, 7, 0),
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    action = controller.exact_probabilities(
        (8, 1, 2, 3, 4, 5, 6, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )

    read = controller._typed_transition_tape_read(
        (action,),
        state_probabilities=state,
        action_probabilities=action,
    )

    mx.eval(read)
    assert read.shape == (
        1,
        controller.config.state_slots,
        controller.config.correction_rank,
    )
    assert bool(mx.all(read == 0))


def test_public_transition_tape_reader_retains_order_and_query_context() -> None:
    controller = _controller()
    controller.transition_memory_output = mx.zeros_like(
        controller.transition_memory_output
    )
    controller.transition_tape_output = mx.random.normal(
        controller.transition_tape_output.shape,
        key=mx.random.key(131),
    )
    state = controller.exact_probabilities(
        (0, 3, 5, 7, 0),
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    changed_state = controller.exact_probabilities(
        (0, 3, 9, 7, 0),
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    first = controller.exact_probabilities(
        (8, 1, 2, 3, 4, 5, 6, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )
    second = controller.exact_probabilities(
        (9, 6, 5, 4, 3, 2, 1, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )
    current = controller.exact_probabilities(
        (10, 11, 12, 13, 14, 15, 16, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )

    forward = controller._typed_transition_memory(
        (first, second, current),
        state_probabilities=state,
        action_probabilities=current,
    )
    reversed_prefix = controller._typed_transition_memory(
        (second, first, current),
        state_probabilities=state,
        action_probabilities=current,
    )
    changed_query = controller._typed_transition_memory(
        (first, second, current),
        state_probabilities=changed_state,
        action_probabilities=current,
    )

    mx.eval(forward, reversed_prefix, changed_query)
    assert not bool(mx.allclose(forward, reversed_prefix))
    assert not bool(mx.allclose(forward, changed_query))


def test_typed_transition_processor_attaches_as_exact_noop() -> None:
    controller = _controller()
    state = controller.exact_probabilities(
        (0, 7, 11, 13, 0),
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    action = controller.exact_probabilities(
        (8, 1, 0, 2, 3, 4, 31, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )

    logits = controller.typed_transition_processor_logits(state, action, None)

    mx.eval(logits)
    assert logits.shape == state.shape
    assert bool(mx.all(logits == 0))


def test_copy_write_processor_is_identity_at_zero_attachment() -> None:
    controller = _controller()
    values = (3, 7, 11, 13, 0)
    state = controller.exact_probabilities(
        values,
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    action = controller.exact_probabilities(
        (8, 1, 0, 2, 3, 4, 31, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )

    copied = controller.resolve_transition_processor_logits(
        None,
        state,
        action,
        None,
        transition_processor_mode="copy_write",
    )
    regenerated = controller.resolve_transition_processor_logits(
        None,
        state,
        action,
        None,
        transition_processor_mode="authoritative",
    )

    mx.eval(copied, regenerated)
    assert tuple(mx.argmax(copied[0], axis=-1).tolist()) == values
    assert tuple(mx.argmax(regenerated[0], axis=-1).tolist()) != values


def test_copy_write_processor_retains_identity_with_small_positive_prior() -> None:
    controller = _controller()
    values = (3, 7, 11, 13, 0)
    state = controller.exact_probabilities(
        values,
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    action = controller.exact_probabilities(
        (8, 1, 0, 2, 3, 4, 31, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )

    copied = controller.resolve_transition_processor_logits(
        None,
        state,
        action,
        None,
        transition_processor_mode="copy_write",
        transition_copy_prior_logit_bias=0.05,
    )

    mx.eval(copied)
    assert tuple(mx.argmax(copied[0], axis=-1).tolist()) == values


@pytest.mark.parametrize("bias", (-0.01, float("inf"), 8.01, True))
def test_copy_write_processor_rejects_invalid_prior_bias(bias: object) -> None:
    controller = _controller()
    state = controller.exact_probabilities(
        (3, 7, 11, 13, 0),
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    action = controller.exact_probabilities(
        (8, 1, 0, 2, 3, 4, 31, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )

    with pytest.raises(ValueError, match="copy prior"):
        controller.resolve_transition_processor_logits(
            None,
            state,
            action,
            None,
            transition_processor_mode="copy_write",
            transition_copy_prior_logit_bias=bias,
        )


def test_copy_write_processor_can_override_any_committed_register() -> None:
    controller = _controller()
    state = controller.exact_probabilities(
        (0, 7, 11, 13, 0),
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    action = controller.exact_probabilities(
        (8, 1, 0, 2, 3, 4, 31, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )
    target = (1, 8, 12, 14, 1)
    categories = mx.arange(controller.config.state_cardinality)[None, None, :]
    candidate = 4.0 * (
        categories == mx.array((target,), dtype=mx.int32)[..., None]
    ).astype(mx.float32)
    original = controller.typed_transition_processor_logits
    controller.typed_transition_processor_logits = lambda *_args, **_kwargs: candidate
    try:
        resolved = controller.resolve_transition_processor_logits(
            None,
            state,
            action,
            None,
            transition_processor_mode="copy_write",
        )
    finally:
        controller.typed_transition_processor_logits = original

    mx.eval(resolved)
    assert tuple(mx.argmax(resolved[0], axis=-1).tolist()) == target


def test_cross_register_transition_tissue_is_zero_attached_then_causal() -> None:
    controller = _controller()
    assert bool(mx.all(controller.transition_processor_state_cross_projection == 0))
    controller.transition_processor_state_cross_projection = (
        controller.transition_processor_state_cross_projection.at[1, 2].add(
            mx.eye(controller.config.correction_rank)
        )
    )
    controller.transition_processor_output = mx.random.normal(
        controller.transition_processor_output.shape,
        key=mx.random.key(1251),
    )
    state = controller.exact_probabilities(
        (0, 7, 11, 13, 0),
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    changed = controller.exact_probabilities(
        (0, 7, 12, 13, 0),
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    action = controller.exact_probabilities(
        (8, 1, 0, 2, 3, 4, 31, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )

    baseline = controller.typed_transition_processor_logits(state, action, None)
    observed = controller.typed_transition_processor_logits(changed, action, None)
    mx.eval(baseline, observed)
    assert not bool(mx.allclose(baseline[:, 1], observed[:, 1]))


def test_typed_transition_processor_opcode_experts_are_isolated() -> None:
    controller = _controller()
    controller.transition_processor_opcode_output = (
        controller.transition_processor_opcode_output.at[8].add(1.0)
    )
    state = controller.exact_probabilities(
        (0, 7, 11, 13, 0),
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    opcode_eight = controller.exact_probabilities(
        (8, 1, 0, 2, 3, 4, 31, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )
    opcode_nine = controller.exact_probabilities(
        (9, 1, 0, 2, 3, 4, 31, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )

    active = controller.typed_transition_processor_logits(
        state, opcode_eight, None
    )
    isolated = controller.typed_transition_processor_logits(
        state, opcode_nine, None
    )

    mx.eval(active, isolated)
    assert bool(mx.any(active != 0))
    assert bool(mx.all(isolated == 0))


def test_typed_transition_processor_hidden_experts_are_noop_isolated_and_controlled() -> None:
    controller = _controller()
    controller.transition_processor_output = mx.random.normal(
        controller.transition_processor_output.shape,
        key=mx.random.key(120),
    )
    controller.transition_processor_opcode_hidden = (
        controller.transition_processor_opcode_hidden.at[8].add(
            mx.eye(controller.config.correction_rank)[None, :, :]
        )
    )
    state = controller.exact_probabilities(
        (0, 7, 11, 13, 0),
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    opcode_eight = controller.exact_probabilities(
        (8, 1, 0, 2, 3, 4, 31, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )
    opcode_nine = controller.exact_probabilities(
        (9, 1, 0, 2, 3, 4, 31, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )

    active = controller.typed_transition_processor_logits(state, opcode_eight, None)
    isolated = controller.typed_transition_processor_logits(state, opcode_nine, None)
    lesioned = controller.typed_transition_processor_logits(
        state,
        opcode_eight,
        None,
        opcode_expert_routing="lesion",
    )
    isolated_lesioned = controller.typed_transition_processor_logits(
        state,
        opcode_nine,
        None,
        opcode_expert_routing="lesion",
    )
    control_eight = controller.typed_transition_processor_logits(
        state,
        opcode_eight,
        None,
        opcode_expert_routing="uniform",
    )
    control_nine = controller.typed_transition_processor_logits(
        state,
        opcode_nine,
        None,
        opcode_expert_routing="uniform",
    )

    mx.eval(
        active,
        isolated,
        lesioned,
        isolated_lesioned,
        control_eight,
        control_nine,
    )
    assert not bool(mx.allclose(active, lesioned))
    assert bool(mx.allclose(isolated, isolated_lesioned))
    assert not bool(mx.allclose(control_eight, lesioned))
    assert not bool(mx.allclose(control_nine, isolated_lesioned))
    with pytest.raises(ValueError, match="routing differs"):
        controller.typed_transition_processor_logits(
            state,
            opcode_eight,
            None,
            opcode_expert_routing="private_family",
        )


def test_typed_transition_processor_interaction_experts_specialize_before_compression() -> None:
    controller = _controller()
    controller.transition_processor_opcode_interaction_down = (
        controller.transition_processor_opcode_interaction_down.at[8].add(1.0)
    )
    controller.transition_processor_output = mx.random.normal(
        controller.transition_processor_output.shape,
        key=mx.random.key(121),
    )
    state = controller.exact_probabilities(
        (0, 7, 11, 13, 0),
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    opcode_eight = controller.exact_probabilities(
        (8, 1, 0, 2, 3, 4, 31, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )
    opcode_nine = controller.exact_probabilities(
        (9, 1, 0, 2, 3, 4, 31, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )

    active = controller.typed_transition_processor_logits(
        state,
        opcode_eight,
        None,
    )
    active_lesioned = controller.typed_transition_processor_logits(
        state,
        opcode_eight,
        None,
        opcode_expert_routing="lesion",
    )
    isolated = controller.typed_transition_processor_logits(
        state,
        opcode_nine,
        None,
    )
    isolated_lesioned = controller.typed_transition_processor_logits(
        state,
        opcode_nine,
        None,
        opcode_expert_routing="lesion",
    )

    mx.eval(active, active_lesioned, isolated, isolated_lesioned)
    assert not bool(mx.allclose(active, active_lesioned))
    assert bool(mx.allclose(isolated, isolated_lesioned))


def test_typed_transition_processor_preserves_state_action_and_history_identity() -> None:
    controller = _controller()
    controller.transition_processor_output = mx.random.normal(
        controller.transition_processor_output.shape,
        key=mx.random.key(119),
    )
    state = controller.exact_probabilities(
        (0, 7, 11, 13, 0),
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    changed_state = controller.exact_probabilities(
        (0, 8, 11, 13, 0),
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    action = controller.exact_probabilities(
        (8, 1, 0, 2, 3, 4, 31, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )
    changed_action = controller.exact_probabilities(
        (8, 1, 2, 0, 3, 4, 31, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )
    history = mx.random.normal(
        (1, controller.config.state_slots, controller.config.correction_rank),
        key=mx.random.key(120),
    )

    baseline = controller.typed_transition_processor_logits(state, action, history)
    state_changed = controller.typed_transition_processor_logits(
        changed_state, action, history
    )
    action_changed = controller.typed_transition_processor_logits(
        state, changed_action, history
    )
    history_changed = controller.typed_transition_processor_logits(
        state, action, history + 1.0
    )

    mx.eval(baseline, state_changed, action_changed, history_changed)
    assert not bool(mx.allclose(baseline, state_changed))
    assert not bool(mx.allclose(baseline, action_changed))
    assert not bool(mx.allclose(baseline, history_changed))


def test_typed_transition_processor_has_an_independent_lesion() -> None:
    controller = _controller()
    controller.transition_processor_output = mx.random.normal(
        controller.transition_processor_output.shape,
        key=mx.random.key(121),
    )
    problem = mx.random.normal((1, 7, 64), key=mx.random.key(122))
    hidden = mx.random.normal((1, 10, 64), key=mx.random.key(123))
    state = controller.exact_probabilities(
        (0, 7, 11, 13, 0),
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    action = controller.exact_probabilities(
        (8, 1, 0, 2, 3, 4, 31, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )
    action_state = controller.commit_action_probabilities(action)

    intact = controller.state_transition_logits(
        problem,
        hidden,
        state_slot_start=3,
        step=0,
        action_state=action_state,
        state_probabilities=state,
        action_probabilities=action,
        action_probability_history=(action,),
        microcode_lesion=True,
    )
    lesioned = controller.state_transition_logits(
        problem,
        hidden,
        state_slot_start=3,
        step=0,
        action_state=action_state,
        state_probabilities=state,
        action_probabilities=action,
        action_probability_history=(action,),
        microcode_lesion=True,
        transition_processor_lesion=True,
    )

    mx.eval(intact, lesioned)
    assert not bool(mx.allclose(intact, lesioned))


def test_public_prefix_replay_attaches_as_noop_and_has_independent_lesion() -> None:
    controller = _controller()
    state = controller.exact_probabilities(
        (0, 7, 11, 13, 0),
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    first = controller.exact_probabilities(
        (8, 1, 0, 2, 3, 4, 31, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )
    second = controller.exact_probabilities(
        (9, 2, 0, 3, 4, 5, 31, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )
    local = controller.typed_transition_processor_logits(
        state,
        second,
        controller._typed_transition_memory(
            (first, second),
            state_probabilities=state,
            action_probabilities=second,
        ),
    )

    attached, candidate, gate = controller.typed_transition_replay_logits(
        local,
        (first, second),
        action_probabilities=second,
        replay_mode="active",
    )
    lesioned, _candidate, lesion_gate = controller.typed_transition_replay_logits(
        local,
        (first, second),
        action_probabilities=second,
        replay_mode="lesion",
    )
    mx.eval(local, attached, candidate, gate, lesioned, lesion_gate)
    assert bool(mx.array_equal(attached, local))
    assert bool(mx.array_equal(lesioned, local))
    assert bool(mx.all(candidate == 0))
    assert bool(mx.all(lesion_gate == 0))


def test_forced_public_prefix_replay_is_state_independent_and_order_sensitive() -> None:
    controller = _controller()
    controller.transition_replay_output = mx.random.normal(
        controller.transition_replay_output.shape,
        key=mx.random.key(4981),
    )
    first = controller.exact_probabilities(
        (8, 1, 0, 2, 3, 4, 31, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )
    second = controller.exact_probabilities(
        (9, 2, 0, 3, 4, 5, 31, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )
    local_a = mx.random.normal(
        (1, controller.config.state_slots, controller.config.state_cardinality),
        key=mx.random.key(4982),
    )
    local_b = mx.random.normal(
        local_a.shape,
        key=mx.random.key(4983),
    )

    forced_a, _candidate_a, _gate_a = controller.typed_transition_replay_logits(
        local_a,
        (first, second),
        action_probabilities=second,
        replay_mode="forced",
    )
    forced_b, _candidate_b, _gate_b = controller.typed_transition_replay_logits(
        local_b,
        (first, second),
        action_probabilities=second,
        replay_mode="forced",
    )
    reversed_prefix, _candidate_c, _gate_c = (
        controller.typed_transition_replay_logits(
            local_a,
            (second, first),
            action_probabilities=first,
            replay_mode="forced",
        )
    )
    mx.eval(forced_a, forced_b, reversed_prefix)
    assert bool(mx.array_equal(forced_a, forced_b))
    assert not bool(mx.allclose(forced_a, reversed_prefix))


def test_authoritative_transition_matches_the_direct_training_surface() -> None:
    controller = _controller()
    controller.transition_processor_output = mx.random.normal(
        controller.transition_processor_output.shape,
        key=mx.random.key(124),
    )
    controller.state_transition_output = mx.random.normal(
        controller.state_transition_output.shape,
        key=mx.random.key(125),
    )
    problem = mx.random.normal((1, 7, 64), key=mx.random.key(126))
    hidden = mx.random.normal((1, 10, 64), key=mx.random.key(127))
    state = controller.exact_probabilities(
        (0, 7, 11, 13, 0),
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    action = controller.exact_probabilities(
        (8, 1, 0, 2, 3, 4, 31, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )
    history = controller._typed_transition_memory((action,))
    direct = controller.typed_transition_processor_logits(state, action, history)
    deployed = controller.state_transition_logits(
        problem,
        hidden,
        state_slot_start=3,
        step=0,
        action_state=controller.commit_action_probabilities(action),
        state_probabilities=state,
        action_probabilities=action,
        action_probability_history=(action,),
        microcode_lesion=True,
        transition_processor_mode="authoritative",
    )
    residual = controller.state_transition_logits(
        problem,
        hidden,
        state_slot_start=3,
        step=0,
        action_state=controller.commit_action_probabilities(action),
        state_probabilities=state,
        action_probabilities=action,
        action_probability_history=(action,),
        microcode_lesion=True,
        transition_processor_mode="residual",
    )

    mx.eval(direct, deployed, residual)
    assert bool(mx.array_equal(direct, deployed))
    assert not bool(mx.allclose(direct, residual))


def test_microcode_forbidden_authoritative_path_stutters_terminal_state() -> None:
    controller = _controller()
    controller.transition_processor_output = mx.random.normal(
        controller.transition_processor_output.shape,
        key=mx.random.key(128),
    )
    problem = mx.random.normal((1, 7, 64), key=mx.random.key(129))
    hidden = mx.random.normal((1, 10, 64), key=mx.random.key(130))
    terminal_values = (5, 7, 11, 13, 1)
    state = controller.exact_probabilities(
        terminal_values,
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    action = controller.exact_probabilities(
        (32, 32, 32, 32, 32, 32, 32, 1),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )

    logits = controller.state_transition_logits(
        problem,
        hidden,
        state_slot_start=3,
        step=5,
        action_state=controller.commit_action_probabilities(action),
        state_probabilities=state,
        action_probabilities=action,
        action_probability_history=(action,),
        microcode_lesion=True,
        transition_processor_mode="authoritative",
    )

    mx.eval(logits)
    assert tuple(int(value) for value in mx.argmax(logits[0], axis=-1).tolist()) == (
        terminal_values
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
    controller.initial_state_bias = mx.full((5, 33), -100.0)
    for slot in range(5):
        controller.state_transition_bias = (
            controller.state_transition_bias.at[slot, 0].add(200.0)
        )
        controller.initial_state_bias = (
            controller.initial_state_bias.at[slot, 0].add(200.0)
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

    entries_per_step = (
        controller.config.action_slots + 3 * controller.config.state_slots
    )
    assert intact.process_tape_entries == entries_per_step * plan.iterations
    assert intact.process_tape_active_entries == intact.process_tape_entries
    assert lesioned.process_tape_entries == 0
    assert lesioned.process_tape_active_entries == 0
    assert not bool(mx.array_equal(intact_final, lesioned_final))


def test_initial_parser_and_recurrent_transition_do_not_alias() -> None:
    controller = _controller()
    before = mx.array(controller.state_transition_bias)

    controller.initial_state_bias = controller.initial_state_bias + 1.0

    assert bool(mx.array_equal(controller.state_transition_bias, before))
    assert not bool(
        mx.array_equal(controller.initial_state_bias, controller.state_transition_bias)
    )


def test_process_tape_identity_preserves_step_and_entry_kind() -> None:
    controller = _controller()
    value = mx.ones((1, 3, controller.config.hidden_size), dtype=mx.float32)

    action_zero = controller.encode_process_tape_entry(value, step=0, kind="action")
    state_zero = controller.encode_process_tape_entry(
        value, step=0, kind="state_post"
    )
    delta_zero = controller.encode_process_tape_entry(
        value, step=0, kind="state_delta"
    )
    action_one = controller.encode_process_tape_entry(value, step=1, kind="action")

    assert not bool(mx.array_equal(action_zero, state_zero))
    assert not bool(mx.array_equal(state_zero, delta_zero))
    assert not bool(mx.array_equal(action_zero, action_one))
    assert not bool(mx.array_equal(state_zero, action_one))


def test_process_tape_attention_distinguishes_noncommutative_order() -> None:
    controller = _controller()
    candidate = mx.arange(12 * controller.config.hidden_size).reshape(
        (1, 12, controller.config.hidden_size)
    ).astype(mx.float32)
    committed = candidate
    first = mx.ones((1, 5, controller.config.hidden_size), dtype=mx.float32)
    second = mx.full((1, 5, controller.config.hidden_size), 2.0)
    forward = mx.concatenate(
        [
            controller.encode_process_tape_entry(first, step=0, kind="state_post"),
            controller.encode_process_tape_entry(second, step=1, kind="state_post"),
        ],
        axis=1,
    )
    reversed_order = mx.concatenate(
        [
            controller.encode_process_tape_entry(second, step=0, kind="state_post"),
            controller.encode_process_tape_entry(first, step=1, kind="state_post"),
        ],
        axis=1,
    )
    mask = mx.ones((1, 10), dtype=mx.bool_)

    forward_answer = controller.attend_answer_to_state(
        candidate,
        committed,
        state_slot_start=4,
        process_memory=forward,
        process_memory_mask=mask,
    )
    reversed_answer = controller.attend_answer_to_state(
        candidate,
        committed,
        state_slot_start=4,
        process_memory=reversed_order,
        process_memory_mask=mask,
    )

    assert not bool(mx.array_equal(forward_answer[:, 8:, :], reversed_answer[:, 8:, :]))


def test_process_tape_reader_is_causal_and_masks_terminal_suffix() -> None:
    controller = _controller()
    prefix = mx.arange(3 * controller.config.hidden_size).reshape(
        (1, 3, controller.config.hidden_size)
    ).astype(mx.float32)
    first_suffix = mx.ones((1, 2, controller.config.hidden_size), dtype=mx.float32)
    second_suffix = mx.full(
        (1, 2, controller.config.hidden_size), 7.0, dtype=mx.float32
    )
    first = mx.concatenate([prefix, first_suffix], axis=1)
    second = mx.concatenate([prefix, second_suffix], axis=1)
    all_live = mx.ones((1, 5), dtype=mx.bool_)
    terminal_mask = mx.array([[True, True, True, False, False]])

    first_context = controller.contextualize_process_tape(first, all_live)
    second_context = controller.contextualize_process_tape(second, all_live)
    masked_first = controller.contextualize_process_tape(first, terminal_mask)
    masked_second = controller.contextualize_process_tape(second, terminal_mask)

    assert bool(mx.array_equal(first_context[:, :3, :], second_context[:, :3, :]))
    assert not bool(mx.array_equal(first_context[:, 3:, :], second_context[:, 3:, :]))
    assert bool(mx.array_equal(masked_first[:, :3, :], masked_second[:, :3, :]))
    assert bool(mx.array_equal(masked_first[:, 3:, :], first_suffix))
    assert bool(mx.array_equal(masked_second[:, 3:, :], second_suffix))


def test_process_tape_identity_rejects_invalid_entries() -> None:
    controller = _controller()
    value = mx.ones((1, 3, controller.config.hidden_size), dtype=mx.float32)

    with pytest.raises(ValueError, match="non-negative"):
        controller.encode_process_tape_entry(value, step=-1, kind="action")
    with pytest.raises(ValueError, match="transition schema"):
        controller.encode_process_tape_entry(value, step=0, kind="unknown")
    with pytest.raises(ValueError, match="residual layout"):
        controller.encode_process_tape_entry(
            value[..., :-1], step=0, kind="state_post"
        )


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
