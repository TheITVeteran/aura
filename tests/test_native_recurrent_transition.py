"""Contracts for the identity-initialized native recurrent transition."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
optim = pytest.importorskip("mlx.optimizers")

from mlx.utils import tree_flatten  # noqa: E402

from core.brain.llm.latent_cortex.recurrent_transition_core import (  # noqa: E402
    RecurrentTransitionCore,
    RecurrentTransitionCoreConfig,
)
from core.learning.native_recurrent_transition import (  # noqa: E402
    ActionCodebookSpec,
    encode_transition_action,
    evaluate_native_transition,
    native_transition_loss,
    native_transition_value_and_grad,
)
from core.learning.recurrence_curriculum import nested_boolean  # noqa: E402
from core.learning.recurrent_transition_supervision import (  # noqa: E402
    StateCodebookSpec,
    decode_trace_state,
    encode_trace_state,
    encode_trace_state_operand,
)


def _core() -> RecurrentTransitionCore:
    core = RecurrentTransitionCore(
        RecurrentTransitionCoreConfig(
            hidden_size=32,
            bottleneck_size=16,
            attention_heads=4,
        )
    )
    mx.eval(core.parameters())
    return core


def _inputs():
    return (
        mx.random.normal((1, 7, 32), key=mx.random.key(41)),
        mx.random.normal((1, 5, 32), key=mx.random.key(42)),
    )


def _action(program, transition_index: int = 0):
    return encode_transition_action(
        program,
        transition_index=transition_index,
        width=16,
        codebook=ActionCodebookSpec(),
    )


def _typed_state(program, transition_index: int = 0):
    return encode_trace_state_operand(
        program.state_trace,
        state_index=transition_index,
        width=16,
        codebook=StateCodebookSpec(max_program_depth=program.state_trace.depth),
    )


def test_native_core_attaches_as_exact_identity_and_protects_semantic_slots():
    core = _core()
    state, context = _inputs()
    program = nested_boolean(2, 17).transition_program
    assert program is not None
    action = _action(program)
    typed_state = _typed_state(program)
    attached = core(state, context, typed_state, action)
    mx.eval(attached.state, attached.write_gate, attached.delta)

    assert bool(mx.array_equal(attached.state, state))
    assert attached.state_features.shape == (1, 3, 16)
    assert attached.action_features.shape == (1, 3, 16)
    assert attached.write_gate.shape == (1, 3, 1)
    assert attached.delta.shape == (1, 3, 32)

    core.delta_up.weight = mx.ones_like(core.delta_up.weight) * 0.01
    moved = core(state, context, typed_state, action).state
    mx.eval(moved)
    assert bool(mx.array_equal(moved[:, :-3, :], state[:, :-3, :]))
    assert not bool(mx.array_equal(moved[:, -3:, :], state[:, -3:, :]))


def test_native_core_refuses_malformed_workspace_shapes():
    core = _core()
    state, context = _inputs()
    program = nested_boolean(2, 17).transition_program
    assert program is not None
    action = _action(program)
    typed_state = _typed_state(program)

    with pytest.raises(ValueError, match="tensor shape"):
        core(state[:, -3:, :], context, typed_state, action)
    with pytest.raises(ValueError, match="tensor shape"):
        core(state, context[:, :, :-1], typed_state, action)
    with pytest.raises(ValueError, match="tensor shape"):
        core(state, context, typed_state[:, :, :-1], action)
    with pytest.raises(ValueError, match="tensor shape"):
        core(state, context, typed_state, action[:, :, :-1])
    with pytest.raises(ValueError, match="configuration"):
        RecurrentTransitionCoreConfig(
            hidden_size=32,
            bottleneck_size=15,
            attention_heads=4,
        )


def test_typed_action_is_a_causal_operand_of_the_state_update():
    core = _core()
    state, context = _inputs()
    first = nested_boolean(2, 17).transition_program
    second = nested_boolean(2, 18).transition_program
    assert first is not None and second is not None
    first_action = _action(first)
    second_action = _action(second)
    typed_state = _typed_state(first)
    if bool(mx.array_equal(first_action, second_action)):
        second = nested_boolean(2, 19).transition_program
        assert second is not None
        second_action = _action(second)
    assert not bool(mx.array_equal(first_action, second_action))

    core.delta_up.weight = mx.ones_like(core.delta_up.weight) * 0.01
    first_state = core(state, context, typed_state, first_action).state
    second_state = core(state, context, typed_state, second_action).state
    mx.eval(first_state, second_state)
    assert not bool(mx.array_equal(first_state[:, -3:, :], second_state[:, -3:, :]))


def test_trace_codebook_round_trips_initial_and_terminal_states():
    task = nested_boolean(2, 17)
    trace = task.transition_trace
    assert trace is not None
    state, _context = _inputs()
    codebook = StateCodebookSpec(max_program_depth=2)

    for state_index in (0, 1, 2):
        encoded = encode_trace_state(
            state,
            trace,
            state_index=state_index,
            codebook=codebook,
        )
        assert (
            decode_trace_state(
                encoded,
                trace,
                state_index=state_index,
                codebook=codebook,
            )
            == trace.states[state_index]
        )


def test_native_objective_reaches_core_and_receipt_keeps_labels_private():
    core = _core()
    state, context = _inputs()
    task = nested_boolean(2, 29)
    program = task.transition_program
    assert program is not None
    state_codebook = StateCodebookSpec(max_program_depth=2)
    action_codebook = ActionCodebookSpec()

    result = native_transition_value_and_grad(
        core,
        state,
        context,
        program,
        transition_index=0,
        state_codebook=state_codebook,
        action_codebook=action_codebook,
    )
    flattened = tree_flatten(result.gradients)
    assert flattened
    assert all(bool(mx.all(mx.isfinite(value))) for _path, value in flattened)
    assert any(float(mx.max(mx.abs(value))) > 0.0 for _path, value in flattened)

    evaluation = evaluate_native_transition(
        core,
        state,
        context,
        program,
        transition_index=0,
        state_codebook=state_codebook,
        action_codebook=action_codebook,
    )
    receipt = evaluation.receipt()
    assert "predicted_state" not in receipt
    assert "expected_state" not in receipt
    assert "predicted_action" not in receipt
    assert "expected_action" not in receipt
    assert receipt["state_codebook_sha256"] == state_codebook.sha256
    assert receipt["action_codebook_sha256"] == action_codebook.sha256


def test_one_shared_core_can_reduce_exact_transition_loss():
    core = _core()
    state, context = _inputs()
    task = nested_boolean(2, 47)
    program = task.transition_program
    assert program is not None
    state_codebook = StateCodebookSpec(max_program_depth=2)
    action_codebook = ActionCodebookSpec()
    optimizer = optim.Adam(learning_rate=0.01)

    def loss_fn(candidate: RecurrentTransitionCore):
        return native_transition_loss(
            candidate,
            state,
            context,
            program,
            transition_index=0,
            state_codebook=state_codebook,
            action_codebook=action_codebook,
        )

    loss_and_grad = nn.value_and_grad(core, loss_fn)
    initial = float(loss_fn(core))
    for _step in range(24):
        loss, gradients = loss_and_grad(core)
        optimizer.update(core, gradients)
        mx.eval(core.parameters(), optimizer.state, loss)
    final = float(loss_fn(core))

    assert final < initial * 0.5
