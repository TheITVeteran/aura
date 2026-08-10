"""Contracts for exact one-step supervision of the shared recurrent operator."""

from __future__ import annotations

import copy

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_lm")

from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec  # noqa: E402
from core.brain.llm.latent_cortex.recurrence_adapter import (  # noqa: E402
    ScopedLoRALinear,
)
from core.learning.recurrence_curriculum import nested_boolean  # noqa: E402
from core.learning.recurrence_native_objective_v2 import (  # noqa: E402
    prepare_recurrent_transition_input,
    validate_recurrent_transition_input_receipt,
)
from core.learning.recurrent_transition_supervision import (  # noqa: E402
    StateCodebookSpec,
    decode_structured_state,
    encode_structured_state,
    evaluate_state_supervised_transition,
    state_supervised_transition_value_and_grad,
)


def _model() -> Model:
    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=32,
            num_hidden_layers=4,
            intermediate_size=64,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=64,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rope_theta=10000.0,
        )
    )
    model.freeze()
    for layer_index in (1, 2):
        parent = model.model.layers[layer_index].self_attn
        parent.o_proj = ScopedLoRALinear.from_base(
            parent.o_proj,
            r=2,
            scale=1.0,
            block_index=layer_index,
            site=f"model.layers.{layer_index}.self_attn.o_proj",
        )
    mx.eval(model.parameters())
    return model


def _spec(**changes) -> RLCExecutionSpec:
    payload = RLCExecutionSpec(
        n_slots=4,
        branch_roles=("constructive_solution",),
        recurrent_steps=2,
        exchange_interval=2,
    ).to_dict()
    payload.update(changes)
    return RLCExecutionSpec.from_dict(payload)


def test_prepared_transition_is_student_rollin_with_replayable_receipt():
    model = _model()
    spec = _spec()
    first = prepare_recurrent_transition_input(
        model,
        [3, 7, 11, 19],
        spec=spec,
        transition_index=0,
    )
    second = prepare_recurrent_transition_input(
        model,
        [3, 7, 11, 19],
        spec=spec,
        transition_index=1,
    )

    assert first.transition_index == 0
    assert second.transition_index == 1
    assert first.parent_branch_sha256s != second.parent_branch_sha256s
    receipt = validate_recurrent_transition_input_receipt(second.receipt())
    assert receipt["receipt_sha256"] == second.receipt_sha256

    tampered = copy.deepcopy(receipt)
    tampered["transition_index"] = 0
    with pytest.raises(ValueError, match="invalid"):
        validate_recurrent_transition_input_receipt(tampered)


def test_fixed_codebook_round_trips_exact_state_without_learning_a_decoder():
    task = nested_boolean(2, 17)
    trace = task.transition_trace
    assert trace is not None
    codebook = StateCodebookSpec(max_program_depth=2)
    state = mx.random.normal((1, 4, 32), key=mx.random.key(91))
    encoded = encode_structured_state(
        state,
        trace,
        transition_index=0,
        codebook=codebook,
    )

    # The fixed codebook is itself the decoder; no learned head can absorb
    # credit that belongs to the recurrent transition.
    predicted = decode_structured_state(
        encoded,
        trace,
        transition_index=0,
        codebook=codebook,
    )
    assert predicted == trace.states[1]


def test_state_loss_reaches_recurrent_adapter_and_public_receipt_hides_labels():
    model = _model()
    spec = _spec()
    task = nested_boolean(2, 29)
    trace = task.transition_trace
    assert trace is not None
    codebook = StateCodebookSpec(max_program_depth=2)
    prepared = prepare_recurrent_transition_input(
        model,
        [5, 9, 17, 3, 42],
        spec=spec,
        transition_index=0,
    )

    result = state_supervised_transition_value_and_grad(
        model,
        prepared,
        trace,
        spec=spec,
        codebook=codebook,
    )
    flattened = tree_flatten(result.gradients)
    assert flattened
    assert all(bool(mx.all(mx.isfinite(value))) for _path, value in flattened)
    assert any(float(mx.max(mx.abs(value))) > 0.0 for _path, value in flattened)

    evaluation = evaluate_state_supervised_transition(
        model,
        prepared,
        trace,
        spec=spec,
        codebook=codebook,
    )
    receipt = evaluation.receipt()
    assert "predicted" not in receipt
    assert "expected" not in receipt
    assert trace.states[1] != ()


def test_state_supervision_refuses_depth_and_execution_spec_drift():
    model = _model()
    spec = _spec()
    prepared = prepare_recurrent_transition_input(
        model,
        [3, 5, 7],
        spec=spec,
        transition_index=0,
    )
    task = nested_boolean(2, 31)
    trace = task.transition_trace
    assert trace is not None
    codebook = StateCodebookSpec(max_program_depth=1)
    with pytest.raises(ValueError, match="outside the codebook"):
        state_supervised_transition_value_and_grad(
            model,
            prepared,
            trace,
            spec=spec,
            codebook=codebook,
        )

    valid_codebook = StateCodebookSpec(max_program_depth=2)
    with pytest.raises(ValueError, match="execution spec differs"):
        state_supervised_transition_value_and_grad(
            model,
            prepared,
            trace,
            spec=_spec(alpha=0.75),
            codebook=valid_codebook,
        )
