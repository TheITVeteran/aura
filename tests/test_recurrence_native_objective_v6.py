"""Exact structural branch-training contracts for objective v6."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
optim = pytest.importorskip("mlx.optimizers")
pytest.importorskip("mlx_lm")

from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.execution_spec import (  # noqa: E402
    RLCExecutionSpec,
)
from core.brain.llm.latent_cortex.recurrence_adapter import (  # noqa: E402
    ScopedLoRALinear,
)
from core.learning.recurrence_native_objective_v2 import (  # noqa: E402
    LivePathForward,
    _advance_recurrent_states,
    _prepare_recurrent_prefix,
)
from core.learning.recurrence_native_objective_v4 import (  # noqa: E402
    pairwise_separations,
)
from core.learning.recurrence_native_objective_v5 import (  # noqa: E402
    GeneratedRollinSelectionConfig,
)
from core.learning.recurrence_native_objective_v6 import (  # noqa: E402
    BranchSpecializationConfig,
    branch_specialization_live_path_loss,
    branch_specialization_live_path_value_and_grad,
    generated_rollin_specialization_value_and_grad,
    validate_branch_specialization_receipt,
    validate_generated_rollin_specialization_receipt,
)
from core.learning.role_conditioned_lora import wrap_role_conditioned  # noqa: E402

PROMPT = [5, 9, 17, 3, 42]
ANSWER = [7, 11, 23]


def _model() -> Model:
    args = ModelArgs(
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
    model = Model(args)
    model.freeze()
    for index in (1, 2):
        parent = model.model.layers[index].self_attn
        wrapped = ScopedLoRALinear.from_base(parent.o_proj, r=2, scale=1.0)
        wrapped.lora_a = mx.ones_like(wrapped.lora_a) * 0.03
        wrapped.lora_b = mx.ones_like(wrapped.lora_b) * 0.03
        parent.o_proj = wrapped
    wrap_role_conditioned(model, branches=2)
    mx.eval(model.parameters())
    return model


def _spec() -> RLCExecutionSpec:
    return RLCExecutionSpec(
        n_slots=3,
        branch_roles=("constructive_solution", "critical_audit"),
        recurrent_steps=2,
        exchange_interval=1,
    )


def _tree_max_difference(left, right) -> float:
    left_flat = tree_flatten(left)
    right_flat = tree_flatten(right)
    assert [path for path, _value in left_flat] == [
        path for path, _value in right_flat
    ]
    differences = [
        mx.max(mx.abs(left_value - right_value))
        for (_path, left_value), (_other_path, right_value) in zip(
            left_flat,
            right_flat,
            strict=True,
        )
    ]
    mx.eval(differences)
    return max(float(value) for value in differences)


def _rehash(value: dict) -> dict:
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    value["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return value


def test_exact_adjoint_matches_full_unroll_gradient():
    model = _model()
    spec = _spec()
    config = BranchSpecializationConfig(weight=2.5, target_separation=0.30)
    exact = branch_specialization_live_path_value_and_grad(
        model,
        PROMPT,
        spec=spec,
        config=config,
    )

    def full_objective(current_model):
        (
            _prompt_embeddings,
            _seeds,
            prompts,
            states,
            anchors,
            prelude_end,
            coda_start,
        ) = _prepare_recurrent_prefix(current_model, PROMPT, spec=spec)
        for step in range(spec.recurrent_steps):
            states = _advance_recurrent_states(
                current_model,
                prompts,
                states,
                anchors,
                spec,
                step,
                prelude_end,
                coda_start,
            )
        forward = LivePathForward(
            branch_logits=(),
            branch_states=states,
            exchanges=0,
            prompt_tokens=len(PROMPT),
            answer_tokens=0,
            bridge_tokens=0,
        )
        separations = pairwise_separations(forward, comm_slot=spec.comm_slot)
        return config.weight * sum(
            mx.maximum(config.target_separation - value, 0.0)
            for value in separations
        ) / len(separations)

    full_value, full_gradients = nn.value_and_grad(model, full_objective)(model)
    mx.eval(full_value, full_gradients)
    assert exact.value == pytest.approx(float(full_value), rel=2e-5, abs=2e-6)
    assert _tree_max_difference(exact.gradients, full_gradients) < 2e-4


def test_structural_update_moves_collapsed_branches_apart():
    model = _model()
    spec = _spec()
    config = BranchSpecializationConfig(weight=8.0, target_separation=0.30)
    before = branch_specialization_live_path_loss(
        model,
        PROMPT,
        spec=spec,
        config=config,
    )
    result = branch_specialization_live_path_value_and_grad(
        model,
        PROMPT,
        spec=spec,
        config=config,
    )
    optimizer = optim.SGD(learning_rate=0.01)
    optimizer.update(model, result.gradients)
    mx.eval(model.trainable_parameters(), optimizer.state)
    after = branch_specialization_live_path_loss(
        model,
        PROMPT,
        spec=spec,
        config=config,
    )
    assert after.separations[0] > before.separations[0]
    assert after.raw_penalty < before.raw_penalty


def test_receipt_replays_hinge_and_rejects_rehashed_measurement_tamper():
    model = _model()
    result = branch_specialization_live_path_value_and_grad(
        model,
        PROMPT,
        spec=_spec(),
    )
    receipt = result.evaluation.receipt()
    assert validate_branch_specialization_receipt(receipt) == receipt

    attacked = copy.deepcopy(receipt)
    attacked["separations"][0] += 0.1
    _rehash(attacked)
    with pytest.raises(ValueError, match="raw penalty does not replay"):
        validate_branch_specialization_receipt(attacked)


def test_composite_receipt_binds_generated_and_structural_objectives():
    model = _model()
    result = generated_rollin_specialization_value_and_grad(
        model,
        PROMPT,
        ANSWER,
        spec=_spec(),
        base_seed=90210,
        generated_config=GeneratedRollinSelectionConfig(
            student_forcing_probability=0.5,
            sampling_temperature=0.8,
            branch_softmin_temperature=0.5,
        ),
        specialization_config=BranchSpecializationConfig(weight=2.0),
    )
    receipt = result.evaluation.receipt()
    assert (
        validate_generated_rollin_specialization_receipt(receipt)
        == receipt
    )
    assert result.value == pytest.approx(
        result.evaluation.generated.value
        + result.evaluation.specialization.value
    )

    attacked = copy.deepcopy(receipt)
    attacked["specialization_value"] += 0.01
    _rehash(attacked)
    with pytest.raises(ValueError, match="specialization value does not replay"):
        validate_generated_rollin_specialization_receipt(attacked)


def test_single_branch_is_rejected_instead_of_reporting_vacuous_diversity():
    model = _model()
    one_branch = RLCExecutionSpec(
        n_slots=3,
        branch_roles=("constructive_solution",),
        recurrent_steps=2,
    )
    with pytest.raises(ValueError, match="two or more"):
        branch_specialization_live_path_loss(
            model,
            PROMPT,
            spec=one_branch,
        )
