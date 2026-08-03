"""Generated-prefix and branch-selection contracts for objective v5."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_lm")

from mlx.utils import tree_flatten, tree_map  # noqa: E402
from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.execution_spec import (  # noqa: E402
    RLCExecutionSpec,
)
from core.brain.llm.latent_cortex.recurrence_adapter import (  # noqa: E402
    ScopedLoRALinear,
)
from core.learning.recurrence_native_objective_v2 import (  # noqa: E402
    cached_supervised_live_path_value_and_grad,
)
from core.learning.recurrence_native_objective_v5 import (  # noqa: E402
    GeneratedRollinSelectionConfig,
    detached_softmin_weights,
    deterministic_mixed_rollin,
    generated_rollin_live_path_loss,
    generated_rollin_live_path_value_and_grad,
    validate_generated_rollin_receipt,
)

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
    mx.eval(model.parameters())
    return model


def _spec(*, branches: tuple[str, ...] = ("constructive_solution",)) -> RLCExecutionSpec:
    return RLCExecutionSpec(
        n_slots=3,
        branch_roles=branches,
        recurrent_steps=2,
        exchange_interval=1,
    )


def _tree_max_difference(left, right) -> float:
    left_flat = tree_flatten(left)
    right_flat = tree_flatten(right)
    assert [path for path, _value in left_flat] == [path for path, _value in right_flat]
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


def _rehash_receipt(value: dict) -> dict:
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


def test_generated_rollin_config_is_strict_and_hash_bound():
    config = GeneratedRollinSelectionConfig(
        student_forcing_probability=0.75,
        sampling_temperature=0.6,
        branch_softmin_temperature=0.4,
    )
    assert GeneratedRollinSelectionConfig.from_dict(config.to_dict()) == config
    assert len(config.sha256) == 64

    malformed = config.to_dict()
    malformed["unknown"] = True
    with pytest.raises(ValueError, match="fields"):
        GeneratedRollinSelectionConfig.from_dict(malformed)
    with pytest.raises(ValueError, match="branch_softmin_temperature"):
        GeneratedRollinSelectionConfig(branch_softmin_temperature=0.0)


def test_deterministic_mixed_rollin_has_true_probability_boundaries():
    target = (1, 2, 3, 4)
    generated = (5, 6, 7, 8)

    teacher, teacher_positions = deterministic_mixed_rollin(
        target,
        generated,
        probability=0.0,
        base_seed=9,
        branch_index=0,
    )
    student, student_positions = deterministic_mixed_rollin(
        target,
        generated,
        probability=1.0,
        base_seed=9,
        branch_index=0,
    )

    assert teacher == target
    assert teacher_positions == ()
    assert student == (5, 6, 7, 4)
    assert student_positions == (0, 1, 2)
    assert deterministic_mixed_rollin(
        target,
        generated,
        probability=0.5,
        base_seed=77,
        branch_index=1,
    ) == deterministic_mixed_rollin(
        target,
        generated,
        probability=0.5,
        base_seed=77,
        branch_index=1,
    )


def test_softmin_weights_favor_lower_loss_and_preserve_symmetry():
    weights = detached_softmin_weights((0.5, 2.0), temperature=0.5)
    assert sum(weights) == pytest.approx(1.0, abs=1e-15)
    assert weights[0] > weights[1] > 0.0
    assert detached_softmin_weights((1.0, 1.0), temperature=0.5) == (0.5, 0.5)


def test_single_branch_zero_student_forcing_is_exact_v2_gradient():
    model = _model()
    spec = _spec()
    legacy = cached_supervised_live_path_value_and_grad(
        model,
        PROMPT,
        ANSWER,
        spec=spec,
    )
    current = generated_rollin_live_path_value_and_grad(
        model,
        PROMPT,
        ANSWER,
        spec=spec,
        base_seed=19,
        config=GeneratedRollinSelectionConfig(
            student_forcing_probability=0.0,
        ),
    )

    assert current.value == pytest.approx(legacy.value, abs=1e-6)
    assert current.branch_values == pytest.approx(legacy.branch_values, abs=1e-6)
    assert current.branch_weights == (1.0,)
    assert _tree_max_difference(current.gradients, legacy.gradients) < 1e-6


def test_multi_branch_gradient_is_detached_softmin_combination():
    model = _model()
    spec = _spec(branches=("constructive_solution", "critical_audit"))
    config = GeneratedRollinSelectionConfig(
        student_forcing_probability=1.0,
        sampling_temperature=0.0,
        branch_softmin_temperature=0.7,
    )
    combined = generated_rollin_live_path_value_and_grad(
        model,
        PROMPT,
        ANSWER,
        spec=spec,
        base_seed=211,
        config=config,
    )
    branch_zero = generated_rollin_live_path_value_and_grad(
        model,
        PROMPT,
        ANSWER,
        spec=spec,
        base_seed=211,
        config=config,
        branch_indices=(0,),
    )
    branch_one = generated_rollin_live_path_value_and_grad(
        model,
        PROMPT,
        ANSWER,
        spec=spec,
        base_seed=211,
        config=config,
        branch_indices=(1,),
    )
    expected = tree_map(
        lambda left, right: (
            combined.branch_weights[0] * left
            + combined.branch_weights[1] * right
        ),
        branch_zero.gradients,
        branch_one.gradients,
    )
    mx.eval(expected)

    assert combined.branch_values == pytest.approx(
        (branch_zero.value, branch_one.value),
        abs=1e-6,
    )
    assert combined.branch_weights[0] != pytest.approx(0.5, abs=1e-4)
    assert _tree_max_difference(combined.gradients, expected) < 2e-6


def test_streamed_branch_accumulation_is_order_invariant():
    model = _model()
    spec = _spec(branches=("constructive_solution", "critical_audit"))
    config = GeneratedRollinSelectionConfig(
        student_forcing_probability=1.0,
        sampling_temperature=0.0,
        branch_softmin_temperature=0.7,
    )
    forward = generated_rollin_live_path_value_and_grad(
        model,
        PROMPT,
        ANSWER,
        spec=spec,
        base_seed=612,
        config=config,
        branch_indices=(0, 1),
    )
    reverse = generated_rollin_live_path_value_and_grad(
        model,
        PROMPT,
        ANSWER,
        spec=spec,
        base_seed=612,
        config=config,
        branch_indices=(1, 0),
    )

    assert forward.value == pytest.approx(reverse.value, abs=1e-7)
    assert dict(zip(forward.branch_indices, forward.branch_values, strict=True)) == pytest.approx(
        dict(zip(reverse.branch_indices, reverse.branch_values, strict=True)),
        abs=1e-6,
    )
    assert _tree_max_difference(forward.gradients, reverse.gradients) < 2e-6


def test_evaluation_matches_gradient_receipt_and_rejects_resealed_arithmetic():
    model = _model()
    spec = _spec(branches=("constructive_solution", "critical_audit"))
    config = GeneratedRollinSelectionConfig(
        student_forcing_probability=0.6,
        sampling_temperature=0.0,
        branch_softmin_temperature=0.8,
    )
    before = tuple(
        (path, mx.array(value))
        for path, value in tree_flatten(model.trainable_parameters())
    )
    mx.eval([value for _path, value in before])

    evaluation = generated_rollin_live_path_loss(
        model,
        PROMPT,
        ANSWER,
        spec=spec,
        base_seed=9001,
        config=config,
    )
    result = generated_rollin_live_path_value_and_grad(
        model,
        PROMPT,
        ANSWER,
        spec=spec,
        base_seed=9001,
        config=config,
    )

    assert evaluation.value == pytest.approx(result.value, abs=1e-6)
    assert evaluation.branch_values == pytest.approx(result.branch_values, abs=1e-6)
    assert evaluation.branch_weights == pytest.approx(result.branch_weights, abs=1e-12)
    assert validate_generated_rollin_receipt(evaluation.receipt()) == evaluation.receipt()
    after = tuple(tree_flatten(model.trainable_parameters()))
    assert all(
        bool(mx.array_equal(before_value, after_value))
        for (_path, before_value), (_other_path, after_value) in zip(
            before,
            after,
            strict=True,
        )
    )

    attacked = copy.deepcopy(evaluation.receipt())
    attacked["branches"][0]["selection_weight"] += 0.01
    attacked["branches"][1]["selection_weight"] -= 0.01
    _rehash_receipt(attacked)
    with pytest.raises(ValueError, match="weights do not replay"):
        validate_generated_rollin_receipt(attacked)
