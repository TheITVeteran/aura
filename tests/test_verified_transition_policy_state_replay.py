"""Exact objective, gradient, and Adam transition replay contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx.optimizers")

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from core.learning.recurrent_grpo import (  # noqa: E402
    build_recurrent_policy_optimizer,
    recurrent_policy_optimizer_config,
    recurrent_policy_tensor_map_sha256,
)
from core.learning.verified_transition_policy_state_replay import (  # noqa: E402
    POLICY_STATE_REPLAY_RECEIPT_SCHEMA,
    VerifiedTransitionPolicyStateReplayError,
    replay_verified_policy_transition,
    validate_policy_state_replay_receipt,
)


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = mx.array([[0.25, -0.5], [0.75, 0.125]])
        self.right = mx.array([0.2, -0.3])

    def __call__(self, inputs: Any) -> Any:
        return inputs @ self.left + self.right


@dataclass(frozen=True)
class ObjectiveResult:
    gradients: Any
    loss: float

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "test.exact_objective.v1",
            "loss_hex": self.loss.hex(),
            "has_gradient": True,
        }


def _objective(model: TinyPolicy, *, gradient_scale: float = 1.0) -> ObjectiveResult:
    inputs = mx.array([[0.4, -0.7], [0.2, 0.9]])
    targets = mx.array([[0.1, 0.8], [-0.3, 0.2]])

    def loss_fn(active: TinyPolicy) -> Any:
        residual = active(inputs) - targets
        return mx.mean(residual * residual)

    loss, gradients = nn.value_and_grad(model, loss_fn)(model)
    if gradient_scale != 1.0:
        gradients = tree_unflatten(
            [(name, value * gradient_scale) for name, value in tree_flatten(gradients)]
        )
    mx.eval(loss, gradients)
    return ObjectiveResult(gradients=gradients, loss=float(loss))


def _producer_material() -> dict[str, Any]:
    model = TinyPolicy()
    pre_adapter = dict(tree_flatten(model.trainable_parameters()))
    optimizer = build_recurrent_policy_optimizer(1e-3)
    optimizer.init(model.trainable_parameters())
    pre_optimizer = dict(tree_flatten(optimizer.state))
    objective = _objective(model)
    optimizer.update(model, objective.gradients)
    mx.eval(model.trainable_parameters(), optimizer.state)
    return {
        "pre_adapter": pre_adapter,
        "pre_optimizer": pre_optimizer,
        "post_adapter": dict(tree_flatten(model.trainable_parameters())),
        "post_optimizer": dict(tree_flatten(optimizer.state)),
        "objective_receipt": objective.receipt(),
    }


def _replay(**overrides: Any) -> dict[str, Any]:
    material = _producer_material()
    material.update(overrides)
    execution_spec = "a" * 64
    return replay_verified_policy_transition(
        model=TinyPolicy(),
        pre_adapter_tensors=material["pre_adapter"],
        pre_optimizer_tensors=material["pre_optimizer"],
        expected_post_adapter_tensors=material["post_adapter"],
        expected_post_optimizer_tensors=material["post_optimizer"],
        expected_objective_receipt=material["objective_receipt"],
        objective_factory=material.get("objective_factory", _objective),
        optimizer_config=material.get(
            "optimizer_config",
            recurrent_policy_optimizer_config(1e-3),
        ),
        execution_spec_sha256=execution_spec,
        expected_policy_before_sha256=recurrent_policy_tensor_map_sha256(
            material["pre_adapter"], execution_spec
        ),
        expected_policy_after_sha256=recurrent_policy_tensor_map_sha256(
            material["post_adapter"], execution_spec
        ),
    )


def test_exact_replay_recomputes_every_gradient_and_matches_both_post_states():
    receipt = _replay()

    assert receipt["schema"] == POLICY_STATE_REPLAY_RECEIPT_SCHEMA
    assert receipt["objective_recomputed"] is True
    assert receipt["all_gradient_tensors_recomputed"] is True
    assert receipt["external_policy_state_replayed"] is True
    assert receipt["optimizer_update_count"] == 1
    assert receipt["gradient_identity"]["tensor_count"] == 2
    assert [tensor["name"] for tensor in receipt["gradient_identity"]["tensors"]] == [
        "left",
        "right",
    ]
    assert all(
        len(tensor["value_sha256"]) == 64 for tensor in receipt["gradient_identity"]["tensors"]
    )


def test_objective_receipt_drift_fails_before_update():
    expected = _producer_material()["objective_receipt"]
    expected["loss_hex"] = (0.0).hex()
    with pytest.raises(
        VerifiedTransitionPolicyStateReplayError,
        match="objective_receipt_mismatch",
    ):
        _replay(objective_receipt=expected)


def test_gradient_drift_cannot_reach_a_claimed_post_state():
    with pytest.raises(
        VerifiedTransitionPolicyStateReplayError,
        match="post_adapter_mismatch",
    ):
        _replay(
            objective_factory=lambda model: _objective(
                model,
                gradient_scale=0.5,
            )
        )


def test_adapter_post_state_value_and_dtype_drift_fail_closed():
    material = _producer_material()
    changed = dict(material["post_adapter"])
    changed["left"] = changed["left"] + 1e-4
    with pytest.raises(
        VerifiedTransitionPolicyStateReplayError,
        match="post_adapter_mismatch",
    ):
        _replay(post_adapter=changed)

    changed = dict(material["post_adapter"])
    changed["left"] = changed["left"].astype(mx.float16)
    with pytest.raises(
        VerifiedTransitionPolicyStateReplayError,
        match="post_adapter_mismatch",
    ):
        _replay(post_adapter=changed)


def test_optimizer_post_state_and_config_drift_fail_closed():
    material = _producer_material()
    changed = dict(material["post_optimizer"])
    changed["step"] = changed["step"] + 1
    with pytest.raises(
        VerifiedTransitionPolicyStateReplayError,
        match="post_optimizer_mismatch",
    ):
        _replay(post_optimizer=changed)

    config = recurrent_policy_optimizer_config(1e-3)
    config["bias_correction"] = True
    with pytest.raises(
        VerifiedTransitionPolicyStateReplayError,
        match="optimizer_config_mismatch",
    ):
        _replay(optimizer_config=config)


def test_pre_state_key_shape_and_dtype_drift_fail_before_objective():
    material = _producer_material()
    missing = dict(material["pre_adapter"])
    missing.pop("right")
    with pytest.raises(
        VerifiedTransitionPolicyStateReplayError,
        match="adapter_keyset_mismatch",
    ):
        _replay(pre_adapter=missing)

    changed = dict(material["pre_optimizer"])
    changed["step"] = changed["step"].astype(mx.int64)
    with pytest.raises(
        VerifiedTransitionPolicyStateReplayError,
        match="optimizer_layout_mismatch",
    ):
        _replay(pre_optimizer=changed)


def test_objective_cannot_mutate_policy_while_claiming_same_receipt():
    def mutating(model: TinyPolicy) -> ObjectiveResult:
        result = _objective(model)
        model.left = model.left + 1.0
        mx.eval(model.trainable_parameters())
        return result

    with pytest.raises(
        VerifiedTransitionPolicyStateReplayError,
        match="objective_mutated_policy",
    ):
        _replay(objective_factory=mutating)


def _reseal(receipt: dict[str, Any]) -> dict[str, Any]:
    import hashlib
    import json

    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return receipt


def test_receipt_validator_rejects_resealed_semantic_falsehoods():
    receipt = _replay()
    assert validate_policy_state_replay_receipt(receipt) == receipt

    receipt["external_policy_state_replayed"] = False
    with pytest.raises(
        VerifiedTransitionPolicyStateReplayError,
        match="receipt_invalid",
    ):
        validate_policy_state_replay_receipt(_reseal(receipt))


def test_receipt_validator_rejects_reordered_or_mismatched_gradient_inventory():
    receipt = _replay()
    gradient = receipt["gradient_identity"]
    gradient["tensors"] = list(reversed(gradient["tensors"]))
    gradient_unsigned = {
        key: value for key, value in gradient.items() if key != "tensor_root_sha256"
    }
    import hashlib
    import json

    gradient["tensor_root_sha256"] = hashlib.sha256(
        json.dumps(
            gradient_unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    with pytest.raises(
        VerifiedTransitionPolicyStateReplayError,
        match="tensor_inventory_invalid",
    ):
        validate_policy_state_replay_receipt(_reseal(receipt))
