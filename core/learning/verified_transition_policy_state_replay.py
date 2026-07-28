"""Independent exact replay of one verified recurrent policy transition.

This module is the arithmetic kernel used by the external campaign verifier. It
does not trust a producer's claim that an objective or optimizer update ran. A
caller supplies an independently reconstructed model and objective callback;
the kernel restores the sealed pre-state, recomputes the objective and every
gradient tensor, reconstructs the frozen Adam optimizer, and requires exact
post-state identity.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Any, Never, cast

from core.learning.recurrent_grpo import (
    build_recurrent_policy_optimizer,
    recurrent_policy_optimizer_config,
    recurrent_policy_tensor_map_sha256,
)

POLICY_STATE_REPLAY_RECEIPT_SCHEMA = "aura.verified_transition.policy_state_replay_receipt.v1"
TENSOR_MAP_IDENTITY_SCHEMA = "aura.tensor_map_identity.v1"

_TENSOR_IDENTITY_KEYS = frozenset(
    {
        "schema",
        "role",
        "tensor_count",
        "tensor_keys_sha256",
        "tensors",
        "tensor_root_sha256",
    }
)
_TENSOR_ENTRY_KEYS = frozenset({"name", "dtype", "shape", "value_sha256"})
_REPLAY_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "execution_spec_sha256",
        "policy_before_sha256",
        "policy_after_sha256",
        "objective_receipt_sha256",
        "optimizer_config",
        "optimizer_config_sha256",
        "pre_adapter_identity",
        "pre_optimizer_identity",
        "gradient_identity",
        "post_adapter_identity",
        "post_optimizer_identity",
        "optimizer_update_count",
        "objective_recomputed",
        "all_gradient_tensors_recomputed",
        "adapter_post_state_exact",
        "optimizer_post_state_exact",
        "external_policy_state_replayed",
        "receipt_sha256",
    }
)


class VerifiedTransitionPolicyStateReplayError(RuntimeError):
    """The independently replayed transition differs from producer evidence."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise VerifiedTransitionPolicyStateReplayError(code)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (
        TypeError,
        ValueError,
        UnicodeError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise VerifiedTransitionPolicyStateReplayError(
            "policy_state_replay_document_not_canonical"
        ) from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"policy_state_replay_{role}_invalid")
    return value


def _clone(value: Any, *, role: str) -> Any:
    try:
        return json.loads(_canonical_json_bytes(value))
    except (
        TypeError,
        ValueError,
        UnicodeError,
        OverflowError,
        RecursionError,
    ):
        _fail(f"policy_state_replay_{role}_not_canonical")


def _flat_tensor_map(value: Any, *, role: str) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or not value
        or any(not isinstance(key, str) or not key for key in value)
        or any(isinstance(tensor, Mapping) for tensor in value.values())
    ):
        _fail(f"policy_state_replay_{role}_tensor_map_invalid")
    result = dict(value)
    try:
        import mlx.core as mx

        mx.eval(*result.values())
        for tensor in result.values():
            tuple(tensor.shape)
            str(tensor.dtype)
    except Exception as exc:
        raise VerifiedTransitionPolicyStateReplayError(
            f"policy_state_replay_{role}_tensor_map_invalid"
        ) from exc
    return result


def _tensor_payload(value: Any) -> bytes:
    import mlx.core as mx
    import numpy as np

    try:
        array = np.asarray(value)
    except RuntimeError:
        array = np.asarray(value.astype(mx.float32))
    return array.tobytes(order="C")


def tensor_map_identity(
    tensors: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    """Bind every tensor's name, shape, dtype, and exact evaluated value."""

    normalized = _flat_tensor_map(tensors, role=role)
    entries = []
    for name in sorted(normalized):
        tensor = normalized[name]
        entries.append(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "value_sha256": hashlib.sha256(_tensor_payload(tensor)).hexdigest(),
            }
        )
    body = {
        "schema": TENSOR_MAP_IDENTITY_SCHEMA,
        "role": role,
        "tensor_count": len(entries),
        "tensor_keys_sha256": _digest([entry["name"] for entry in entries]),
        "tensors": entries,
    }
    return {**body, "tensor_root_sha256": _digest(body)}


def _validate_tensor_map_identity(value: Any, *, role: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TENSOR_IDENTITY_KEYS:
        _fail(f"policy_state_replay_{role}_identity_schema_invalid")
    identity = cast(
        dict[str, Any],
        _clone(value, role=f"{role}_identity"),
    )
    entries = identity.get("tensors")
    if (
        identity.get("schema") != TENSOR_MAP_IDENTITY_SCHEMA
        or identity.get("role") != role
        or type(identity.get("tensor_count")) is not int
        or identity["tensor_count"] <= 0
        or not isinstance(entries, list)
        or len(entries) != identity["tensor_count"]
    ):
        _fail(f"policy_state_replay_{role}_identity_invalid")
    names = []
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != _TENSOR_ENTRY_KEYS:
            _fail(f"policy_state_replay_{role}_tensor_entry_invalid")
        name = entry.get("name")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(dtype, str)
            or not dtype
            or not isinstance(shape, list)
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
        ):
            _fail(f"policy_state_replay_{role}_tensor_entry_invalid")
        _sha256(
            entry.get("value_sha256"),
            role=f"{role}_tensor_value",
        )
        names.append(name)
    if names != sorted(set(names)) or identity.get("tensor_keys_sha256") != _digest(names):
        _fail(f"policy_state_replay_{role}_tensor_inventory_invalid")
    unsigned = dict(identity)
    root = unsigned.pop("tensor_root_sha256")
    if root != _digest(unsigned):
        _fail(f"policy_state_replay_{role}_identity_digest_mismatch")
    return identity


def validate_policy_state_replay_receipt(value: Any) -> dict[str, Any]:
    """Strictly reconstruct a positive external policy-state replay receipt."""

    if not isinstance(value, Mapping) or set(value) != _REPLAY_RECEIPT_KEYS:
        _fail("policy_state_replay_receipt_schema_invalid")
    receipt = cast(dict[str, Any], _clone(value, role="receipt"))
    unsigned = dict(receipt)
    observed = _sha256(
        unsigned.pop("receipt_sha256"),
        role="receipt",
    )
    if observed != _digest(unsigned):
        _fail("policy_state_replay_receipt_digest_mismatch")
    for field in (
        "execution_spec_sha256",
        "policy_before_sha256",
        "policy_after_sha256",
        "objective_receipt_sha256",
        "optimizer_config_sha256",
    ):
        _sha256(receipt.get(field), role=field)
    optimizer_config = receipt.get("optimizer_config")
    if not isinstance(optimizer_config, Mapping):
        _fail("policy_state_replay_optimizer_config_invalid")
    _optimizer_learning_rate(optimizer_config)
    if receipt["optimizer_config_sha256"] != _digest(dict(optimizer_config)):
        _fail("policy_state_replay_optimizer_config_digest_mismatch")
    identities = {
        role: _validate_tensor_map_identity(receipt.get(f"{role}_identity"), role=role)
        for role in (
            "pre_adapter",
            "pre_optimizer",
            "gradient",
            "post_adapter",
            "post_optimizer",
        )
    }
    by_name = {
        role: {entry["name"]: entry for entry in identity["tensors"]}
        for role, identity in identities.items()
    }
    adapter_names = set(by_name["pre_adapter"])
    optimizer_names = set(by_name["pre_optimizer"])
    if (
        receipt.get("schema") != POLICY_STATE_REPLAY_RECEIPT_SCHEMA
        or receipt["policy_before_sha256"] == receipt["policy_after_sha256"]
        or set(by_name["post_adapter"]) != adapter_names
        or set(by_name["gradient"]) != adapter_names
        or set(by_name["post_optimizer"]) != optimizer_names
        or any(
            by_name["gradient"][name][field] != by_name["pre_adapter"][name][field]
            for name in adapter_names
            for field in ("dtype", "shape")
        )
        or receipt.get("optimizer_update_count") != 1
        or receipt.get("objective_recomputed") is not True
        or receipt.get("all_gradient_tensors_recomputed") is not True
        or receipt.get("adapter_post_state_exact") is not True
        or receipt.get("optimizer_post_state_exact") is not True
        or receipt.get("external_policy_state_replayed") is not True
    ):
        _fail("policy_state_replay_receipt_invalid")
    return receipt


def _tensor_maps_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_map = _flat_tensor_map(left, role="comparison_left")
    right_map = _flat_tensor_map(right, role="comparison_right")
    if set(left_map) != set(right_map):
        return False
    try:
        import mlx.core as mx

        comparisons = []
        for key in sorted(left_map):
            if tuple(left_map[key].shape) != tuple(right_map[key].shape) or str(
                left_map[key].dtype
            ) != str(right_map[key].dtype):
                return False
            comparisons.append(mx.array_equal(left_map[key], right_map[key]))
        mx.eval(*comparisons)
        return all(bool(value) for value in comparisons)
    except Exception:
        return False


def _require_same_layout(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    role: str,
) -> None:
    observed_map = _flat_tensor_map(observed, role=f"{role}_observed")
    expected_map = _flat_tensor_map(expected, role=f"{role}_expected")
    if set(observed_map) != set(expected_map):
        _fail(f"policy_state_replay_{role}_keyset_mismatch")
    for key in sorted(observed_map):
        if tuple(observed_map[key].shape) != tuple(expected_map[key].shape) or str(
            observed_map[key].dtype
        ) != str(expected_map[key].dtype):
            _fail(f"policy_state_replay_{role}_layout_mismatch")


def _optimizer_learning_rate(config: Mapping[str, Any]) -> float:
    if not isinstance(config, Mapping):
        _fail("policy_state_replay_optimizer_config_invalid")
    try:
        learning_rate = float.fromhex(cast(str, config.get("learning_rate_hex")))
    except (TypeError, ValueError):
        _fail("policy_state_replay_optimizer_config_invalid")
    expected = recurrent_policy_optimizer_config(learning_rate)
    if dict(config) != expected:
        _fail("policy_state_replay_optimizer_config_mismatch")
    return learning_rate


def _objective_material(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_method = getattr(value, "receipt", None)
    gradients = getattr(value, "gradients", None)
    if not callable(receipt_method) or gradients is None:
        _fail("policy_state_replay_objective_result_invalid")
    try:
        receipt = _clone(receipt_method(), role="objective_receipt")
        from mlx.utils import tree_flatten

        flat_gradients = dict(tree_flatten(gradients))
    except Exception as exc:
        raise VerifiedTransitionPolicyStateReplayError(
            "policy_state_replay_objective_result_invalid"
        ) from exc
    return cast(dict[str, Any], receipt), _flat_tensor_map(
        flat_gradients,
        role="gradient",
    )


def replay_verified_policy_transition(
    *,
    model: Any,
    pre_adapter_tensors: Mapping[str, Any],
    pre_optimizer_tensors: Mapping[str, Any],
    expected_post_adapter_tensors: Mapping[str, Any],
    expected_post_optimizer_tensors: Mapping[str, Any],
    expected_objective_receipt: Mapping[str, Any],
    objective_factory: Callable[[Any], Any],
    optimizer_config: Mapping[str, Any],
    execution_spec_sha256: str,
    expected_policy_before_sha256: str,
    expected_policy_after_sha256: str,
) -> dict[str, Any]:
    """Recompute and exactly replay one producer transition in isolation."""

    from mlx.utils import tree_flatten, tree_unflatten

    execution_spec = _sha256(
        execution_spec_sha256,
        role="execution_spec",
    )
    policy_before = _sha256(
        expected_policy_before_sha256,
        role="policy_before",
    )
    policy_after = _sha256(
        expected_policy_after_sha256,
        role="policy_after",
    )
    if policy_before == policy_after:
        _fail("policy_state_replay_policy_unchanged")
    if not callable(objective_factory):
        _fail("policy_state_replay_objective_factory_invalid")

    pre_adapter = _flat_tensor_map(pre_adapter_tensors, role="pre_adapter")
    pre_optimizer = _flat_tensor_map(
        pre_optimizer_tensors,
        role="pre_optimizer",
    )
    expected_post_adapter = _flat_tensor_map(
        expected_post_adapter_tensors,
        role="expected_post_adapter",
    )
    expected_post_optimizer = _flat_tensor_map(
        expected_post_optimizer_tensors,
        role="expected_post_optimizer",
    )
    if recurrent_policy_tensor_map_sha256(pre_adapter, execution_spec) != policy_before:
        _fail("policy_state_replay_pre_policy_digest_mismatch")
    if recurrent_policy_tensor_map_sha256(expected_post_adapter, execution_spec) != policy_after:
        _fail("policy_state_replay_post_policy_digest_mismatch")

    try:
        import mlx.core as mx

        current_adapter = dict(tree_flatten(model.trainable_parameters()))
        _require_same_layout(current_adapter, pre_adapter, role="adapter")
        model.load_weights(list(pre_adapter.items()), strict=False)
        mx.eval(model.trainable_parameters())
        restored_adapter = dict(tree_flatten(model.trainable_parameters()))
    except VerifiedTransitionPolicyStateReplayError:
        raise
    except Exception as exc:
        raise VerifiedTransitionPolicyStateReplayError(
            "policy_state_replay_adapter_restore_failed"
        ) from exc
    if not _tensor_maps_equal(restored_adapter, pre_adapter):
        _fail("policy_state_replay_pre_adapter_restore_mismatch")

    learning_rate = _optimizer_learning_rate(optimizer_config)
    try:
        optimizer = build_recurrent_policy_optimizer(learning_rate)
        optimizer.init(model.trainable_parameters())
        initialized_optimizer = dict(tree_flatten(optimizer.state))
        _require_same_layout(
            initialized_optimizer,
            pre_optimizer,
            role="optimizer",
        )
        restored_optimizer = tree_unflatten(pre_optimizer)
        if not isinstance(restored_optimizer, dict):
            _fail("policy_state_replay_optimizer_tree_invalid")
        optimizer.state = restored_optimizer
        optimizer.init(model.trainable_parameters())
        mx.eval(optimizer.state)
        observed_pre_optimizer = dict(tree_flatten(optimizer.state))
    except VerifiedTransitionPolicyStateReplayError:
        raise
    except Exception as exc:
        raise VerifiedTransitionPolicyStateReplayError(
            "policy_state_replay_optimizer_restore_failed"
        ) from exc
    if not _tensor_maps_equal(observed_pre_optimizer, pre_optimizer):
        _fail("policy_state_replay_pre_optimizer_restore_mismatch")

    expected_receipt = cast(
        dict[str, Any],
        _clone(expected_objective_receipt, role="expected_objective_receipt"),
    )
    objective_receipt, gradients = _objective_material(objective_factory(model))
    if objective_receipt != expected_receipt:
        _fail("policy_state_replay_objective_receipt_mismatch")
    observed_after_objective = dict(tree_flatten(model.trainable_parameters()))
    if not _tensor_maps_equal(observed_after_objective, pre_adapter):
        _fail("policy_state_replay_objective_mutated_policy")
    _require_same_layout(gradients, pre_adapter, role="gradient")
    finite = [mx.all(mx.isfinite(value)) for value in gradients.values()]
    mx.eval(*finite)
    if not all(bool(value) for value in finite):
        _fail("policy_state_replay_gradient_nonfinite")

    try:
        optimizer.update(model, gradients)
        mx.eval(model.trainable_parameters(), optimizer.state)
        observed_post_adapter = dict(tree_flatten(model.trainable_parameters()))
        observed_post_optimizer = dict(tree_flatten(optimizer.state))
    except Exception as exc:
        raise VerifiedTransitionPolicyStateReplayError(
            "policy_state_replay_optimizer_update_failed"
        ) from exc
    if not _tensor_maps_equal(observed_post_adapter, expected_post_adapter):
        _fail("policy_state_replay_post_adapter_mismatch")
    if not _tensor_maps_equal(observed_post_optimizer, expected_post_optimizer):
        _fail("policy_state_replay_post_optimizer_mismatch")

    pre_adapter_identity = tensor_map_identity(pre_adapter, role="pre_adapter")
    pre_optimizer_identity = tensor_map_identity(
        pre_optimizer,
        role="pre_optimizer",
    )
    gradient_identity = tensor_map_identity(gradients, role="gradient")
    post_adapter_identity = tensor_map_identity(
        observed_post_adapter,
        role="post_adapter",
    )
    post_optimizer_identity = tensor_map_identity(
        observed_post_optimizer,
        role="post_optimizer",
    )
    body = {
        "schema": POLICY_STATE_REPLAY_RECEIPT_SCHEMA,
        "execution_spec_sha256": execution_spec,
        "policy_before_sha256": policy_before,
        "policy_after_sha256": policy_after,
        "objective_receipt_sha256": _digest(expected_receipt),
        "optimizer_config": dict(optimizer_config),
        "optimizer_config_sha256": _digest(dict(optimizer_config)),
        "pre_adapter_identity": pre_adapter_identity,
        "pre_optimizer_identity": pre_optimizer_identity,
        "gradient_identity": gradient_identity,
        "post_adapter_identity": post_adapter_identity,
        "post_optimizer_identity": post_optimizer_identity,
        "optimizer_update_count": 1,
        "objective_recomputed": True,
        "all_gradient_tensors_recomputed": True,
        "adapter_post_state_exact": True,
        "optimizer_post_state_exact": True,
        "external_policy_state_replayed": True,
    }
    return validate_policy_state_replay_receipt({**body, "receipt_sha256": _digest(body)})


__all__ = [
    "POLICY_STATE_REPLAY_RECEIPT_SCHEMA",
    "TENSOR_MAP_IDENTITY_SCHEMA",
    "VerifiedTransitionPolicyStateReplayError",
    "replay_verified_policy_transition",
    "tensor_map_identity",
    "validate_policy_state_replay_receipt",
]
