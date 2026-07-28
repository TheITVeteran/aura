"""Sealed identity of a deterministic recurrent policy before training."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Never, cast

from core.brain.llm.latent_cortex.recurrent_grpo_adapter_identity import (
    REQUIRED_SOURCE_ROLES,
)
from core.learning.verified_token_trace import validate_tokenizer_bundle_identity
from core.learning.verified_transition_episode import canonical_json_bytes

INITIAL_RECURRENT_POLICY_PROBE_SCHEMA = (
    "aura.verified_transition.initial_recurrent_policy_probe.v1"
)
_PROBE_KEYS = frozenset(
    {
        "schema",
        "campaign_id",
        "initial_policy_sha256",
        "dataset_sha256",
        "execution_spec_sha256",
        "base_checkpoint",
        "model_behavior_bundle",
        "tokenizer_bundle",
        "adapter_initialization",
        "source_bindings",
        "created_at_unix_ns",
        "receipt_sha256",
    }
)
_ADAPTER_KEYS = frozenset(
    {
        "seed",
        "rank",
        "layers",
        "targets",
    }
)


class InitialRecurrentPolicyProbeError(RuntimeError):
    """Stable probe-construction or validation failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise InitialRecurrentPolicyProbeError(code)


def _clone(value: Any) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError):
        _fail("initial_policy_probe_not_canonical_json")


def _sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"initial_policy_probe_{role}_invalid")
    return value


def _identifier(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
    ):
        _fail(f"initial_policy_probe_{role}_invalid")
    return value


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def build_initial_recurrent_policy_probe(
    *,
    campaign_id: str,
    initial_policy_sha256: str,
    dataset_sha256: str,
    execution_spec_sha256: str,
    base_checkpoint: Mapping[str, Any],
    model_behavior_bundle: Mapping[str, Any],
    tokenizer_bundle: Mapping[str, Any],
    adapter_initialization: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
    created_at_unix_ns: int,
) -> dict[str, Any]:
    body = {
        "schema": INITIAL_RECURRENT_POLICY_PROBE_SCHEMA,
        "campaign_id": campaign_id,
        "initial_policy_sha256": initial_policy_sha256,
        "dataset_sha256": dataset_sha256,
        "execution_spec_sha256": execution_spec_sha256,
        "base_checkpoint": dict(base_checkpoint),
        "model_behavior_bundle": dict(model_behavior_bundle),
        "tokenizer_bundle": dict(tokenizer_bundle),
        "adapter_initialization": dict(adapter_initialization),
        "source_bindings": dict(source_bindings),
        "created_at_unix_ns": created_at_unix_ns,
    }
    return validate_initial_recurrent_policy_probe(
        {**body, "receipt_sha256": _digest(body)}
    )


def validate_initial_recurrent_policy_probe(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _PROBE_KEYS:
        _fail("initial_policy_probe_schema_invalid")
    document = cast(dict[str, Any], _clone(value))
    unsigned = dict(document)
    receipt = unsigned.pop("receipt_sha256")
    adapter = document.get("adapter_initialization")
    sources = document.get("source_bindings")
    if (
        document.get("schema") != INITIAL_RECURRENT_POLICY_PROBE_SCHEMA
        or receipt != _digest(unsigned)
        or not isinstance(document.get("base_checkpoint"), Mapping)
        or not isinstance(document.get("model_behavior_bundle"), Mapping)
        or not isinstance(adapter, Mapping)
        or set(adapter) != _ADAPTER_KEYS
        or type(adapter.get("seed")) is not int
        or not 0 <= adapter["seed"] <= 0xFFFFFFFF
        or type(adapter.get("rank")) is not int
        or adapter["rank"] <= 0
        or type(adapter.get("layers")) is not int
        or adapter["layers"] <= 0
        or not isinstance(adapter.get("targets"), list)
        or not adapter["targets"]
        or any(not isinstance(item, str) or not item for item in adapter["targets"])
        or not isinstance(sources, Mapping)
        or set(sources) != REQUIRED_SOURCE_ROLES
        or type(document.get("created_at_unix_ns")) is not int
        or document["created_at_unix_ns"] <= 0
    ):
        _fail("initial_policy_probe_invalid")
    _identifier(document.get("campaign_id"), role="campaign_id")
    for role in (
        "initial_policy_sha256",
        "dataset_sha256",
        "execution_spec_sha256",
        "receipt_sha256",
    ):
        _sha256(document.get(role), role=role)
    validate_tokenizer_bundle_identity(document.get("tokenizer_bundle"))
    for role, binding in sources.items():
        _identifier(role, role="source_role")
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256", "size_bytes"}
            or not isinstance(binding.get("path"), str)
            or type(binding.get("size_bytes")) is not int
            or binding["size_bytes"] <= 0
        ):
            _fail("initial_policy_probe_source_binding_invalid")
        _sha256(binding.get("sha256"), role="source_binding_sha256")
    return document


def validate_initial_recurrent_policy_probe_identity(
    value: Any,
    *,
    campaign_id: str,
    initial_policy_sha256: str,
    dataset_sha256: str,
    execution_spec_sha256: str,
    base_checkpoint: Mapping[str, Any],
    model_behavior_bundle: Mapping[str, Any],
    tokenizer_bundle: Mapping[str, Any],
    adapter_initialization: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild a sealed probe at its original time and require exact identity."""

    document = validate_initial_recurrent_policy_probe(value)
    expected = build_initial_recurrent_policy_probe(
        campaign_id=campaign_id,
        initial_policy_sha256=initial_policy_sha256,
        dataset_sha256=dataset_sha256,
        execution_spec_sha256=execution_spec_sha256,
        base_checkpoint=base_checkpoint,
        model_behavior_bundle=model_behavior_bundle,
        tokenizer_bundle=tokenizer_bundle,
        adapter_initialization=adapter_initialization,
        source_bindings=source_bindings,
        created_at_unix_ns=document["created_at_unix_ns"],
    )
    if document != expected:
        _fail("initial_policy_probe_identity_mismatch")
    return document


__all__ = [
    "INITIAL_RECURRENT_POLICY_PROBE_SCHEMA",
    "InitialRecurrentPolicyProbeError",
    "build_initial_recurrent_policy_probe",
    "validate_initial_recurrent_policy_probe",
    "validate_initial_recurrent_policy_probe_identity",
]
