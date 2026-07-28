"""Tests for the deterministic initial recurrent-policy probe receipt."""

from __future__ import annotations

import copy
import hashlib

import pytest

from core.brain.llm.latent_cortex.recurrent_grpo_adapter_identity import (
    REQUIRED_SOURCE_ROLES,
)
from core.learning.verified_token_trace import (
    build_tokenizer_bundle_identity,
    tokenizer_file_bindings_from_bytes,
)
from core.learning.verified_transition_episode import canonical_json_bytes
from core.learning.verified_transition_policy_probe import (
    INITIAL_RECURRENT_POLICY_PROBE_SCHEMA_V2,
    InitialRecurrentPolicyProbeError,
    build_initial_policy_state_custody,
    build_initial_recurrent_policy_probe,
    validate_initial_policy_state_custody,
    validate_initial_recurrent_policy_probe,
    validate_initial_recurrent_policy_probe_identity,
)


def _probe() -> dict:
    tokenizer = build_tokenizer_bundle_identity(
        tokenizer_class="tests.IntegerTokenizer",
        tokenizer_files=tokenizer_file_bindings_from_bytes(
            {
                "tokenizer.json": b"tokenizer",
                "tokenizer_config.json": b"config",
            }
        ),
        chat_template=None,
        special_token_map={},
        encode_options={},
        decode_options={},
        implementation_source_sha256="a" * 64,
    )
    return build_initial_recurrent_policy_probe(
        campaign_id="resident-32b-recurrent-grpo-cp420",
        initial_policy_sha256="b" * 64,
        dataset_sha256="c" * 64,
        execution_spec_sha256="d" * 64,
        base_checkpoint={"sha256": "e" * 64},
        model_behavior_bundle={"sha256": "f" * 64},
        tokenizer_bundle=tokenizer,
        adapter_initialization={
            "seed": 17,
            "rank": 8,
            "layers": 8,
            "targets": ["o_proj", "v_proj", "q_proj"],
        },
        source_bindings={
            role: {
                "path": "tools/train_grpo.py",
                "sha256": "1" * 64,
                "size_bytes": 10,
            }
            for role in REQUIRED_SOURCE_ROLES
        },
        created_at_unix_ns=1_900_000_000_000_000_000,
    )


def _optimizer_config() -> dict:
    return {
        "class_name": "mlx.optimizers.Adam",
        "learning_rate_hex": (1e-5).hex(),
        "betas_hex": [(0.9).hex(), (0.999).hex()],
        "eps_hex": (1e-8).hex(),
        "bias_correction": False,
    }


def _adapter_keys() -> list[str]:
    return [
        "model.layers.1.self_attn.o_proj.lora_a",
        "model.layers.1.self_attn.o_proj.lora_b",
        "model.layers.1.self_attn.q_proj.lora_a",
        "model.layers.1.self_attn.q_proj.lora_b",
    ]


def _adapter_artifact(policy_sha256: str) -> dict:
    keys = _adapter_keys()
    return {
        "path": "initial_adapter.safetensors",
        "sha256": "2" * 64,
        "size_bytes": 1024,
        "tensor_count": len(keys),
        "tensor_keys": keys,
        "tensor_keys_sha256": hashlib.sha256(
            canonical_json_bytes(keys)
        ).hexdigest(),
        "policy_sha256": policy_sha256,
    }


def _optimizer_artifact() -> dict:
    keys = sorted(
        [
            "step",
            "learning_rate",
            *[
                f"{key}.{moment}"
                for key in _adapter_keys()
                for moment in ("m", "v")
            ],
        ]
    )
    return {
        "path": "initial_optimizer.safetensors",
        "sha256": "5" * 64,
        "size_bytes": 2048,
        "tensor_count": len(keys),
        "tensor_keys": keys,
        "tensor_keys_sha256": hashlib.sha256(
            canonical_json_bytes(keys)
        ).hexdigest(),
    }


def test_policy_probe_round_trips_exactly() -> None:
    probe = _probe()
    assert validate_initial_recurrent_policy_probe(probe) == probe


def test_policy_probe_v2_binds_custodied_initial_adapter() -> None:
    probe = _probe()
    identity = {
        key: probe[key]
        for key in (
            "campaign_id",
            "initial_policy_sha256",
            "dataset_sha256",
            "execution_spec_sha256",
            "base_checkpoint",
            "model_behavior_bundle",
            "tokenizer_bundle",
            "adapter_initialization",
            "source_bindings",
        )
    }
    artifact = _adapter_artifact(probe["initial_policy_sha256"])

    upgraded = build_initial_recurrent_policy_probe(
        **identity,
        initial_adapter_artifact=artifact,
        optimizer_initialization=_optimizer_config(),
        initial_optimizer_artifact=_optimizer_artifact(),
        created_at_unix_ns=probe["created_at_unix_ns"],
    )

    assert upgraded["schema"] == INITIAL_RECURRENT_POLICY_PROBE_SCHEMA_V2
    assert upgraded["initial_adapter_artifact"] == artifact
    assert validate_initial_recurrent_policy_probe(upgraded) == upgraded
    assert (
        validate_initial_recurrent_policy_probe_identity(
            upgraded,
            **identity,
            initial_adapter_artifact=artifact,
            optimizer_initialization=_optimizer_config(),
            initial_optimizer_artifact=_optimizer_artifact(),
        )
        == upgraded
    )


def test_policy_probe_v2_rejects_adapter_policy_substitution() -> None:
    probe = _probe()
    identity = {
        key: probe[key]
        for key in (
            "campaign_id",
            "initial_policy_sha256",
            "dataset_sha256",
            "execution_spec_sha256",
            "base_checkpoint",
            "model_behavior_bundle",
            "tokenizer_bundle",
            "adapter_initialization",
            "source_bindings",
        )
    }
    artifact = _adapter_artifact("4" * 64)

    with pytest.raises(
        InitialRecurrentPolicyProbeError,
        match="initial_policy_probe_adapter_policy_mismatch",
    ):
        build_initial_recurrent_policy_probe(
            **identity,
            initial_adapter_artifact=artifact,
            optimizer_initialization=_optimizer_config(),
            initial_optimizer_artifact=_optimizer_artifact(),
            created_at_unix_ns=probe["created_at_unix_ns"],
        )


def test_policy_probe_v2_rejects_resealed_optimizer_topology_gap() -> None:
    probe = _probe()
    identity = {
        key: probe[key]
        for key in (
            "campaign_id",
            "initial_policy_sha256",
            "dataset_sha256",
            "execution_spec_sha256",
            "base_checkpoint",
            "model_behavior_bundle",
            "tokenizer_bundle",
            "adapter_initialization",
            "source_bindings",
        )
    }
    optimizer_artifact = _optimizer_artifact()
    optimizer_artifact["tensor_keys"].remove(
        "model.layers.1.self_attn.q_proj.lora_b.v"
    )
    optimizer_artifact["tensor_count"] = len(
        optimizer_artifact["tensor_keys"]
    )
    optimizer_artifact["tensor_keys_sha256"] = hashlib.sha256(
        canonical_json_bytes(optimizer_artifact["tensor_keys"])
    ).hexdigest()

    with pytest.raises(
        InitialRecurrentPolicyProbeError,
        match="initial_policy_probe_optimizer_adapter_topology_mismatch",
    ):
        build_initial_recurrent_policy_probe(
            **identity,
            initial_adapter_artifact=_adapter_artifact(
                probe["initial_policy_sha256"]
            ),
            optimizer_initialization=_optimizer_config(),
            initial_optimizer_artifact=optimizer_artifact,
            created_at_unix_ns=probe["created_at_unix_ns"],
        )


def test_policy_probe_v2_rejects_noncanonical_optimizer_float_encoding() -> None:
    probe = _probe()
    optimizer_config = _optimizer_config()
    optimizer_config["learning_rate_hex"] = "0x1.4f8b588e368f10p-17"

    with pytest.raises(
        InitialRecurrentPolicyProbeError,
        match="initial_policy_probe_optimizer_initialization_invalid",
    ):
        build_initial_recurrent_policy_probe(
            **{
                key: probe[key]
                for key in (
                    "campaign_id",
                    "initial_policy_sha256",
                    "dataset_sha256",
                    "execution_spec_sha256",
                    "base_checkpoint",
                    "model_behavior_bundle",
                    "tokenizer_bundle",
                    "adapter_initialization",
                    "source_bindings",
                )
            },
            initial_adapter_artifact=_adapter_artifact(
                probe["initial_policy_sha256"]
            ),
            optimizer_initialization=optimizer_config,
            initial_optimizer_artifact=_optimizer_artifact(),
            created_at_unix_ns=probe["created_at_unix_ns"],
        )


def test_materialized_policy_state_custody_cross_binds_probe_and_artifact(
    tmp_path,
) -> None:
    snapshot = (tmp_path / "initial_adapter.safetensors").resolve()
    optimizer_snapshot = (
        tmp_path / "initial_optimizer.safetensors"
    ).resolve()
    artifact = {
        **_adapter_artifact("b" * 64),
        "path": snapshot.name,
    }

    custody = build_initial_policy_state_custody(
        initial_policy_probe_sha256="1" * 64,
        initial_policy_sha256="b" * 64,
        execution_spec_sha256="d" * 64,
        adapter_initialization={
            "seed": 17,
            "rank": 8,
            "layers": 8,
            "targets": ["q_proj"],
        },
        optimizer_initialization=_optimizer_config(),
        initial_adapter_artifact=artifact,
        initial_optimizer_artifact=_optimizer_artifact(),
        initial_adapter_path=snapshot,
        initial_optimizer_path=optimizer_snapshot,
    )

    assert validate_initial_policy_state_custody(custody) == custody
    drifted = copy.deepcopy(custody)
    drifted["initial_adapter_path"] = str(
        snapshot.with_name("substituted.safetensors")
    )
    with pytest.raises(
        InitialRecurrentPolicyProbeError,
        match="initial_policy_state_custody_invalid",
    ):
        validate_initial_policy_state_custody(drifted)


@pytest.mark.parametrize(
    "path,value",
    [
        (("initial_policy_sha256",), "0" * 64),
        (("adapter_initialization", "seed"), 18),
        (("source_bindings", "trainer", "size_bytes"), 11),
    ],
)
def test_policy_probe_rejects_resealed_state_substitution(
    path: tuple[str, ...],
    value: object,
) -> None:
    probe = copy.deepcopy(_probe())
    target = probe
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(
        InitialRecurrentPolicyProbeError,
        match="initial_policy_probe_invalid",
    ):
        validate_initial_recurrent_policy_probe(probe)


def test_policy_probe_rejects_incomplete_source_closure() -> None:
    probe = copy.deepcopy(_probe())
    probe["source_bindings"].pop("transition_provider")
    unsigned = dict(probe)
    unsigned.pop("receipt_sha256")
    probe["receipt_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()

    with pytest.raises(
        InitialRecurrentPolicyProbeError,
        match="initial_policy_probe_invalid",
    ):
        validate_initial_recurrent_policy_probe(probe)


def test_policy_probe_identity_reuses_original_sealed_time() -> None:
    probe = _probe()
    identity = {
        key: probe[key]
        for key in (
            "campaign_id",
            "initial_policy_sha256",
            "dataset_sha256",
            "execution_spec_sha256",
            "base_checkpoint",
            "model_behavior_bundle",
            "tokenizer_bundle",
            "adapter_initialization",
            "source_bindings",
        )
    }

    assert (
        validate_initial_recurrent_policy_probe_identity(
            probe,
            **identity,
        )
        == probe
    )


def test_policy_probe_identity_rejects_runtime_drift() -> None:
    probe = _probe()
    identity = {
        key: probe[key]
        for key in (
            "campaign_id",
            "initial_policy_sha256",
            "dataset_sha256",
            "execution_spec_sha256",
            "base_checkpoint",
            "model_behavior_bundle",
            "tokenizer_bundle",
            "adapter_initialization",
            "source_bindings",
        )
    }
    identity["initial_policy_sha256"] = "f" * 64

    with pytest.raises(
        InitialRecurrentPolicyProbeError,
        match="initial_policy_probe_identity_mismatch",
    ):
        validate_initial_recurrent_policy_probe_identity(
            probe,
            **identity,
        )
