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
    InitialRecurrentPolicyProbeError,
    build_initial_recurrent_policy_probe,
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


def test_policy_probe_round_trips_exactly() -> None:
    probe = _probe()
    assert validate_initial_recurrent_policy_probe(probe) == probe


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
