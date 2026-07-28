"""Source, model, and tensor-state custody for external policy replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
    full_weight_checkpoint_identity,
    model_behavior_bundle_identity,
)
from core.learning.recurrent_grpo import (
    ExactAdjointInterventionConfig,
    RecurrentGRPOConfig,
    VerifiedTrajectoryGroupConfig,
    recurrent_policy_optimizer_config,
)
from core.learning.verified_transition_episode import canonical_json_bytes
from core.learning.verified_transition_measurement_chain import (
    recurrent_grpo_config_contract,
)
from core.learning.verified_transition_policy_probe import (
    build_initial_policy_state_custody,
)
from core.learning.verified_transition_policy_state_replay import (
    POLICY_STATE_REPLAY_CONTRACT_SCHEMA,
    VerifiedTransitionPolicyStateReplayError,
    build_policy_state_replay_contract,
    validate_policy_state_replay_contract,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _write(path: Path, payload: bytes, *, private: bool = False) -> Path:
    path.write_bytes(payload)
    if private:
        path.chmod(0o600)
    return path.resolve(strict=True)


def _custody(
    root: Path,
    *,
    execution_spec_sha256: str,
    initial_policy_sha256: str,
) -> dict[str, Any]:
    adapter_path = _write(root / "initial-adapter.safetensors", b"sealed-adapter", private=True)
    optimizer_path = _write(
        root / "initial-optimizer.safetensors",
        b"sealed-optimizer",
        private=True,
    )
    adapter_keys = ["layer.lora_a", "layer.lora_b"]
    optimizer_keys = [
        "layer.lora_a.m",
        "layer.lora_a.v",
        "layer.lora_b.m",
        "layer.lora_b.v",
        "learning_rate",
        "step",
    ]
    adapter_artifact = {
        "path": adapter_path.name,
        "sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        "size_bytes": adapter_path.stat().st_size,
        "tensor_count": len(adapter_keys),
        "tensor_keys": adapter_keys,
        "tensor_keys_sha256": hashlib.sha256(_canonical(adapter_keys)).hexdigest(),
        "policy_sha256": initial_policy_sha256,
    }
    optimizer_artifact = {
        "path": optimizer_path.name,
        "sha256": hashlib.sha256(optimizer_path.read_bytes()).hexdigest(),
        "size_bytes": optimizer_path.stat().st_size,
        "tensor_count": len(optimizer_keys),
        "tensor_keys": optimizer_keys,
        "tensor_keys_sha256": hashlib.sha256(_canonical(optimizer_keys)).hexdigest(),
    }
    return build_initial_policy_state_custody(
        initial_policy_probe_sha256=_sha("policy-probe"),
        initial_policy_sha256=initial_policy_sha256,
        execution_spec_sha256=execution_spec_sha256,
        adapter_initialization={
            "seed": 17,
            "rank": 8,
            "layers": 8,
            "targets": ["q_proj"],
        },
        optimizer_initialization=recurrent_policy_optimizer_config(1e-5),
        initial_adapter_artifact=adapter_artifact,
        initial_optimizer_artifact=optimizer_artifact,
        initial_adapter_path=adapter_path,
        initial_optimizer_path=optimizer_path,
    )


def _material(root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    model = root / "model"
    model.mkdir()
    _write(model / "weights.safetensors", b"resident-weights")
    for name, document in (
        ("config.json", {"model_type": "qwen2"}),
        ("tokenizer.json", {"version": "1.0"}),
        ("tokenizer_config.json", {"eos_token": "<eos>"}),
    ):
        _write(model / name, _canonical(document))

    spec = RLCExecutionSpec(recurrent_steps=4)
    spec_path = _write(root / "execution-spec.json", _canonical(spec.to_dict()))
    replay_source = _write(root / "replay-source.py", b"REPLAY_VERSION = 1\n")
    objective_source = _write(root / "objective-source.py", b"OBJECTIVE_VERSION = 1\n")
    policy_sha = _sha("initial-policy")
    custody = _custody(
        root,
        execution_spec_sha256=spec.sha256,
        initial_policy_sha256=policy_sha,
    )
    trajectory = VerifiedTrajectoryGroupConfig(
        intervention_config=ExactAdjointInterventionConfig(
            lesion_steps=(1,),
            causality_weight=0.4,
            causality_margin=0.1,
            stopping_steps=(1, 2),
            stopping_weight=0.3,
            stopping_ponder_cost=0.01,
            stopping_temperature=0.2,
        )
    ).to_dict()
    contract = build_policy_state_replay_contract(
        preregistration_contract_sha256=_sha("preregistration"),
        initial_policy_sha256=policy_sha,
        model_path=model,
        base_checkpoint=full_weight_checkpoint_identity(model),
        behavior_bundle=model_behavior_bundle_identity(model),
        execution_spec_path=spec_path,
        execution_spec_document=spec.to_dict(),
        source_bindings={
            "objective": _binding(objective_source),
            "replay": _binding(replay_source),
        },
        initial_policy_state_custody=custody,
        recurrent_grpo_config=recurrent_grpo_config_contract(
            RecurrentGRPOConfig(kl_coefficient=0.02, advantage_clip=4.0)
        ),
        verified_trajectory_config=trajectory,
        external_verifier_max_seconds=21_600,
    )
    return contract, {
        "adapter": Path(custody["initial_adapter_path"]),
        "behavior": model / "config.json",
        "optimizer": Path(custody["initial_optimizer_path"]),
        "source": replay_source,
        "weights": model / "weights.safetensors",
    }


def _reseal(contract: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: value for key, value in contract.items() if key != "contract_sha256"}
    contract["contract_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    return contract


def test_contract_binds_every_replay_input_and_recomputes_tiny_model_identity(
    tmp_path: Path,
) -> None:
    contract, _paths = _material(tmp_path)

    assert contract["schema"] == POLICY_STATE_REPLAY_CONTRACT_SCHEMA
    assert contract["external_verifier_max_seconds"] == 21_600
    assert set(contract["source_bindings"]) == {"objective", "replay"}
    assert canonical_json_bytes(contract)
    assert (
        validate_policy_state_replay_contract(
            contract,
            verify_files=True,
            verify_model=True,
        )
        == contract
    )


@pytest.mark.parametrize("role", ["adapter", "optimizer", "source"])
def test_bound_replay_artifact_byte_replacement_fails_closed(
    tmp_path: Path,
    role: str,
) -> None:
    contract, paths = _material(tmp_path)
    paths[role].chmod(0o600)
    paths[role].write_bytes(b"replaced-with-different-bytes")

    with pytest.raises(
        VerifiedTransitionPolicyStateReplayError,
        match=("artifact_mismatch" if role != "source" else "binding_mismatch"),
    ):
        validate_policy_state_replay_contract(contract, verify_files=True)


@pytest.mark.parametrize("role", ["weights", "behavior"])
def test_full_model_recomputation_detects_bound_model_drift(
    tmp_path: Path,
    role: str,
) -> None:
    contract, paths = _material(tmp_path)
    paths[role].write_bytes(b"model-drift")

    with pytest.raises(
        VerifiedTransitionPolicyStateReplayError,
        match="model_identity_mismatch",
    ):
        validate_policy_state_replay_contract(
            contract,
            verify_files=True,
            verify_model=True,
        )


def test_resealed_cross_binding_drift_cannot_change_initial_policy(
    tmp_path: Path,
) -> None:
    contract, _paths = _material(tmp_path)
    contract["initial_policy_sha256"] = _sha("substituted-policy")

    with pytest.raises(
        VerifiedTransitionPolicyStateReplayError,
        match="state_or_objective_contract_mismatch",
    ):
        validate_policy_state_replay_contract(_reseal(contract))


def test_model_recomputation_requires_stable_file_verification(tmp_path: Path) -> None:
    contract, _paths = _material(tmp_path)

    with pytest.raises(
        VerifiedTransitionPolicyStateReplayError,
        match="model_verification_requires_files",
    ):
        validate_policy_state_replay_contract(contract, verify_model=True)
