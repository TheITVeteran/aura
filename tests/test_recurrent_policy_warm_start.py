from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import pytest

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.learning.recurrent_policy_warm_start import (
    RecurrentPolicyWarmStartError,
    apply_recurrent_warm_start,
    build_recurrent_warm_start_contract,
    load_recurrent_warm_start_contract,
    plan_recurrent_warm_start,
    validate_recurrent_warm_start_contract,
    validate_recurrent_warm_start_receipt,
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _tensor_key(layer: int, target: str, factor: str) -> str:
    return f"model.layers.{layer}.self_attn.{target}.lora_{factor}"


def _source_tensors() -> dict[str, mx.array]:
    tensors = {}
    for layer in (1, 2):
        for target in ("o_proj", "v_proj"):
            tensors[_tensor_key(layer, target, "a")] = mx.full(
                (3, 2),
                float(layer),
            )
            tensors[_tensor_key(layer, target, "b")] = mx.full(
                (2, 3),
                float(layer + 10),
            )
    return tensors


def _current_tensors() -> dict[str, mx.array]:
    tensors = {}
    for target in ("o_proj", "q_proj", "v_proj"):
        tensors[_tensor_key(2, target, "a")] = mx.zeros((3, 2))
        tensors[_tensor_key(2, target, "b")] = mx.zeros((2, 3))
    return tensors


def _write_source(
    root: Path,
    *,
    step: int = 5,
    max_steps: int = 10,
) -> tuple[Path, Path, Path]:
    checkpoint = root / "artifacts" / "source" / "checkpoint"
    checkpoint.mkdir(parents=True)
    adapter = checkpoint / "adapter.safetensors"
    mx.save_safetensors(str(adapter), _source_tensors())

    source_spec = root / "artifacts" / "source" / "execution_spec.json"
    source_spec_payload = {
        "schema": "aura.rlc_execution_spec.v1",
        "n_slots": 4,
        "decode_bridge_policy": "assistant_answer",
    }
    source_spec.write_bytes(canonical_json_bytes(source_spec_payload))
    spec_sha = _sha(canonical_json_bytes(source_spec_payload))

    training = root / "artifacts" / "source" / "training_config.json"
    training_payload = {
        "schema": "aura.recurrence_native_training_config.v2",
        "max_steps": max_steps,
        "execution_spec_sha256": spec_sha,
        "base_checkpoint": {"method": "sha256", "fingerprint": "a" * 64, "files": 4},
        "model_behavior_bundle": {
            "bundle_sha256": "b" * 64,
            "file_count": 1,
            "files": [{"path": "config.json", "sha256": "c" * 64, "size_bytes": 1}],
        },
    }
    training.write_bytes(canonical_json_bytes(training_payload))

    complete = checkpoint / "complete.json"
    adapter_raw = adapter.read_bytes()
    complete_payload = {
        "schema": "aura.recurrence_native_checkpoint.v3",
        "checkpoint_id": f"step-{step}",
        "step": step,
        "cursor": step,
        "config_sha256": _sha(training.read_bytes()),
        "execution_spec_sha256": spec_sha,
        "adapter": {
            "path": adapter.name,
            "sha256": _sha(adapter_raw),
            "size_bytes": len(adapter_raw),
        },
    }
    complete.write_bytes(canonical_json_bytes(complete_payload))
    return complete, training, source_spec


def _contract(root: Path, *, step: int = 5, max_steps: int = 10) -> dict:
    complete, training, spec = _write_source(
        root,
        step=step,
        max_steps=max_steps,
    )
    return build_recurrent_warm_start_contract(
        repo_root=Path(root.anchor),
        complete_path=complete,
        training_config_path=training,
        execution_spec_path=spec,
        copy_targets=("v_proj", "o_proj"),
        initialize_targets=("q_proj",),
    )


def test_build_contract_binds_partial_checkpoint_and_tensor_topology(tmp_path: Path) -> None:
    contract = _contract(tmp_path)

    source = contract["source_checkpoint"]
    assert source["checkpoint_status"] == "bounded_partial_checkpoint"
    assert source["step"] == 5
    assert source["max_steps"] == 10
    assert source["tensor_count"] == 8
    assert contract["transfer"]["copy_targets"] == ["o_proj", "v_proj"]
    assert contract["transfer"]["initialize_targets"] == ["q_proj"]


def test_build_contract_accepts_terminal_source_checkpoint(tmp_path: Path) -> None:
    contract = _contract(tmp_path, step=10, max_steps=10)
    assert contract["source_checkpoint"]["checkpoint_status"] == "complete_checkpoint"


def test_contract_rejects_source_adapter_tamper(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    repo_root = Path(tmp_path.anchor)
    adapter = repo_root / contract["source_checkpoint"]["adapter"]["path"]
    adapter.write_bytes(adapter.read_bytes() + b"tamper")

    with pytest.raises(
        RecurrentPolicyWarmStartError,
        match="warm_start_adapter_binding_mismatch",
    ):
        validate_recurrent_warm_start_contract(contract, repo_root=repo_root)


def test_contract_rejects_wrong_runtime_base_identity(tmp_path: Path) -> None:
    contract = _contract(tmp_path)

    with pytest.raises(
        RecurrentPolicyWarmStartError,
        match="warm_start_base_checkpoint_mismatch",
    ):
        validate_recurrent_warm_start_contract(
            contract,
            repo_root=Path(tmp_path.anchor),
            expected_base_checkpoint={
                "method": "sha256",
                "fingerprint": "d" * 64,
                "files": 4,
            },
        )


def test_load_contract_requires_canonical_repo_contained_file(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    path = tmp_path / "warm_start.json"
    path.write_bytes(canonical_json_bytes(contract))

    assert load_recurrent_warm_start_contract(
        path,
        repo_root=Path(tmp_path.anchor),
    ) == contract
    path.write_text(json.dumps(contract, indent=2))
    with pytest.raises(
        RecurrentPolicyWarmStartError,
        match="warm_start_contract_noncanonical",
    ):
        load_recurrent_warm_start_contract(path, repo_root=Path(tmp_path.anchor))


def test_transfer_plan_copies_only_declared_current_intersection(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    copied, report = plan_recurrent_warm_start(
        current_tensors=_current_tensors(),
        source_tensors=_source_tensors(),
        contract=contract,
    )

    assert len(copied) == 4
    assert all("layers.2" in key for key in copied)
    assert all("q_proj" not in key for key in copied)
    assert report["initialized_tensor_count"] == 2
    assert report["dropped_source_tensor_count"] == 4


def test_transfer_plan_rejects_missing_required_factor(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    source = _source_tensors()
    source.pop(_tensor_key(2, "v_proj", "b"))

    with pytest.raises(
        RecurrentPolicyWarmStartError,
        match="warm_start_required_tensor_missing_or_incompatible",
    ):
        plan_recurrent_warm_start(
            current_tensors=_current_tensors(),
            source_tensors=source,
            contract=contract,
        )


class _FakeModel:
    def __init__(self, tensors: dict[str, mx.array]) -> None:
        self.tensors = tensors

    def trainable_parameters(self):
        return self.tensors

    def load_weights(self, weights, *, strict: bool):
        assert strict is False
        self.tensors.update(dict(weights))


def _policy_sha(model: _FakeModel) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.tensors.items()):
        digest.update(key.encode("ascii"))
        digest.update(bytes(memoryview(value.astype(mx.float32))))
    return digest.hexdigest()


def test_apply_transfer_preserves_initialized_factors_and_seals_receipt(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    model = _FakeModel(_current_tensors())
    before = _policy_sha(model)

    receipt = apply_recurrent_warm_start(
        model,
        contract=contract,
        repo_root=Path(tmp_path.anchor),
        policy_before_sha256=before,
        policy_after=_policy_sha,
    )

    validated = validate_recurrent_warm_start_receipt(receipt)
    assert validated["copied_tensor_count"] == 4
    assert validated["initialized_tensor_count"] == 2
    assert validated["policy_after_sha256"] != before
    assert bool(
        mx.array_equal(
            model.tensors[_tensor_key(2, "q_proj", "a")],
            mx.zeros((3, 2)),
        ).item()
    )
