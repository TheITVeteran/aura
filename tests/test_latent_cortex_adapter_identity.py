from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from core.brain.llm.latent_cortex import adapter_identity

BASE_SHA = "a" * 64
OBJECTIVE_SHA = "b" * 64
ADAPTER_BYTES = b"deterministic recurrence adapter bytes"


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _tensor_metadata():
    return [
        {
            "key": "model.layers.0.self_attn.o_proj.lora_a",
            "shape": [16, 4],
            "dtype": "float32",
        },
        {
            "key": "model.layers.0.self_attn.o_proj.lora_b",
            "shape": [4, 16],
            "dtype": "float32",
        },
        {
            "key": "model.layers.0.self_attn.v_proj.lora_a",
            "shape": [16, 4],
            "dtype": "float32",
        },
        {
            "key": "model.layers.0.self_attn.v_proj.lora_b",
            "shape": [4, 16],
            "dtype": "float32",
        },
    ]


def _training_receipt():
    return {
        "schema": adapter_identity.TRAINING_RECEIPT_SCHEMA,
        "objective_schema": "aura.recurrence_native_objective.v1",
        "checkpoint": {"method": "sha256", "fingerprint": BASE_SHA, "files": 4},
        "lora": {
            "rank": 4,
            "targets": ["v_proj", "o_proj"],
            "wrapped_projections": 2,
            "trainable_params": 128,
        },
        "steps": 100,
    }


def _manifest():
    receipt_bytes = _canonical(_training_receipt())
    return {
        "schema": adapter_identity.MANIFEST_SCHEMA,
        "schema_version": adapter_identity.MANIFEST_SCHEMA_VERSION,
        "adapter_id": "resident-32b-r1",
        "base_checkpoint": {"method": "sha256", "fingerprint": BASE_SHA},
        "adapter": {
            "path": "adapter_final.safetensors",
            "sha256": hashlib.sha256(ADAPTER_BYTES).hexdigest(),
            "size_bytes": len(ADAPTER_BYTES),
        },
        "training_receipt": {
            "path": "receipt.json",
            "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "size_bytes": len(receipt_bytes),
            "schema": adapter_identity.TRAINING_RECEIPT_SCHEMA,
            "schema_version": adapter_identity.TRAINING_RECEIPT_SCHEMA_VERSION,
            "objective": {
                "name": "aura.recurrence_native_objective.v1",
                "schema_version": 1,
                "source_sha256": OBJECTIVE_SHA,
            },
        },
        "lora": {
            "rank": 4,
            "targets": ["o_proj", "v_proj"],
            "wrapped_projection_count": 2,
        },
        "tensors": _tensor_metadata(),
    }


def _validate(manifest=None, **overrides):
    arguments = {
        "actual_base_checkpoint_fingerprint": BASE_SHA,
        "adapter_bytes": ADAPTER_BYTES,
        "training_receipt_bytes": _canonical(_training_receipt()),
        "tensor_metadata": list(reversed(_tensor_metadata())),
    }
    arguments.update(overrides)
    return adapter_identity.validate_adapter_identity(
        _manifest() if manifest is None else manifest, **arguments
    )


def test_valid_identity_is_immutable_and_receipt_round_trips():
    receipt = _validate()

    assert receipt.adapter_id == "resident-32b-r1"
    assert receipt.tensor_count == 4
    assert receipt.targets == ("o_proj", "v_proj")
    assert receipt.objective_source_provenance == "posthoc_manifest_binding"
    with pytest.raises(FrozenInstanceError):
        receipt.rank = 8
    encoded = _canonical(receipt.to_dict())
    assert adapter_identity.parse_identity_receipt(encoded, manifest=_manifest()) == receipt


def test_deterministic_identity_ignores_mapping_and_inventory_order():
    first = _manifest()
    second = dict(reversed(list(first.items())))
    second["tensors"] = list(reversed(first["tensors"]))
    second["lora"] = {
        "wrapped_projection_count": 2,
        "targets": ["v_proj", "o_proj"],
        "rank": 4,
    }

    first_receipt = _validate(first)
    second_receipt = _validate(second)

    assert first_receipt == second_receipt
    assert first_receipt.composite_identity_sha256 == (
        adapter_identity.composite_identity_sha256(second)
    )
    assert first_receipt.manifest_sha256 == adapter_identity.manifest_sha256(second)


def test_wrong_base_checkpoint_fails_closed():
    with pytest.raises(
        adapter_identity.AdapterIdentityError,
        match="base_checkpoint_fingerprint_mismatch",
    ):
        _validate(actual_base_checkpoint_fingerprint="f" * 64)


def test_changed_adapter_bytes_fail_closed_even_at_same_size():
    changed = b"X" + ADAPTER_BYTES[1:]
    with pytest.raises(adapter_identity.AdapterIdentityError, match="adapter_sha256_mismatch"):
        _validate(adapter_bytes=changed)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_tensor_key_set_must_be_exact(mutation):
    metadata = _tensor_metadata()
    if mutation == "missing":
        metadata.pop()
    else:
        metadata.extend(
            [
                {
                    "key": "model.layers.1.self_attn.o_proj.lora_a",
                    "shape": [16, 4],
                    "dtype": "float32",
                },
                {
                    "key": "model.layers.1.self_attn.o_proj.lora_b",
                    "shape": [4, 16],
                    "dtype": "float32",
                },
            ]
        )
    with pytest.raises(adapter_identity.AdapterIdentityError):
        _validate(tensor_metadata=metadata)


def test_tensor_shape_and_dtype_are_identity_material():
    changed_shape = _tensor_metadata()
    changed_shape[0]["shape"] = [32, 4]
    with pytest.raises(adapter_identity.AdapterIdentityError, match="tensor_metadata_mismatch"):
        _validate(tensor_metadata=changed_shape)

    changed_dtype = _tensor_metadata()
    changed_dtype[0]["dtype"] = "float16"
    with pytest.raises(adapter_identity.AdapterIdentityError, match="tensor_metadata_mismatch"):
        _validate(tensor_metadata=changed_dtype)


@pytest.mark.parametrize("bad_rank", [True, 4.0, "4", 0, -1])
def test_malformed_rank_numbers_are_rejected(bad_rank):
    manifest = _manifest()
    manifest["lora"]["rank"] = bad_rank
    with pytest.raises(adapter_identity.AdapterIdentityError, match="lora_rank_invalid"):
        adapter_identity.parse_manifest(manifest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("adapter", "../adapter.safetensors"),
        ("adapter", "/var/adapter.safetensors"),
        ("adapter", "nested/../../adapter.safetensors"),
        ("adapter", "nested\\adapter.safetensors"),
        ("receipt", "../receipt.json"),
    ],
)
def test_artifact_paths_reject_traversal(field, value):
    manifest = _manifest()
    if field == "adapter":
        manifest["adapter"]["path"] = value
    else:
        manifest["training_receipt"]["path"] = value
    with pytest.raises(adapter_identity.AdapterIdentityError, match="path_invalid"):
        adapter_identity.parse_manifest(manifest)


def test_duplicate_targets_are_rejected_in_manifest_and_training_receipt():
    manifest = _manifest()
    manifest["lora"]["targets"] = ["o_proj", "o_proj"]
    with pytest.raises(adapter_identity.AdapterIdentityError, match="lora_targets_duplicate"):
        adapter_identity.parse_manifest(manifest)

    receipt = _training_receipt()
    receipt["lora"]["targets"] = ["o_proj", "o_proj"]
    receipt_bytes = _canonical(receipt)
    manifest = _manifest()
    manifest["training_receipt"]["sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
    manifest["training_receipt"]["size_bytes"] = len(receipt_bytes)
    with pytest.raises(adapter_identity.AdapterIdentityError, match="lora_targets_duplicate"):
        _validate(manifest, training_receipt_bytes=receipt_bytes)


def test_training_receipt_bytes_and_objective_are_bound_exactly():
    receipt_bytes = bytearray(_canonical(_training_receipt()))
    receipt_bytes[-2] = ord("1") if receipt_bytes[-2] != ord("1") else ord("2")
    with pytest.raises(
        adapter_identity.AdapterIdentityError, match="training_receipt_sha256_mismatch"
    ):
        _validate(training_receipt_bytes=bytes(receipt_bytes))

    receipt = _training_receipt()
    receipt["objective_schema"] = "aura.recurrence_native_objective.v2"
    changed_bytes = _canonical(receipt)
    manifest = _manifest()
    manifest["training_receipt"]["sha256"] = hashlib.sha256(changed_bytes).hexdigest()
    manifest["training_receipt"]["size_bytes"] = len(changed_bytes)
    with pytest.raises(
        adapter_identity.AdapterIdentityError, match="training_receipt_objective_mismatch"
    ):
        _validate(manifest, training_receipt_bytes=changed_bytes)


def test_missing_and_unknown_manifest_fields_fail_closed():
    missing = _manifest()
    del missing["training_receipt"]["objective"]
    with pytest.raises(
        adapter_identity.AdapterIdentityError,
        match="training_receipt_binding_schema_invalid",
    ):
        adapter_identity.parse_manifest(missing)

    extra = _manifest()
    extra["unverified"] = True
    with pytest.raises(
        adapter_identity.AdapterIdentityError, match="adapter_manifest_schema_invalid"
    ):
        adapter_identity.parse_manifest(extra)


def test_duplicate_json_keys_and_nonfinite_numbers_are_rejected():
    duplicate = b'{"schema":"x","schema":"y"}'
    with pytest.raises(
        adapter_identity.AdapterIdentityError,
        match="adapter_manifest_duplicate_json_key",
    ):
        adapter_identity.parse_manifest(duplicate)

    raw = _canonical(_manifest()).replace(b'"rank":4', b'"rank":NaN')
    with pytest.raises(
        adapter_identity.AdapterIdentityError, match="adapter_manifest_number_invalid"
    ):
        adapter_identity.parse_manifest(raw)


def test_partial_projection_pair_and_rank_mismatch_are_rejected():
    manifest = _manifest()
    manifest["tensors"].pop(1)
    with pytest.raises(
        adapter_identity.AdapterIdentityError,
        match="tensor_projection_pair_incomplete",
    ):
        adapter_identity.parse_manifest(manifest)

    manifest = _manifest()
    manifest["tensors"][0]["shape"] = [16, 8]
    with pytest.raises(adapter_identity.AdapterIdentityError, match="tensor_rank_mismatch"):
        adapter_identity.parse_manifest(manifest)


@pytest.mark.parametrize(
    ("key", "shape", "reason"),
    [
        (
            "unrelated.layers.0.self_attn.o_proj.lora_a",
            [16, 4],
            "tensor_target_mismatch",
        ),
        (
            "model.layers.0.self_attn.o_proj.lora_a",
            [4, 16],
            "tensor_rank_mismatch",
        ),
        (
            "model.layers.0.self_attn.o_proj.lora_a",
            [16, 4, 1],
            "tensor_shape_not_matrix",
        ),
    ],
)
def test_tensor_topology_requires_runtime_addressable_oriented_matrices(
    key, shape, reason
):
    manifest = _manifest()
    manifest["tensors"][0]["key"] = key
    manifest["tensors"][0]["shape"] = shape

    with pytest.raises(adapter_identity.AdapterIdentityError, match=reason):
        adapter_identity.parse_manifest(manifest)


def test_receipt_cannot_be_rebound_to_changed_manifest():
    receipt = _validate().to_dict()
    changed = copy.deepcopy(_manifest())
    changed["adapter_id"] = "resident-32b-r2"

    with pytest.raises(
        adapter_identity.AdapterIdentityError,
        match="adapter_identity_receipt_manifest_mismatch",
    ):
        adapter_identity.parse_identity_receipt(receipt, manifest=changed)


def test_legacy_v1_manifest_builder_binds_real_receipt_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_final.safetensors").write_bytes(ADAPTER_BYTES)
    (adapter_dir / "receipt.json").write_bytes(_canonical(_training_receipt()))
    objective = tmp_path / "objective.py"
    objective.write_text("OBJECTIVE_SCHEMA = 'v1'\n", encoding="utf-8")
    monkeypatch.setattr(
        adapter_identity,
        "inspect_mlx_tensor_metadata",
        lambda _path: adapter_identity.normalize_tensor_metadata(_tensor_metadata()),
    )

    manifest = adapter_identity.build_legacy_v1_manifest(
        adapter_dir,
        adapter_id="resident-32b-r1",
        actual_base_checkpoint_fingerprint=BASE_SHA,
        objective_source_path=objective,
    )
    receipt = adapter_identity.validate_adapter_identity(
        manifest,
        actual_base_checkpoint_fingerprint=BASE_SHA,
        adapter_bytes=ADAPTER_BYTES,
        training_receipt_bytes=_canonical(_training_receipt()),
        tensor_metadata=_tensor_metadata(),
    )

    assert receipt.adapter_sha256 == hashlib.sha256(ADAPTER_BYTES).hexdigest()
    assert receipt.objective_source_sha256 == hashlib.sha256(objective.read_bytes()).hexdigest()
    assert receipt.objective_source_provenance == "posthoc_manifest_binding"
