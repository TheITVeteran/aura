"""Crash and identity contracts for recurrence-native training checkpoints."""
from __future__ import annotations

import json

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from core.learning.recurrence_training_state import (  # noqa: E402
    RecurrenceCheckpointError,
    load_recurrence_checkpoint,
    save_recurrence_checkpoint,
)

IDENTITIES = {
    "config_sha256": "a" * 64,
    "dataset_sha256": "b" * 64,
    "execution_spec_sha256": "c" * 64,
}


def _save(root, *, step=7):
    return save_recurrence_checkpoint(
        root,
        adapter_tensors={"model.layers.1.lora_a": mx.ones((2, 2))},
        optimizer_tensors={
            "step": mx.array(step),
            "model.layers.1.lora_a.m": mx.ones((2, 2)) * 2,
        },
        state={
            **IDENTITIES,
            "step": step,
            "epoch": 1,
            "cursor": 3,
            "order": [2, 0, 1],
        },
    )


def _load(root, **changes):
    identities = dict(IDENTITIES)
    identities.update(changes)
    return load_recurrence_checkpoint(root, **{
        f"expected_{key}": value for key, value in identities.items()
    })


def test_checkpoint_round_trip_preserves_adapter_optimizer_and_cursor(tmp_path):
    checkpoint = _save(tmp_path / "run")
    loaded = _load(tmp_path / "run")
    assert loaded.checkpoint_dir == checkpoint
    assert loaded.state["step"] == 7
    assert loaded.state["order"] == [2, 0, 1]
    assert bool(
        mx.array_equal(
            loaded.adapter_tensors["model.layers.1.lora_a"], mx.ones((2, 2))
        )
    )
    assert int(loaded.optimizer_state["step"]) == 7


def test_identity_mismatch_fails_before_tensor_acceptance(tmp_path):
    _save(tmp_path / "run")
    with pytest.raises(RecurrenceCheckpointError, match="config_sha256 mismatch"):
        _load(tmp_path / "run", config_sha256="d" * 64)


def test_torn_or_tampered_generation_cannot_resume(tmp_path):
    checkpoint = _save(tmp_path / "run")
    adapter = checkpoint / "adapter.safetensors"
    adapter.write_bytes(adapter.read_bytes() + b"tamper")
    with pytest.raises(RecurrenceCheckpointError, match="adapter size mismatch"):
        _load(tmp_path / "run")


def test_unpublished_orphan_does_not_move_latest_pointer(tmp_path):
    root = tmp_path / "run"
    first = _save(root, step=3)
    orphan = root / "checkpoints" / "step-99999999-orphan"
    orphan.mkdir()
    (orphan / "adapter.safetensors").write_bytes(b"partial")
    loaded = _load(root)
    assert loaded.checkpoint_dir == first


def test_latest_pointer_digest_is_binding(tmp_path):
    root = tmp_path / "run"
    _save(root)
    pointer_path = root / "latest.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["complete_sha256"] = "0" * 64
    pointer_path.write_text(json.dumps(pointer))
    with pytest.raises(RecurrenceCheckpointError, match="completion digest"):
        _load(root)
