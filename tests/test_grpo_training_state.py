"""Exact, tamper-evident GRPO checkpoint and resume contracts."""

from __future__ import annotations

import json

import pytest

from core.learning.grpo_training_state import (
    GRPOCheckpointError,
    load_grpo_checkpoint,
    save_grpo_checkpoint,
)

mx = pytest.importorskip("mlx.core")

PROTOCOL = "a" * 64
DATASET = "b" * 64


def _state(step: int = 3):
    return {
        "protocol_sha256": PROTOCOL,
        "dataset_sha256": DATASET,
        "step": step,
        "curriculum": {"schema": "aura.adaptive_curriculum.v1", "cells": []},
        "telemetry": {"schema": "aura.grpo.v2", "groups": step},
        "history": [{"step": 0, "overall": 0.25}],
        "baseline_eval": {"step": 0, "overall": 0.25},
        "calibration": {"partial": False},
        "elapsed_training_s": 12.5,
        "invocation_count": 1,
        "rng_strategy": "stateless_sha256_step_seeded_v1",
        "optimizer_updates": step,
        "last_step_kind": "optimizer_update" if step else "initial",
        "last_step_committed": True,
    }


def _save(tmp_path, *, step=3, keep=3):
    return save_grpo_checkpoint(
        tmp_path,
        adapter_tensors={"layer.lora_a": mx.array([[1.0, 2.0]])},
        optimizer_tensors={"state.mean": mx.array([0.5, 0.25])},
        state=_state(step),
        keep=keep,
    )


def test_checkpoint_round_trips_exact_state_and_tensors(tmp_path):
    checkpoint = _save(tmp_path)

    loaded = load_grpo_checkpoint(
        tmp_path,
        expected_protocol_sha256=PROTOCOL,
        expected_dataset_sha256=DATASET,
    )

    assert loaded.checkpoint_dir == checkpoint
    assert loaded.state["step"] == 3
    assert loaded.state["last_step_committed"] is True
    assert loaded.state["optimizer_updates"] == 3
    assert set(loaded.adapter_tensors) == {"layer.lora_a"}
    assert bool(mx.array_equal(
        loaded.optimizer_state["state"]["mean"], mx.array([0.5, 0.25])
    ))


def test_checkpoint_refuses_protocol_or_dataset_drift(tmp_path):
    _save(tmp_path)

    with pytest.raises(GRPOCheckpointError, match="protocol_sha256 mismatch"):
        load_grpo_checkpoint(
            tmp_path,
            expected_protocol_sha256="c" * 64,
            expected_dataset_sha256=DATASET,
        )
    with pytest.raises(GRPOCheckpointError, match="dataset_sha256 mismatch"):
        load_grpo_checkpoint(
            tmp_path,
            expected_protocol_sha256=PROTOCOL,
            expected_dataset_sha256="d" * 64,
        )


def test_checkpoint_refuses_tensor_tampering(tmp_path):
    checkpoint = _save(tmp_path)
    adapter = checkpoint / "adapter.safetensors"
    adapter.write_bytes(adapter.read_bytes() + b"tamper")

    with pytest.raises(GRPOCheckpointError, match="adapter size mismatch"):
        load_grpo_checkpoint(
            tmp_path,
            expected_protocol_sha256=PROTOCOL,
            expected_dataset_sha256=DATASET,
        )


def test_checkpoint_refuses_incomplete_training_step(tmp_path):
    state = _state(step=2)
    state["last_step_committed"] = False

    with pytest.raises(GRPOCheckpointError, match="incomplete training step"):
        save_grpo_checkpoint(
            tmp_path,
            adapter_tensors={"layer.lora_a": mx.array([1.0])},
            optimizer_tensors={"state.mean": mx.array([0.0])},
            state=state,
        )


def test_checkpoint_distinguishes_degenerate_step_from_optimizer_update(tmp_path):
    state = _state(step=4)
    state["optimizer_updates"] = 3
    state["last_step_kind"] = "degenerate_group"

    save_grpo_checkpoint(
        tmp_path,
        adapter_tensors={"layer.lora_a": mx.array([1.0])},
        optimizer_tensors={"state.mean": mx.array([0.0])},
        state=state,
    )
    loaded = load_grpo_checkpoint(
        tmp_path,
        expected_protocol_sha256=PROTOCOL,
        expected_dataset_sha256=DATASET,
    )

    assert loaded.state["step"] == 4
    assert loaded.state["optimizer_updates"] == 3
    assert loaded.state["last_step_kind"] == "degenerate_group"


def test_checkpoint_refuses_more_optimizer_updates_than_steps(tmp_path):
    state = _state(step=2)
    state["optimizer_updates"] = 3

    with pytest.raises(GRPOCheckpointError, match="optimizer update count"):
        save_grpo_checkpoint(
            tmp_path,
            adapter_tensors={"layer.lora_a": mx.array([1.0])},
            optimizer_tensors={"state.mean": mx.array([0.0])},
            state=state,
        )


def test_checkpoint_pointer_cannot_escape_run_root(tmp_path):
    _save(tmp_path)
    pointer = json.loads((tmp_path / "latest.json").read_text())
    pointer["checkpoint"] = "../outside"
    (tmp_path / "latest.json").write_text(json.dumps(pointer))

    with pytest.raises(GRPOCheckpointError, match="path is invalid"):
        load_grpo_checkpoint(
            tmp_path,
            expected_protocol_sha256=PROTOCOL,
            expected_dataset_sha256=DATASET,
        )


def test_checkpoint_generation_retention_is_bounded(tmp_path):
    for step in range(1, 6):
        _save(tmp_path, step=step, keep=2)

    generations = list((tmp_path / "checkpoints").iterdir())
    assert len(generations) == 2
    loaded = load_grpo_checkpoint(
        tmp_path,
        expected_protocol_sha256=PROTOCOL,
        expected_dataset_sha256=DATASET,
    )
    assert loaded.state["step"] == 5
