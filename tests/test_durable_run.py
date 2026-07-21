"""Surviving a dead machine (CP237).

The detached supervisor survives sleep; it does not survive power loss or a
jetsam kill, which is how the CP227 gate died mid-run. A checkpoint
half-written when power fails must never load as if complete.
"""
from __future__ import annotations

import json

import pytest

from core.learning.durable_run import DURABLE_RUN_SCHEMA, DurableRun


def test_a_fresh_run_resumes_from_zero(tmp_path):
    run = DurableRun(tmp_path)
    assert run.latest() is None
    assert run.resume_step() == 0


def test_a_saved_checkpoint_is_the_resume_point(tmp_path):
    run = DurableRun(tmp_path)
    run.save(50, {"adapter": "a.safetensors", "curriculum": {"x": 1}})
    run.save(100, {"adapter": "b.safetensors", "curriculum": {"x": 2}})
    latest = run.latest()
    assert latest.step == 100
    assert latest.payload["curriculum"] == {"x": 2}
    assert run.resume_step() == 100


def test_a_torn_latest_checkpoint_falls_back_to_the_previous_good_one(tmp_path):
    """The core durability guarantee: a partial write never loads as
    complete."""
    run = DurableRun(tmp_path)
    run.save(50, {"step": 50})
    run.save(100, {"step": 100})
    # Simulate a torn write of the newest checkpoint.
    torn = tmp_path / "checkpoint_00000100.json"
    torn.write_text('{"schema": "aura.durable_run.v1", "step": 100, "payl')
    latest = run.latest()
    assert latest is not None
    assert latest.step == 50, "a torn checkpoint must not be treated as valid"


def test_a_torn_manifest_still_resumes_by_scanning(tmp_path):
    run = DurableRun(tmp_path)
    run.save(50, {"step": 50})
    (tmp_path / "checkpoint_manifest.json").write_text("{not valid json")
    latest = run.latest()
    assert latest is not None and latest.step == 50


def test_manifest_points_at_the_latest_after_each_save(tmp_path):
    run = DurableRun(tmp_path)
    run.save(10, {"s": 10})
    run.save(20, {"s": 20})
    manifest = json.loads((tmp_path / "checkpoint_manifest.json").read_text())
    assert manifest["latest"] == "checkpoint_00000020.json"


def test_old_checkpoints_are_pruned_but_keep_is_respected(tmp_path):
    run = DurableRun(tmp_path, keep=2)
    for step in (10, 20, 30, 40):
        run.save(step, {"s": step})
    remaining = sorted(p.name for p in tmp_path.glob("checkpoint_0*.json"))
    assert remaining == ["checkpoint_00000030.json", "checkpoint_00000040.json"]
    # Pruning must not break resume.
    assert run.resume_step() == 40


def test_saved_payload_carries_the_schema(tmp_path):
    run = DurableRun(tmp_path)
    path = run.save(1, {"anything": True})
    assert json.loads(path.read_text())["schema"] == DURABLE_RUN_SCHEMA


def test_invalid_inputs_are_refused(tmp_path):
    with pytest.raises(ValueError, match="keep"):
        DurableRun(tmp_path, keep=0)
    with pytest.raises(ValueError, match="step"):
        DurableRun(tmp_path).save(-1, {})
