"""Tests for CRSM→LoRA loop closure verification."""
from __future__ import annotations

import json
import time

from core.consciousness.crsm_loop_monitor import CRSMLoopMonitor


def _monitor(tmp_path):
    ds = tmp_path / "synthetic_training" / "lora_dataset.jsonl"
    fused = tmp_path / "fused-model"
    ds.parent.mkdir(parents=True, exist_ok=True)
    fused.mkdir(parents=True, exist_ok=True)
    return CRSMLoopMonitor(
        dataset_path=ds,
        fused_model_dir=fused,
        marker_path=ds.parent / ".crsm_consumed.json",
    )


def _write_lines(path, n):
    path.write_text("".join(json.dumps({"text": f"sample {i}"}) + "\n" for i in range(n)), encoding="utf-8")


def test_idle_when_no_captures(tmp_path):
    m = _monitor(tmp_path)
    assert m.loop_state()["state"] == "idle"


def test_open_when_captures_accumulate_without_training(tmp_path):
    m = _monitor(tmp_path)
    _write_lines(m.dataset_path, 100)            # lots of captures, no model
    state = m.audit()
    assert state["state"] == "open"
    assert state["unconsumed"] == 100


def test_closed_after_training_consumes_dataset(tmp_path):
    m = _monitor(tmp_path)
    _write_lines(m.dataset_path, 100)
    # a newer fused model appears + training marks consumption
    (m.fused_model_dir / "Aura-32B-new").mkdir()
    m.mark_dataset_consumed(model_path=str(m.fused_model_dir / "Aura-32B-new"), lines_consumed=100)
    state = m.loop_state()
    assert state["state"] == "closed"
    assert state["unconsumed"] == 0


def test_open_again_when_new_captures_arrive_after_training(tmp_path):
    m = _monitor(tmp_path)
    _write_lines(m.dataset_path, 50)
    (m.fused_model_dir / "model-a").mkdir()
    m.mark_dataset_consumed(model_path="model-a", lines_consumed=50)
    assert m.loop_state()["state"] == "closed"
    # more captures arrive, dataset grows past the warn threshold
    time.sleep(0.01)
    _write_lines(m.dataset_path, 120)
    assert m.loop_state()["unconsumed"] == 70
    assert m.loop_state()["state"] == "open"


def test_marker_round_trip(tmp_path):
    m = _monitor(tmp_path)
    _write_lines(m.dataset_path, 30)
    m.mark_dataset_consumed(model_path="/models/x", lines_consumed=30)
    data = json.loads(m.marker_path.read_text())
    assert data["lines_consumed"] == 30 and data["model_path"] == "/models/x"
