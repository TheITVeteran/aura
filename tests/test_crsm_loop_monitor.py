"""Tests for CRSM→LoRA loop closure verification."""
from __future__ import annotations

import hashlib
import json
import time

from core.consciousness.crsm_loop_monitor import CRSMLoopMonitor


def _monitor(tmp_path):
    ds = tmp_path / "synthetic_training" / "lora_dataset.jsonl"
    fused = tmp_path / "fused-model"
    manifest = tmp_path / "training" / "data" / "crsm_integration_manifest.json"
    training_state = tmp_path / "training" / "adapters" / "aura-personality" / "training_state.json"
    ds.parent.mkdir(parents=True, exist_ok=True)
    fused.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    training_state.parent.mkdir(parents=True, exist_ok=True)
    return CRSMLoopMonitor(
        dataset_path=ds,
        fused_model_dir=fused,
        marker_path=ds.parent / ".crsm_consumed.json",
        integration_manifest_path=manifest,
        training_state_path=training_state,
        training_data_dir=manifest.parent,
    )


def _write_lines(path, n):
    path.write_text("".join(json.dumps({"text": f"sample {i}"}) + "\n" for i in range(n)), encoding="utf-8")


def _jsonl_stats(path):
    digest = hashlib.sha256()
    lines = 0
    with path.open("rb") as fh:
        for raw in fh:
            lines += 1
            digest.update(raw)
    st = path.stat()
    return {"path": str(path), "lines": lines, "size": st.st_size, "mtime": st.st_mtime, "sha256": digest.hexdigest()}


def _write_manifest_with_corpus(m, *, source_lines, accepted):
    train = m.training_data_dir / "train.jsonl"
    valid = m.training_data_dir / "valid.jsonl"
    train.parent.mkdir(parents=True, exist_ok=True)
    _write_lines(train, 3)
    _write_lines(valid, 2)
    m.integration_manifest_path.write_text(
        json.dumps(
            {
                "source_lines": source_lines,
                "source_mtime": m.dataset_path.stat().st_mtime,
                "accepted": accepted,
                "deduplicated": 10,
                "rejected_by_reason": {"too_short": 10},
                "output": {
                    "total_examples": 5,
                    "crsm_examples": accepted,
                    "train": _jsonl_stats(train),
                    "valid": _jsonl_stats(valid),
                },
            }
        ),
        encoding="utf-8",
    )


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
    manifest = tmp_path / "crsm_manifest.json"
    # a newer fused model appears + training marks consumption
    (m.fused_model_dir / "Aura-32B-new").mkdir()
    m.mark_dataset_consumed(
        model_path=str(m.fused_model_dir / "Aura-32B-new"),
        lines_consumed=100,
        accepted_lines=80,
        rejected_lines=20,
        manifest_path=str(manifest),
        source="test",
    )
    state = m.loop_state()
    assert state["state"] == "closed"
    assert state["unconsumed"] == 0
    assert state["accepted_lines"] == 80
    assert state["rejected_lines"] == 20
    assert "80 eligible captures trained" in state["reason"]


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
    model = tmp_path / "models" / "x"
    manifest = tmp_path / "manifest.json"
    m.mark_dataset_consumed(
        model_path=str(model),
        lines_consumed=30,
        accepted_lines=25,
        rejected_lines=5,
        manifest_path=str(manifest),
        source="unit",
    )
    data = json.loads(m.marker_path.read_text())
    assert data["lines_consumed"] == 30 and data["model_path"] == str(model)
    assert data["accepted_lines"] == 25
    assert data["rejected_lines"] == 5
    assert data["manifest_path"] == str(manifest)
    assert data["source"] == "unit"


def test_open_state_reports_current_manifest_and_train_fuse_next_action(tmp_path):
    m = _monitor(tmp_path)
    _write_lines(m.dataset_path, 100)
    _write_manifest_with_corpus(m, source_lines=100, accepted=80)
    m.training_state_path.write_text(
        json.dumps({"phase": "resume_done", "last_iter": 66000, "last_pipeline_rc": 1}),
        encoding="utf-8",
    )

    state = m.loop_state()

    assert state["state"] == "open"
    assert "integrated into the LoRA corpus" in state["reason"]
    assert state["integration_manifest"]["current_for_dataset"] is True
    assert state["integration_manifest"]["accepted"] == 80
    assert state["integration_manifest"]["output_integrity"]["corpus_current"] is True
    assert state["training_state"]["last_iter"] == 66000
    assert state["next_action"]["phase"] == "crsm_delta_train_fuse_publish"
    assert "training/train_and_fuse.py" in " ".join(state["next_action"]["command"])
    assert "--crsm-delta" in state["next_action"]["command"]


def test_restored_training_corpus_invalidates_current_manifest(tmp_path):
    m = _monitor(tmp_path)
    _write_lines(m.dataset_path, 100)
    _write_manifest_with_corpus(m, source_lines=100, accepted=80)
    (m.training_data_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")

    state = m.loop_state()

    assert state["integration_manifest"]["current_for_dataset"] is False
    assert state["integration_manifest"]["output_integrity"]["corpus_current"] is False
    assert state["next_action"]["phase"] == "prepare_dataset"


def test_stale_manifest_reports_prepare_dataset_next_action(tmp_path):
    m = _monitor(tmp_path)
    _write_lines(m.dataset_path, 100)
    m.integration_manifest_path.write_text(
        json.dumps({"source_lines": 50, "source_mtime": m.dataset_path.stat().st_mtime - 10, "accepted": 40}),
        encoding="utf-8",
    )

    state = m.loop_state()

    assert state["state"] == "open"
    assert state["integration_manifest"]["current_for_dataset"] is False
    assert state["next_action"]["phase"] == "prepare_dataset"
    assert state["next_action"]["command"] == ["python", "training/build_dataset_v3.py"]
