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
    # Genuinely trainable captures (real User:/Aura: exchanges, no internal
    # markers) so the loop's eligibility gate counts them — the monitor only
    # reports OPEN when there is something the trainer would actually accept.
    path.write_text(
        "".join(
            json.dumps({
                "text": (
                    f"User: Question {i} about how something works in practice?\n"
                    f"Aura: Here is answer {i} with enough grounded detail to pass the gate."
                ),
                "_quality": 0.8,
            }) + "\n"
            for i in range(n)
        ),
        encoding="utf-8",
    )


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
    source_stats = _jsonl_stats(m.dataset_path)
    m.integration_manifest_path.write_text(
        json.dumps(
            {
                "source_lines": source_lines,
                "source_size": source_stats["size"],
                "source_sha256": source_stats["sha256"],
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


def _publish_active(m, model_path, *, governance=None):
    model = m.fused_model_dir / model_path if not str(model_path).startswith("/") else None
    resolved = model if model is not None else model_path
    if model is not None:
        model.mkdir(parents=True, exist_ok=True)
    active = {
        "active_model_path": str(resolved),
        "fused_at": time.time(),
    }
    if governance:
        active["governance"] = governance
    (m.fused_model_dir / "active.json").write_text(json.dumps(active), encoding="utf-8")
    return str(resolved)


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
    model_path = _publish_active(m, "Aura-32B-new")
    m.mark_dataset_consumed(
        model_path=model_path,
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


def test_overcounted_consumed_marker_cannot_close_current_dataset(tmp_path):
    m = _monitor(tmp_path)
    _write_lines(m.dataset_path, 100)
    model_path = _publish_active(m, "Aura-32B-overcount")
    m.mark_dataset_consumed(
        model_path=model_path,
        lines_consumed=101,
        accepted_lines=101,
        rejected_lines=0,
        source="test-overcount",
    )

    state = m.loop_state()
    assert state["state"] != "closed"
    assert state["marker_matches_dataset"] is False
    assert state["verified_consumption"] is False


def test_open_again_when_new_captures_arrive_after_training(tmp_path):
    m = _monitor(tmp_path)
    _write_lines(m.dataset_path, 50)
    model_path = _publish_active(m, "model-a")
    m.mark_dataset_consumed(model_path=model_path, lines_consumed=50)
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
    assert data["dataset_size"] == m.dataset_path.stat().st_size
    assert data["dataset_sha256"] == _jsonl_stats(m.dataset_path)["sha256"]
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
    assert "in the LoRA corpus but not yet trained/fused" in state["reason"]
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


def test_identical_capture_rewrite_does_not_stale_current_manifest(tmp_path):
    m = _monitor(tmp_path)
    _write_lines(m.dataset_path, 100)
    _write_manifest_with_corpus(m, source_lines=100, accepted=80)
    first = m.loop_state()
    assert first["integration_manifest"]["current_for_dataset"] is True

    original = m.dataset_path.read_text(encoding="utf-8")
    time.sleep(0.01)
    m.dataset_path.write_text(original, encoding="utf-8")

    state = m.loop_state()

    assert state["integration_manifest"]["current_for_dataset"] is True
    assert state["next_action"]["phase"] == "crsm_delta_train_fuse_publish"


def test_consumed_marker_with_content_hash_survives_identical_rewrite(tmp_path):
    m = _monitor(tmp_path)
    _write_lines(m.dataset_path, 30)
    model_path = _publish_active(m, "model-a")
    m.mark_dataset_consumed(
        model_path=model_path,
        lines_consumed=30,
        accepted_lines=25,
        rejected_lines=5,
        source="unit",
    )
    original = m.dataset_path.read_text(encoding="utf-8")
    time.sleep(0.01)
    m.dataset_path.write_text(original, encoding="utf-8")

    state = m.loop_state()

    assert state["state"] == "closed"
    assert state["verified_consumption"] is True
    assert "25 eligible captures trained" in state["reason"]


def test_newer_model_without_matching_consumption_marker_stays_pending(tmp_path):
    m = _monitor(tmp_path)
    _write_lines(m.dataset_path, 10)
    _publish_active(m, "model-newer")

    state = m.loop_state()

    assert state["state"] == "pending"
    assert state["verified_consumption"] is False
    assert "lack a verified active-model consumption marker" in state["reason"]


def test_legacy_consumed_marker_reports_hashless_pending_after_rewrite(tmp_path):
    m = _monitor(tmp_path)
    _write_lines(m.dataset_path, 30)
    _write_manifest_with_corpus(m, source_lines=30, accepted=25)
    m.marker_path.write_text(
        json.dumps({"lines_consumed": 30, "accepted_lines": 25, "rejected_lines": 5}),
        encoding="utf-8",
    )
    time.sleep(0.01)
    m.dataset_path.write_text(m.dataset_path.read_text(encoding="utf-8"), encoding="utf-8")

    state = m.loop_state()

    assert state["state"] == "pending"
    assert "current corpus needs train/fuse confirmation" in state["reason"]
    assert state["next_action"]["phase"] == "crsm_delta_train_fuse_publish"


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


def _write_internal_control_captures(path, n):
    """The idle self-reflection captures Bryan's live poll surfaced: real
    captures from Aura's side, but internal-control (<thought>/<action>,
    will-approved self-reflection) that the training gate always rejects."""
    path.write_text(
        "".join(
            json.dumps({
                "text": (
                    "User: Will-approved self-reflection\n"
                    f"Aura: <thought>\nSelf-reflection {i} accepted as a plasticity signal.\n"
                    "</thought>\n<action>\nI noticed a repair pattern.\n</action>"
                ),
                "_quality": 0.72,
            }) + "\n"
            for i in range(n)
        ),
        encoding="utf-8",
    )


def test_internal_control_captures_are_idle_not_open(tmp_path):
    """The reported false alarm: 33 internal-control captures made the health
    poll cry 'CRSM→LoRA loop OPEN (proof integrity degraded)' when there was
    nothing trainable. The loop must report IDLE, and the advisory (which
    fires only on OPEN) must stay silent."""
    m = _monitor(tmp_path)
    _write_internal_control_captures(m.dataset_path, 33)

    assert m.eligible_capture_count() == 0
    state = m.loop_state()
    assert state["state"] == "idle", state
    assert state["eligible_captures"] == 0
    assert "0 are eligible" in state["reason"]
    assert state["next_action"]["required"] is False
    # The integrity advisory keys off state == 'open'; idle must not trip it.
    assert state["state"] != "open"


def test_open_requires_eligible_captures_not_raw_lines(tmp_path):
    """Mixed corpus: only the trainable captures hold the loop open."""
    m = _monitor(tmp_path)
    # 30 internal-control (ineligible) + 30 real exchanges (eligible)
    internal = [
        json.dumps({"text": "User: Will-approved self-reflection\nAura: <thought>x</thought>"}) + "\n"
        for _ in range(30)
    ]
    eligible = [
        json.dumps({
            "text": f"User: Question {i} about the world?\nAura: A grounded answer {i} with real words."
        }) + "\n"
        for i in range(30)
    ]
    m.dataset_path.write_text("".join(internal + eligible), encoding="utf-8")

    assert m.eligible_capture_count() == 30
    state = m.loop_state()
    assert state["state"] == "open", state
    assert state["eligible_captures"] == 30


def test_eligible_count_cached_by_dataset_sha(tmp_path):
    m = _monitor(tmp_path)
    _write_lines(m.dataset_path, 40)
    first = m.eligible_capture_count()
    assert first == 40
    # cache hit: same sha -> no recompute path; still correct
    assert m.eligible_capture_count() == 40
    # dataset grows -> recompute
    _write_lines(m.dataset_path, 60)
    assert m.eligible_capture_count() == 60
