"""Gating contracts for the autonomous RLC decision pipeline.

Nobody is present when this runs, so every gate has to fail closed on its own
evidence. These tests drive the real entry point with real verdict files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS = REPO_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_rlc_reconciliation_pipeline as pipeline  # noqa: E402


def _write(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=1, sort_keys=True), encoding="utf-8")


def _sweep_verdict(*, vanilla: int, best: int, best_arm: str = "rlc_nodisp") -> dict:
    return {
        "schema": "aura.rlc_reconciliation_sweep.v1",
        "arms": {
            "vanilla": {"correct": vanilla, "total": 28},
            best_arm: {"correct": best, "total": 28},
        },
        "vanilla_correct": vanilla,
        "best_recurrent_arm": best_arm,
        "best_recurrent_correct": best,
        "reaches_parity_with_ordinary_decode": best >= vanilla,
        "decision": (
            "proceed_to_checkpoint_phase"
            if best >= vanilla
            else "recurrent_path_below_ordinary_decode"
        ),
    }


def _run_pipeline(run_dir: Path, monkeypatch, *, checkpoint_root: str = "") -> int:
    argv = [
        "run_rlc_reconciliation_pipeline.py",
        "--run-dir",
        str(run_dir),
        "--model",
        "/nonexistent/resident",
    ]
    if checkpoint_root:
        argv += ["--checkpoint-root", checkpoint_root]
    monkeypatch.setattr(sys, "argv", argv)
    return pipeline.main()


def test_below_ordinary_decode_stops_before_touching_any_weights(tmp_path, monkeypatch):
    """The expected path. It must terminate, not proceed hopefully."""
    run_dir = tmp_path / "run"
    _write(run_dir / "sweep" / "verdict.json", _sweep_verdict(vanilla=13, best=5))

    # Any subprocess launch at all would be a defect on this path.
    def _explode(*_args, **_kwargs):
        raise AssertionError("no phase may run once the sweep verdict is negative")

    monkeypatch.setattr(pipeline, "_run", _explode)

    assert _run_pipeline(run_dir, monkeypatch) == 0
    decision = json.loads((run_dir / "DECISION.json").read_text())
    assert decision["decision"] == "no_fusion_recurrent_path_below_ordinary_decode"
    assert decision["evidence"]["vanilla_correct"] == 13
    assert decision["evidence"]["best_recurrent_correct"] == 5
    # A human-readable decision is written too, and says what was not claimed.
    readable = (run_dir / "DECISION.md").read_text()
    assert "no_fusion_recurrent_path_below_ordinary_decode" in readable
    assert "Not claimed" in readable
    assert not (run_dir / "fused_candidate").exists()


def test_parity_without_a_checkpoint_root_does_not_invent_one(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    _write(run_dir / "sweep" / "verdict.json", _sweep_verdict(vanilla=13, best=13))
    monkeypatch.setattr(
        pipeline, "_run", lambda *a, **k: pytest.fail("no phase should run")
    )

    assert _run_pipeline(run_dir, monkeypatch) == 0
    decision = json.loads((run_dir / "DECISION.json").read_text())
    assert decision["decision"] == "parity_reached_no_checkpoint_root_configured"


def test_a_checkpoint_that_loses_to_ordinary_decode_is_never_fused(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    _write(run_dir / "sweep" / "verdict.json", _sweep_verdict(vanilla=13, best=14))

    calls: list[list[str]] = []

    def _fake_run(_run_dir, argv, *, timeout_s):  # noqa: ARG001
        calls.append(argv)
        _write(
            run_dir / "checkpoints" / "verdict.json",
            {
                "schema": "aura.rlc_checkpoint_sweep.v1",
                "best_checkpoint": "sequence-00000050-step-00000049-abc",
                "best_correct": 1,
                "ordinary_decode_correct_scaled": 1.857,
                "beats_ordinary_decode": False,
            },
        )
        return 0

    monkeypatch.setattr(pipeline, "_run", _fake_run)

    assert _run_pipeline(run_dir, monkeypatch, checkpoint_root="/tmp/ckpt") == 0
    decision = json.loads((run_dir / "DECISION.json").read_text())
    assert decision["decision"] == "no_fusion_no_checkpoint_beats_ordinary_decode"
    # It ran the checkpoint sweep and then stopped -- it never reached fusion.
    assert len(calls) == 1
    assert "run_rlc_checkpoint_sweep.py" in " ".join(calls[0])


def test_a_fused_candidate_that_regresses_ordinary_decode_is_not_activated(
    tmp_path, monkeypatch
):
    """The gate that matters most: fusion unscopes a slot-scoped adapter."""
    run_dir = tmp_path / "run"
    _write(run_dir / "sweep" / "verdict.json", _sweep_verdict(vanilla=13, best=16))

    def _fake_run(_run_dir, argv, *, timeout_s):  # noqa: ARG001
        joined = " ".join(argv)
        if "run_rlc_checkpoint_sweep.py" in joined:
            _write(
                run_dir / "checkpoints" / "verdict.json",
                {
                    "best_checkpoint": "sequence-00000060-step-00000059-def",
                    "best_correct": 3,
                    "beats_ordinary_decode": True,
                },
            )
        elif "fuse_rlc_candidate.py" in joined:
            (run_dir / "fused_candidate").mkdir(parents=True, exist_ok=True)
            (run_dir / "fused_candidate" / "config.json").write_text("{}")
        elif "run_rlc_reconciliation_sweep.py" in joined:
            # Recurrent path improved, ordinary decode fell off a cliff.
            _write(
                run_dir / "candidate_sweep" / "verdict.json",
                {
                    "arms": {"vanilla": {"correct": 9}, "rlc_nodisp": {"correct": 17}},
                    "vanilla_correct": 9,
                    "best_recurrent_arm": "rlc_nodisp",
                    "best_recurrent_correct": 17,
                },
            )
        return 0

    monkeypatch.setattr(pipeline, "_run", _fake_run)

    assert _run_pipeline(run_dir, monkeypatch, checkpoint_root="/tmp/ckpt") == 0
    decision = json.loads((run_dir / "DECISION.json").read_text())
    assert decision["decision"] == "no_activation_fused_candidate_regressed"
    assert decision["evidence"]["ordinary_decode_preserved"] is False
    # The recurrent gain alone did not carry it.
    assert decision["evidence"]["recurrent_gain_reproduced"] is True


def test_a_candidate_that_holds_both_arms_is_staged_with_a_rollback(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    _write(run_dir / "sweep" / "verdict.json", _sweep_verdict(vanilla=13, best=16))

    def _fake_run(_run_dir, argv, *, timeout_s):  # noqa: ARG001
        joined = " ".join(argv)
        if "run_rlc_checkpoint_sweep.py" in joined:
            _write(
                run_dir / "checkpoints" / "verdict.json",
                {
                    "best_checkpoint": "sequence-00000060-step-00000059-def",
                    "best_correct": 3,
                    "beats_ordinary_decode": True,
                },
            )
        elif "fuse_rlc_candidate.py" in joined:
            (run_dir / "fused_candidate").mkdir(parents=True, exist_ok=True)
            (run_dir / "fused_candidate" / "config.json").write_text("{}")
        elif "run_rlc_reconciliation_sweep.py" in joined:
            _write(
                run_dir / "candidate_sweep" / "verdict.json",
                {
                    "arms": {"vanilla": {"correct": 14}, "rlc_nodisp": {"correct": 18}},
                    "vanilla_correct": 14,
                    "best_recurrent_arm": "rlc_nodisp",
                    "best_recurrent_correct": 18,
                },
            )
        return 0

    monkeypatch.setattr(pipeline, "_run", _fake_run)

    assert _run_pipeline(run_dir, monkeypatch, checkpoint_root="/tmp/ckpt") == 0
    decision = json.loads((run_dir / "DECISION.json").read_text())
    assert decision["decision"] == "fused_candidate_passed_staged_for_activation"
    # A rollback target is recorded, and it is the untouched resident.
    assert decision["evidence"]["rollback_path"] == "/nonexistent/resident"
    assert decision["evidence"]["candidate_vanilla_correct"] == 14


def test_an_unfinished_sweep_is_reported_not_assumed(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    (run_dir / "sweep").mkdir(parents=True)
    monkeypatch.setattr(pipeline, "_wait_for_sweep", lambda *a, **k: None)

    assert _run_pipeline(run_dir, monkeypatch) == 1
    decision = json.loads((run_dir / "DECISION.json").read_text())
    assert decision["decision"] == "incomplete_sweep_did_not_finish"
