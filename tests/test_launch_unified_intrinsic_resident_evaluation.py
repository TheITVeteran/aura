from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from tools import launch_unified_intrinsic_resident_evaluation as launcher


def _arguments(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        action="prepare",
        campaign=root,
        output=None,
        stem="checkpoint_best_heldout",
        per_cell=1,
        evaluation_seed=241,
        max_tokens=32,
        task_depths=(1,),
        recurrence_depths=(4,),
        memory_limit_gb=40.0,
        cache_limit_gb=2.0,
        wired_limit_gb=48.0,
        startup_lethal_mb=launcher.DEFAULT_STARTUP_LETHAL_MB,
        steady_lethal_mb=launcher.DEFAULT_STEADY_LETHAL_MB,
        timeout=14_400.0,
    )


def _terminal_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    completion_checkpoint: dict[str, object] | None = None,
) -> tuple[argparse.Namespace, dict[str, object], dict[str, object]]:
    root = tmp_path / "campaign"
    source = tmp_path / "capsule"
    output = root / "training-output"
    inputs = root / "inputs"
    for path in (root, source / "tools", output, inputs):
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    for path in (
        python,
        source / "tools/evaluate_unified_intrinsic_decoding.py",
        source / "tools/unified_intrinsic_preload_barrier.py",
        source / "tools/memory_sentinel.py",
    ):
        path.write_text("pass\n", encoding="utf-8")
    key = root / "heartbeat.key"
    key.write_bytes(b"k" * 32)
    checkpoint: dict[str, object] = {
        "present": True,
        "step": 3,
        "checkpoint_sha256": "a" * 64,
        "receipt_sha256": "b" * 64,
        "generation": "generation-00000003-a",
        "complete": True,
        "training_receipt": {
            "binding": "authoritative_checkpoint",
            "receipt_sha256": "c" * 64,
            "steps": 3,
            "complete": True,
            "halt_reason": "max_steps",
        },
    }
    config: dict[str, object] = {
        "campaign_id": "cp240-test",
        "config_sha256": "d" * 64,
        "source": {"git": {"root": str(source)}},
        "runtime": {"interpreter": {"executable": str(python)}},
        "paths": {
            "campaign_root": str(root),
            "training_output": str(output),
            "heartbeat_key": str(key),
        },
    }
    completion_body = {
        "schema": launcher.resident.COMPLETION_SCHEMA,
        "campaign_id": "cp240-test",
        "config_sha256": "d" * 64,
        "profile": "canary",
        "package": {},
        "launchd": {},
        "checkpoint": completion_checkpoint or checkpoint,
        "attempt_count": 2,
        "resident_training_complete": True,
        "reasoning_gain_proven": False,
        "frontier_level_proven": False,
        "fusion_allowed": False,
    }
    completion = {
        **completion_body,
        "completion_sha256": launcher.canonical_sha256(completion_body),
    }
    (root / "completion-receipt.json").write_bytes(
        launcher.canonical_bytes(completion) + b"\n"
    )
    monkeypatch.setattr(launcher.resident, "_load_config", lambda _path: config)
    monkeypatch.setattr(
        launcher.resident,
        "_checkpoint_snapshot",
        lambda _config: checkpoint,
    )
    return _arguments(root), config, checkpoint


def test_prepare_binds_terminal_checkpoint_and_frozen_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, config, checkpoint = _terminal_fixture(tmp_path, monkeypatch)

    plan = launcher.prepare(arguments)

    assert plan["scientific"]["checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
    assert plan["command"][0] == config["runtime"]["interpreter"]["executable"]
    assert plan["evaluator_command"][1].startswith(
        str(Path(config["source"]["git"]["root"]))
    )
    assert plan["evaluator_command"][-8:] == [
        "--preload-ready-path",
        plan["ready"],
        "--preload-release-path",
        plan["release"],
        "--preload-key-path",
        plan["heartbeat_key"],
        "--preload-config-sha256",
        plan["evaluation_identity_sha256"],
    ]
    assert plan["claims_not_supported"] == [
        "broad_reasoning_gain",
        "frontier_performance",
        "production_fusion",
        "wow_signal",
    ]
    stored = Path(plan["evaluation_root"]) / "evaluation-plan.json"
    assert stored.stat().st_mode & 0o777 == 0o400
    assert launcher._read_document(stored) == plan


def test_prepare_rejects_completion_for_a_different_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, _config, checkpoint = _terminal_fixture(
        tmp_path,
        monkeypatch,
        completion_checkpoint={
            "present": True,
            "step": 2,
            "checkpoint_sha256": "f" * 64,
            "complete": True,
        },
    )
    assert checkpoint["step"] == 3

    with pytest.raises(
        launcher.ResidentEvaluationLaunchError,
        match="exact terminal campaign checkpoint",
    ):
        launcher.prepare(arguments)


def test_launch_requires_sentinel_and_pressure_evidence_before_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, _config, _checkpoint = _terminal_fixture(tmp_path, monkeypatch)
    invocations: list[list[str]] = []
    sentinel_status = {
        "terminal": False,
        "child_pid": 44,
        "child_start_token": "sentinel-token",
    }

    def fake_invoke(argv: list[str]) -> dict[str, object]:
        invocations.append(argv)
        if len(invocations) == 2:
            sentinel_dir = Path(argv[argv.index("--run-dir") + 1])
            (sentinel_dir / "ring.jsonl").parent.mkdir(parents=True, exist_ok=True)
            (sentinel_dir / "ring.jsonl").write_text("{}\n", encoding="ascii")
        return {"state": "running"}

    monkeypatch.setattr(launcher, "_invoke_detached", fake_invoke)
    monkeypatch.setattr(
        launcher,
        "_wait_for_target",
        lambda _path: {
            "child_pid": 33,
            "child_start_token": "target-token",
            "supervisor_pid": 22,
        },
    )
    monkeypatch.setattr(launcher.detached, "_status", lambda _path: sentinel_status)
    monkeypatch.setattr(launcher.detached, "_identity_state", lambda *_args: "alive")
    monkeypatch.setattr(
        launcher.resident,
        "_target_identity",
        lambda _status: (33, "target-token", 22),
    )
    monkeypatch.setattr(
        launcher.resident,
        "_stable_last_ring_entry",
        lambda _path, required_stage: (
            {
                "guard_stage": "startup",
                "active_lethal_mb": launcher.DEFAULT_STARTUP_LETHAL_MB,
            },
            "e" * 64,
        ),
    )
    monkeypatch.setattr(
        launcher.resident,
        "_verify_caffeinate",
        lambda *_args: {"pid": 55},
    )
    monkeypatch.setattr(
        launcher,
        "host_pressure",
        lambda: {"available": True, "under_pressure": False},
    )
    release_calls: list[dict[str, object]] = []

    def fake_release(_path: Path, **kwargs: object) -> dict[str, object]:
        release_calls.append(kwargs)
        return {"hmac_sha256": "f" * 64}

    monkeypatch.setattr(launcher, "publish_release", fake_release)

    result = launcher.launch(arguments)

    assert len(invocations) == 2
    assert result["state"] == "running"
    assert release_calls[0]["expected_target_pid"] == 33
    assert release_calls[0]["sentinel_pid"] == 44
    assert release_calls[0]["host_pressure"] == {
        "available": True,
        "under_pressure": False,
    }


def test_status_rejects_forged_report_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, _config, _checkpoint = _terminal_fixture(tmp_path, monkeypatch)
    plan = launcher.prepare(arguments)
    detached_root = Path(plan["evaluation_root"]) / "detached"
    detached_root.mkdir()
    (detached_root / launcher.detached.PLAN_FILE).write_text("{}\n", encoding="ascii")
    report = {
        "schema": "aura.unified_intrinsic_decode_evaluation.v1",
        "report_sha256": "0" * 64,
    }
    Path(plan["report"]).write_bytes(launcher.canonical_bytes(report) + b"\n")
    monkeypatch.setattr(
        launcher.detached,
        "_status",
        lambda _path: {"terminal": True},
    )

    with pytest.raises(
        launcher.ResidentEvaluationLaunchError,
        match="report hash is invalid",
    ):
        launcher.status(arguments)


def test_status_replays_stored_plan_without_rebuilding_current_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, _config, _checkpoint = _terminal_fixture(tmp_path, monkeypatch)
    plan = launcher.prepare(arguments)
    detached_root = Path(plan["evaluation_root"]) / "detached"
    detached_root.mkdir()
    (detached_root / launcher.detached.PLAN_FILE).write_text("{}\n", encoding="ascii")

    def reject_rebuild(_arguments: argparse.Namespace) -> dict[str, object]:
        raise AssertionError("status must inspect the immutable stored plan")

    monkeypatch.setattr(launcher, "_build_plan", reject_rebuild)
    monkeypatch.setattr(
        launcher.detached,
        "_status",
        lambda _path: {"terminal": False, "receipt": None},
    )

    result = launcher.status(arguments)

    assert result["state"] == "running"
    assert result["plan_sha256"] == plan["plan_sha256"]


def test_depth_parser_rejects_duplicate_and_shallow_recurrence() -> None:
    assert launcher._csv_positive_ints("1,2,4", minimum=1) == (1, 2, 4)
    with pytest.raises(argparse.ArgumentTypeError):
        launcher._csv_positive_ints("2,2", minimum=2)
    with pytest.raises(argparse.ArgumentTypeError):
        launcher._csv_positive_ints("1,4", minimum=2)
