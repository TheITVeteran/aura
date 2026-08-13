from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import launch_unified_intrinsic_resident_evaluation as launcher


def _arguments(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        action="prepare",
        campaign=root,
        output=None,
        evaluator_source_root=None,
        matched_control_campaign=None,
        matched_control_stem="checkpoint_latest",
        stem="checkpoint_answer_bridge_admitted",
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


def test_cli_defaults_to_admitted_checkpoint_and_all_trained_task_depths(
    tmp_path: Path,
) -> None:
    arguments = launcher._parser().parse_args(["prepare", str(tmp_path)])

    assert arguments.stem == "checkpoint_answer_bridge_admitted"
    assert arguments.task_depths == (1, 2, 4)


def test_evaluation_source_manifest_freezes_the_runtime_identity_repair() -> None:
    assert (
        "core/brain/llm/latent_cortex/recurrence_adapter_identity_v2.py"
        in launcher.EVALUATION_SOURCE_FILES
    )


def _terminal_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    completion_checkpoint: dict[str, object] | None = None,
    training_verdict: str = "answer_bridge_admitted",
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
    source_files = tuple(source / relative for relative in launcher.EVALUATION_SOURCE_FILES)
    for path in (
        python,
        *source_files,
        source / "tools/unified_intrinsic_preload_barrier.py",
        source / "tools/memory_sentinel.py",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
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
    (root / "completion-receipt.json").write_bytes(launcher.canonical_bytes(completion) + b"\n")
    monkeypatch.setattr(launcher.resident, "_load_config", lambda _path: config)
    monkeypatch.setattr(
        launcher.resident,
        "_checkpoint_snapshot",
        lambda _config: checkpoint,
    )
    selected_receipt = {
        "step": checkpoint["step"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "receipt_sha256": checkpoint["receipt_sha256"],
        "identity": {
            "identity_sha256": "e" * 64,
            "initial_controller_sha256": "f" * 64,
        },
    }
    monkeypatch.setattr(
        launcher,
        "resolve_checkpoint_generation",
        lambda *_args, **_kwargs: SimpleNamespace(receipt=selected_receipt),
    )
    admission_body = {
        "schema": "aura.unified_intrinsic.answer_bridge_admission.v3",
        "admitted": True,
        "exact": 1,
        "tasks": 1,
    }
    admission = {
        **admission_body,
        "admission_sha256": launcher.canonical_sha256(admission_body),
    }
    training_body = {
        "schema": "aura.unified_intrinsic_training.v1",
        "steps": checkpoint["step"],
        "verdict": training_verdict,
        "answer_bridge_admission": admission,
        "identity": {"dataset": {"holdout_count": 1}},
    }
    training_receipt = {
        **training_body,
        "receipt_sha256": launcher.canonical_sha256(training_body),
    }
    checkpoint["training_receipt"]["receipt_sha256"] = training_receipt["receipt_sha256"]
    (output / "training_receipt.json").write_bytes(
        launcher.canonical_bytes(training_receipt) + b"\n"
    )
    completion_body["checkpoint"] = completion_checkpoint or checkpoint
    completion = {
        **completion_body,
        "completion_sha256": launcher.canonical_sha256(completion_body),
    }
    (root / "completion-receipt.json").write_bytes(launcher.canonical_bytes(completion) + b"\n")
    return _arguments(root), config, checkpoint


def test_prepare_binds_terminal_checkpoint_and_frozen_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, config, checkpoint = _terminal_fixture(tmp_path, monkeypatch)

    plan = launcher.prepare(arguments)

    assert plan["scientific"]["checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
    assert plan["scientific"]["checkpoint"]["stem"] == arguments.stem
    assert plan["scientific"]["matched_control"] == {
        "schema": "aura.unified_intrinsic.matched_control_binding.v1",
        "mode": "campaign_episode_initial",
        "campaign_root": str(arguments.campaign),
        "campaign_identity_sha256": "e" * 64,
        "controller_sha256": "f" * 64,
        "binding_sha256": launcher.canonical_sha256(
            {
                "schema": "aura.unified_intrinsic.matched_control_binding.v1",
                "mode": "campaign_episode_initial",
                "campaign_root": str(arguments.campaign),
                "campaign_identity_sha256": "e" * 64,
                "controller_sha256": "f" * 64,
            }
        ),
    }
    assert plan["scientific"]["checkpoint"]["answer_bridge_admission"] == {
        "admission_sha256": launcher.canonical_sha256(
            {
                "schema": "aura.unified_intrinsic.answer_bridge_admission.v3",
                "admitted": True,
                "exact": 1,
                "tasks": 1,
            }
        ),
        "tasks": 1,
        "exact": 1,
    }
    assert plan["command"][0] == config["runtime"]["interpreter"]["executable"]
    assert plan["scientific"]["evaluator_source_root"] == config["source"]["git"]["root"]
    assert plan["scientific"]["evaluation_source_sha256s"] == {
        relative: launcher._file_sha256(  # noqa: SLF001
            Path(config["source"]["git"]["root"]) / relative
        )
        for relative in launcher.EVALUATION_SOURCE_FILES
    }
    assert plan["evaluator_command"][1].startswith(str(Path(config["source"]["git"]["root"])))
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
    progress_index = plan["evaluator_command"].index("--progress-dir")
    assert plan["evaluator_command"][progress_index + 1] == plan["progress_dir"]
    assert plan["claims_not_supported"] == [
        "broad_reasoning_gain",
        "frontier_performance",
        "production_fusion",
        "wow_signal",
    ]
    stored = Path(plan["evaluation_root"]) / "evaluation-plan.json"
    assert stored.stat().st_mode & 0o777 == 0o400
    assert launcher._read_document(stored) == plan


def test_prepare_binds_repaired_evaluator_source_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, config, _checkpoint = _terminal_fixture(tmp_path, monkeypatch)
    repaired = tmp_path / "repaired-capsule"
    (repaired / "tools").mkdir(parents=True)
    training_source = Path(config["source"]["git"]["root"])
    for relative in (
        *launcher.EVALUATION_SOURCE_FILES,
        "tools/unified_intrinsic_preload_barrier.py",
        "tools/memory_sentinel.py",
    ):
        destination = repaired / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((training_source / relative).read_bytes())
    arguments.evaluator_source_root = repaired

    plan = launcher.prepare(arguments)

    assert plan["scientific"]["evaluator_source_root"] == str(repaired)
    assert plan["scientific"]["training_source_root"] == str(training_source)
    assert plan["evaluator_command"][1] == str(
        repaired / "tools/evaluate_unified_intrinsic_decoding.py"
    )


def test_prepare_binds_an_explicit_root_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, _config, _checkpoint = _terminal_fixture(tmp_path, monkeypatch)
    control = tmp_path / "root-control"
    control.mkdir()
    arguments.matched_control_campaign = control
    binding = {
        "schema": "aura.unified_intrinsic.root_control_binding.v1",
        "mode": "deterministic_pretraining_root",
        "campaign_root": str(control),
        "stem": "checkpoint_latest",
        "controller_sha256": "1" * 64,
        "binding_sha256": "2" * 64,
    }
    calls: list[tuple[Path, str, dict[str, object]]] = []

    def bind(path: Path, *, stem: str, target_identity: dict[str, object]):
        calls.append((path, stem, target_identity))
        return binding

    monkeypatch.setattr(launcher, "root_control_binding", bind)

    plan = launcher.prepare(arguments)

    assert calls and calls[0][0] == control
    assert plan["scientific"]["matched_control"] == binding
    index = plan["evaluator_command"].index("--matched-control-campaign")
    assert plan["evaluator_command"][index : index + 4] == [
        "--matched-control-campaign",
        str(control),
        "--matched-control-stem",
        "checkpoint_latest",
    ]


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


def test_prepare_rejects_missing_admitted_checkpoint_before_model_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, _config, _checkpoint = _terminal_fixture(tmp_path, monkeypatch)

    def missing(*_args: object, **_kwargs: object) -> None:
        raise launcher.UnifiedCheckpointError("missing")

    monkeypatch.setattr(launcher, "resolve_checkpoint_generation", missing)

    with pytest.raises(
        launcher.ResidentEvaluationLaunchError,
        match="checkpoint is unavailable",
    ):
        launcher.prepare(arguments)


def test_prepare_accepts_admission_independent_of_summary_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, _config, checkpoint = _terminal_fixture(
        tmp_path,
        monkeypatch,
        training_verdict="heldout_depth_gain",
    )

    plan = launcher.prepare(arguments)

    assert plan["scientific"]["checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
    assert plan["scientific"]["checkpoint"]["answer_bridge_admission"]["exact"] == 1


def test_prepare_rejects_tampered_answer_bridge_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, config, _checkpoint = _terminal_fixture(tmp_path, monkeypatch)
    output = Path(str(config["paths"]["training_output"]))
    receipt = launcher._read_document(output / "training_receipt.json")
    receipt["answer_bridge_admission"]["exact"] = 0
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = launcher.canonical_sha256(body)
    (output / "training_receipt.json").write_bytes(launcher.canonical_bytes(receipt) + b"\n")

    with pytest.raises(
        launcher.ResidentEvaluationLaunchError,
        match="requires an admitted answer-bridge checkpoint",
    ):
        launcher.prepare(arguments)


def test_prepare_rejects_admitted_generation_for_a_different_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, _config, checkpoint = _terminal_fixture(tmp_path, monkeypatch)
    stale_receipt = {
        "step": checkpoint["step"],
        "checkpoint_sha256": "f" * 64,
        "receipt_sha256": checkpoint["receipt_sha256"],
        "identity": {"identity_sha256": "e" * 64},
    }
    monkeypatch.setattr(
        launcher,
        "resolve_checkpoint_generation",
        lambda *_args, **_kwargs: SimpleNamespace(receipt=stale_receipt),
    )

    with pytest.raises(
        launcher.ResidentEvaluationLaunchError,
        match="requires an admitted answer-bridge checkpoint",
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
    assert result["attempt"] == 1
    assert "detached-attempts/attempt-0001" in result["run_dir"]
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
        "schema": "aura.unified_intrinsic_decode_evaluation.v2",
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


def test_status_accepts_hash_valid_legacy_pretty_report_without_rewriting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, _config, checkpoint = _terminal_fixture(tmp_path, monkeypatch)
    plan = launcher.prepare(arguments)
    detached_root = Path(plan["evaluation_root"]) / "detached"
    detached_root.mkdir()
    (detached_root / launcher.detached.PLAN_FILE).write_text("{}\n", encoding="ascii")
    body = {
        "schema": "aura.unified_intrinsic_decode_evaluation.v2",
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "evaluation_seed": plan["scientific"]["evaluation_seed"],
        "per_cell": plan["scientific"]["per_cell"],
        "max_tokens": plan["scientific"]["max_tokens"],
        "task_depths": plan["scientific"]["task_depths"],
        "recurrence_depths": plan["scientific"]["recurrence_depths"],
        "evaluation_source_sha256s": plan["scientific"]["evaluation_source_sha256s"],
        "matched_control": plan["scientific"]["matched_control"],
    }
    report = {**body, "report_sha256": launcher.canonical_sha256(body)}
    pretty = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("ascii")
    Path(plan["report"]).write_bytes(pretty)
    monkeypatch.setattr(
        launcher.detached,
        "_status",
        lambda _path: {"terminal": True, "receipt": {"passed": True}},
    )

    result = launcher.status(arguments)

    assert result["state"] == "completed"
    assert result["report"] == report
    assert result["report_transport"] == {
        "file_sha256": hashlib.sha256(pretty).hexdigest(),
        "canonical_bytes": False,
        "size_bytes": len(pretty),
    }
    assert Path(plan["report"]).read_bytes() == pretty


def test_status_rejects_a_historical_report_schema_for_a_v3_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, _config, checkpoint = _terminal_fixture(tmp_path, monkeypatch)
    plan = launcher.prepare(arguments)
    detached_root = Path(plan["evaluation_root"]) / "detached"
    detached_root.mkdir()
    (detached_root / launcher.detached.PLAN_FILE).write_text("{}\n", encoding="ascii")
    body = {
        "schema": "aura.unified_intrinsic_decode_evaluation.v1",
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "evaluation_seed": plan["scientific"]["evaluation_seed"],
        "per_cell": plan["scientific"]["per_cell"],
        "max_tokens": plan["scientific"]["max_tokens"],
        "task_depths": plan["scientific"]["task_depths"],
        "recurrence_depths": plan["scientific"]["recurrence_depths"],
        "evaluation_source_sha256s": plan["scientific"]["evaluation_source_sha256s"],
        "matched_control": plan["scientific"]["matched_control"],
    }
    report = {**body, "report_sha256": launcher.canonical_sha256(body)}
    Path(plan["report"]).write_bytes(launcher.canonical_bytes(report) + b"\n")
    monkeypatch.setattr(
        launcher.detached,
        "_status",
        lambda _path: {"terminal": True, "receipt": {"passed": True}},
    )

    with pytest.raises(
        launcher.ResidentEvaluationLaunchError,
        match="differs from its frozen plan",
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


def test_status_uses_latest_immutable_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, _config, _checkpoint = _terminal_fixture(tmp_path, monkeypatch)
    plan = launcher.prepare(arguments)
    attempts = Path(plan["evaluation_root"]) / "detached-attempts"
    first = attempts / "attempt-0001"
    second = attempts / "attempt-0002"
    for path in (first, second):
        path.mkdir(parents=True)
        (path / launcher.detached.PLAN_FILE).write_text("{}\n", encoding="ascii")

    monkeypatch.setattr(
        launcher.detached,
        "_status",
        lambda path: {
            "terminal": path == first,
            "receipt": {"passed": False} if path == first else None,
        },
    )

    result = launcher.status(arguments)

    assert result["state"] == "running"
    assert result["attempt"] == 2
    assert result["attempt_count"] == 2
    assert result["run_dir"] == str(second)


def test_depth_parser_rejects_duplicate_and_shallow_recurrence() -> None:
    assert launcher._csv_positive_ints("1,2,4", minimum=1) == (1, 2, 4)
    with pytest.raises(argparse.ArgumentTypeError):
        launcher._csv_positive_ints("2,2", minimum=2)
    with pytest.raises(argparse.ArgumentTypeError):
        launcher._csv_positive_ints("1,4", minimum=2)
