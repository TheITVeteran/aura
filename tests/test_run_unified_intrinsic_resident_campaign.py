from __future__ import annotations

import hashlib
import json
import subprocess
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.run_unified_intrinsic_resident_campaign as controller
from tools import run_detached_step as detached
from tools.prepare_unified_intrinsic_resident_campaign import (
    _freeze_bootstrap_checkpoint,
    _profile_training,
    _training_cli,
)
from tools.unified_intrinsic_checkpoint import resolve_checkpoint_generation
from tools.unified_intrinsic_resident_identity import canonical_bytes, canonical_sha256


def _private(path: Path) -> Path:
    path.mkdir(parents=True)
    path.chmod(0o700)
    return path


def _model_manifest(root: Path) -> dict:
    files = [
        {"path": "config.json", "size_bytes": 10, "sha256": "4" * 64},
        {
            "path": "model-00001-of-00001.safetensors",
            "size_bytes": 20,
            "sha256": "5" * 64,
        },
        {"path": "tokenizer.json", "size_bytes": 30, "sha256": "6" * 64},
        {
            "path": "tokenizer_config.json",
            "size_bytes": 40,
            "sha256": "7" * 64,
        },
    ]
    body = {
        "schema": "aura.unified_intrinsic.model_manifest.v1",
        "root": str(root),
        "file_count": len(files),
        "files": files,
        "weights": ["model-00001-of-00001.safetensors"],
        "shard_index": None,
        "dimensions": {
            "model_type": "qwen2",
            "num_hidden_layers": 64,
            "hidden_size": 5120,
            "vocab_size": 152064,
            "quantization": {"bits": 4},
        },
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _config(tmp_path: Path, *, profile: str = "canary") -> tuple[Path, dict]:
    root = _private(tmp_path / "campaign")
    inputs = _private(root / "inputs")
    output = _private(root / "training-output")
    attempts = _private(root / "detached-attempts")
    model = _private(tmp_path / "resident-model")
    dataset = inputs / "dataset.json"
    tokenized = inputs / "tokenized_dataset.json"
    dataset.write_bytes(b"{}\n")
    tokenized.write_bytes(b"{}\n")
    dataset.chmod(0o400)
    tokenized.chmod(0o400)
    key = root / "heartbeat.key"
    key.write_bytes(b"k" * 32)
    key.chmod(0o400)
    training = _profile_training(profile)
    campaign_id = f"unit-{profile}"
    bootstrap = None
    if profile == "recovery":
        bootstrap_output = _private(inputs / "bootstrap-output")
        body_without_sha = {
            "schema": "aura.unified_intrinsic.bootstrap_input.v1",
            "stem": "checkpoint_latest",
            "output": str(bootstrap_output),
            "parent_step": 73,
            "parent_checkpoint_sha256": "c" * 64,
            "parent_receipt_sha256": "d" * 64,
            "parent_identity_sha256": "e" * 64,
        }
        bootstrap = {
            **body_without_sha,
            "bootstrap_sha256": canonical_sha256(body_without_sha),
        }
    body = {
        "schema": "aura.unified_intrinsic.resident_campaign.v1",
        "campaign_id": campaign_id,
        "profile": profile,
        "prepared_at": "2026-08-11T00:00:00+00:00",
        "source": {
            "git": {
                "root": str(tmp_path / "source"),
                "commit": "8" * 40,
                "tree": "9" * 40,
            },
            "manifest": {"manifest_sha256": "a" * 64},
        },
        "model": _model_manifest(model),
        "runtime": {
            "identity_sha256": "b" * 64,
            "interpreter": {"executable": "/usr/bin/python3"},
            "behavior_environment": {},
        },
        "dataset": {"identity_sha256": "1" * 64},
        "tokenizer": {"identity_sha256": "2" * 64},
        "tokenized_dataset": {"identity_sha256": "3" * 64},
        **({"bootstrap": bootstrap} if profile == "recovery" else {}),
        "paths": {
            "workspace_root": str(tmp_path),
            "campaign_root": str(root),
            "inputs": str(inputs),
            "training_output": str(output),
            "dataset": str(dataset),
            "tokenized_dataset": str(tokenized),
            "detached_attempts": str(attempts),
            "heartbeat_key": str(key),
            **(
                {"bootstrap_output": bootstrap["output"]}
                if bootstrap is not None
                else {}
            ),
        },
        "heartbeat_key_sha256": hashlib.sha256(b"k" * 32).hexdigest(),
        "training": training,
        "training_args": _training_cli(training),
        "watchdog": {
            "poll_interval_s": 15.0,
            "heartbeat_stale_s": 180.0,
            "attempt_timeout_s": 18000.0,
            "max_attempts": 8,
            "max_consecutive_no_progress": 2,
            "retry_backoff_s": 30.0,
        },
        "launch": {
            "label": f"com.aura.unified-intrinsic.{campaign_id}",
            "launchd_required": True,
            "trainer_caffeinate_required": True,
            "immutable_target_command": True,
        },
        "claims": {
            "resident_mechanics_proven": False,
            "reasoning_gain_proven": False,
            "frontier_level_proven": False,
            "fusion_allowed": False,
        },
    }
    config = {**body, "config_sha256": canonical_sha256(body)}
    path = root / "campaign.json"
    path.write_bytes(canonical_bytes(config) + b"\n")
    path.chmod(0o400)
    return path, config


def test_config_separates_immutable_inputs_from_mutable_output(tmp_path: Path) -> None:
    path, expected = _config(tmp_path)

    observed = controller._load_config(path)

    assert observed == expected
    inputs = Path(observed["paths"]["inputs"])
    output = Path(observed["paths"]["training_output"])
    assert Path(observed["paths"]["dataset"]).is_relative_to(inputs)
    assert Path(observed["paths"]["tokenized_dataset"]).is_relative_to(inputs)
    assert not inputs.is_relative_to(output)
    assert not output.is_relative_to(inputs)


def test_config_rejects_writable_or_symlinked_campaign_identity(tmp_path: Path) -> None:
    path, _expected = _config(tmp_path)
    path.chmod(0o600)
    with pytest.raises(
        controller.UnifiedResidentControllerError,
        match="artifact_custody_invalid",
    ):
        controller._load_config(path)

    path.chmod(0o400)
    link = tmp_path / "campaign-link.json"
    link.symlink_to(path)
    with pytest.raises(
        controller.UnifiedResidentControllerError,
        match="campaign_config_path_is_symlink",
    ):
        controller._load_config(link)


def test_trainer_command_targets_resident_model_and_pid_scoped_guards(
    tmp_path: Path,
) -> None:
    path, raw = _config(tmp_path)
    config = controller._load_config(path)

    command = controller._trainer_command(
        path,
        config,
        invocation_steps=1,
    )

    assert command[command.index("--model") + 1] == raw["model"]["root"]
    assert "--exclusive-model-lane" in command
    assert command[command.index("--out-dir") + 1] == raw["paths"]["training_output"]
    assert command[command.index("--dataset") + 1] == raw["paths"]["dataset"]
    assert command[command.index("--tokenized-dataset") + 1] == (
        raw["paths"]["tokenized_dataset"]
    )
    assert "{pid}" in command[command.index("--resource-stage-path") + 1]
    assert "{pid}" in command[command.index("--preload-ready-path") + 1]
    assert "{pid}" in command[command.index("--preload-release-path") + 1]
    assert command[-2:] == ["--max-invocation-steps", "1"]
    assert command[command.index("--answer-bridge-inner-steps") + 1] == "32"
    assert raw["training"]["per_cell"] == 4
    assert raw["training"]["answer_bridge_steps"] == 36
    assert raw["training"]["eval_every"] == 9
    assert raw["training"]["max_steps"] == 37
    assert raw["training"]["seed"] == 20260811433
    assert raw["training"]["init_seed"] == 20260811433


def test_full_profile_uses_the_decode_admitted_cached_bridge_schedule() -> None:
    training = _profile_training("full")

    assert training["per_cell"] == 8
    assert training["holdout_per_cell"] == 3
    assert training["answer_bridge_steps"] == 72
    assert training["answer_bridge_inner_steps"] == 32
    assert training["eval_every"] == 9
    assert training["max_steps"] == 73


def test_process_canary_trains_scoped_tissue_on_autonomous_frontier_process() -> None:
    training = _profile_training("process_canary")
    arguments = _training_cli(training)

    assert training["window_tissue_mode"] == "scoped_lora"
    assert training["task_source"] == "frontier_process"
    assert training["per_cell"] == 2
    assert training["holdout_per_cell"] == 1
    assert training["state_warmup_steps"] == 28
    assert training["answer_bridge_steps"] == 0
    assert training["max_steps"] == 42
    assert training["state_teacher_forcing_probability"] == 1.0
    assert training["state_teacher_forcing_final_probability"] == 0.0
    assert arguments[arguments.index("--task-source") + 1] == "frontier_process"
    assert arguments[arguments.index("--window-tissue-mode") + 1] == "scoped_lora"


def test_recovery_profile_warm_starts_fresh_data_without_parent_optimizer() -> None:
    training = _profile_training("recovery")

    assert training["per_cell"] == 8
    assert training["holdout_per_cell"] == 3
    assert training["answer_bridge_steps"] == 18
    assert training["max_steps"] == 36
    assert training["seed"] != training["init_seed"]


def test_recovery_config_binds_parent_checkpoint_and_trainer_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, raw = _config(tmp_path, profile="recovery")
    resolved = SimpleNamespace(
        receipt={
            "step": 73,
            "checkpoint_sha256": "c" * 64,
            "receipt_sha256": "d" * 64,
            "identity": {"identity_sha256": "e" * 64},
        }
    )
    monkeypatch.setattr(
        controller,
        "resolve_checkpoint_generation",
        lambda *_args, **_kwargs: resolved,
    )

    config = controller._load_config(path)
    command = controller._trainer_command(path, config, invocation_steps=1)

    assert "--resume-if-available" in command
    assert command[command.index("--bootstrap-output-dir") + 1] == raw["paths"][
        "bootstrap_output"
    ]
    assert command[command.index("--bootstrap-stem") + 1] == "checkpoint_latest"
    assert command[-2:] == ["--max-invocation-steps", "1"]
    assert controller._planned_invocation_steps(config, {"step": 0}) == 1
    assert controller._planned_invocation_steps(config, {"step": 1}) == 35


def test_bootstrap_freeze_copies_one_authoritative_generation(tmp_path: Path) -> None:
    source = _private(tmp_path / "source-output")
    generations = _private(source / "checkpoint_generations")
    checkpoint_id = "checkpoint_latest-step-00000073-" + "a" * 32
    generation = _private(generations / checkpoint_id)
    weights = generation / "bundle.safetensors"
    weights.write_bytes(b"controller-tissue")
    weights.chmod(0o400)
    identity = {"identity_sha256": "e" * 64}
    receipt_body = {
        "schema": "aura.unified_intrinsic_training.v1",
        "checkpoint_generation_schema": "aura.unified_intrinsic_checkpoint.v3",
        "checkpoint_id": checkpoint_id,
        "checkpoint_file": weights.name,
        "checkpoint_size_bytes": weights.stat().st_size,
        "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "identity": identity,
        "step": 73,
        "stem": "checkpoint_latest",
    }
    receipt = {
        **receipt_body,
        "receipt_sha256": canonical_sha256(receipt_body),
    }
    complete = generation / "complete.json"
    complete.write_bytes(canonical_bytes(receipt) + b"\n")
    complete.chmod(0o400)
    generation.chmod(0o500)
    pointer = {
        "schema": "aura.unified_intrinsic_checkpoint_pointer.v2",
        "checkpoint": f"checkpoint_generations/{checkpoint_id}",
        "complete_sha256": hashlib.sha256(complete.read_bytes()).hexdigest(),
        "identity_sha256": identity["identity_sha256"],
        "step": 73,
        "stem": "checkpoint_latest",
    }
    pointer_path = source / "checkpoint_latest_pointer.json"
    pointer_path.write_bytes(canonical_bytes(pointer) + b"\n")
    pointer_path.chmod(0o600)
    inputs = _private(tmp_path / "inputs")

    frozen = _freeze_bootstrap_checkpoint(
        source,
        inputs,
        stem="checkpoint_latest",
    )
    selected = resolve_checkpoint_generation(
        inputs / "bootstrap-output",
        stem="checkpoint_latest",
        required=True,
    )

    assert selected is not None
    assert selected.weights_path.read_bytes() == b"controller-tissue"
    assert frozen["parent_checkpoint_sha256"] == receipt["checkpoint_sha256"]
    assert frozen["parent_receipt_sha256"] == receipt["receipt_sha256"]
    assert frozen["parent_identity_sha256"] == identity["identity_sha256"]


def test_clean_child_exit_waits_for_signed_detached_terminal_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses = iter(
        [
            {"terminal": False, "child_state": "dead"},
            {
                "terminal": True,
                "child_state": "dead",
                "state": "passed",
                "receipt": {"containment_verified": True, "returncode": 0},
            },
        ]
    )
    monkeypatch.setattr(controller.detached, "_status", lambda _run_dir: next(statuses))
    monkeypatch.setattr(controller.time, "sleep", lambda _seconds: None)

    status = controller._await_detached_terminal_handoff(
        tmp_path / "attempt",
        timeout_s=1.0,
    )

    assert status is not None
    assert status["terminal"] is True
    assert status["receipt"]["containment_verified"] is True


def test_missing_terminal_handoff_remains_an_identity_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        controller.detached,
        "_status",
        lambda _run_dir: {"terminal": False, "child_state": "dead"},
    )

    assert (
        controller._await_detached_terminal_handoff(
            tmp_path / "attempt",
            timeout_s=0.0,
        )
        is None
    )


def test_signed_controller_status_rejects_tampering(tmp_path: Path) -> None:
    path, _raw = _config(tmp_path)
    config = controller._load_config(path)
    first = controller._publish_status(config, "validating", {"phase": 1})
    second = controller._publish_status(config, "validating", {"phase": 2})

    assert first["sequence"] == 1
    assert second["sequence"] == 2
    assert controller._read_status(config, required=True) == second

    status_path = Path(config["paths"]["campaign_root"]) / "controller-status.json"
    tampered = json.loads(status_path.read_text(encoding="ascii"))
    tampered["state"] = "completed"
    status_path.write_bytes(canonical_bytes(tampered) + b"\n")
    with pytest.raises(
        controller.UnifiedResidentControllerError,
        match="controller_status_authentication_failed",
    ):
        controller._read_status(config, required=True)


def test_status_never_reports_dead_nonterminal_controller_as_live(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _raw = _config(tmp_path)
    config = controller._load_config(path)
    controller._publish_status(config, "training", {"step": 3})
    monkeypatch.setattr(controller.detached, "_identity_state", lambda *_args: "dead")

    inspection = controller._inspect_status(config)

    assert inspection["effective_state"] == "stale"
    assert inspection["controller_liveness"] == "dead"
    assert inspection["claims_supported"] == []


def test_occupied_model_lane_is_reported_and_never_evicted(monkeypatch) -> None:
    owner = SimpleNamespace(
        owner_id="live-aura",
        model_path="/resident/32b",
        purpose="serve",
        declared_gb=24.0,
        observed_gb=20.0,
        process=SimpleNamespace(pid=321, started_at=123.0),
        preemptible=False,
        metadata={"live_runtime": True},
    )
    lane = SimpleNamespace(owner_observations=lambda: [owner])
    monkeypatch.setattr(controller, "get_model_lane_controller", lambda: lane)

    with pytest.raises(
        controller.UnifiedResidentControllerError,
        match="resident_model_lane_occupied",
    ) as caught:
        controller._require_empty_model_lane()

    assert caught.value.details["owners"][0]["pid"] == 321


def test_empty_checkpoint_output_is_step_zero(tmp_path: Path) -> None:
    path, _raw = _config(tmp_path)
    config = controller._load_config(path)

    assert controller._checkpoint_snapshot(config) == {
        "present": False,
        "step": 0,
        "checkpoint_sha256": None,
        "receipt_sha256": None,
        "complete": False,
        "training_receipt": None,
    }


def test_canary_never_launches_zero_steps_at_terminal_checkpoint(
    tmp_path: Path,
) -> None:
    path, _raw = _config(tmp_path)
    config = controller._load_config(path)
    checkpoint = {
        "present": True,
        "step": config["training"]["max_steps"],
        "complete": False,
        "training_receipt": {
            "binding": "ignored_non_authoritative",
            "reason": "artifact_not_canonical",
        },
    }

    with pytest.raises(
        controller.UnifiedResidentControllerError,
        match="terminal_receipt_unavailable_at_max_step",
    ):
        controller._planned_invocation_steps(config, checkpoint)


def test_canary_plans_first_and_remaining_invocation_steps(tmp_path: Path) -> None:
    path, _raw = _config(tmp_path)
    config = controller._load_config(path)
    assert controller._planned_invocation_steps(config, {"step": 0}) == 1
    assert controller._planned_invocation_steps(config, {"step": 1}) == (
        config["training"]["max_steps"] - 1
    )


def test_first_checkpoint_crash_debris_is_retryable_but_never_authoritative(
    tmp_path: Path,
) -> None:
    path, _raw = _config(tmp_path)
    config = controller._load_config(path)
    output = Path(config["paths"]["training_output"])
    generations = output / "checkpoint_generations"
    generations.mkdir(mode=0o700)
    orphan = generations / f"checkpoint_latest-step-00000001-{'d' * 32}"
    orphan.mkdir(mode=0o500)
    staging = generations / f".checkpoint-stage-{'e' * 32}"
    staging.mkdir(mode=0o700)

    snapshot = controller._checkpoint_snapshot(config)

    assert snapshot["present"] is False
    assert snapshot["step"] == 0
    assert snapshot["ignored_unpointed"] == {
        "orphan_generations": 1,
        "staged_generations": 1,
    }


def test_dead_supervisor_triggers_verified_orphan_containment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "attempt"
    contained: list[dict] = []
    monkeypatch.setattr(
        controller.detached,
        "_stop",
        lambda _run_dir: {"stopped": False, "reason": "supervisor_not_alive"},
    )
    monkeypatch.setattr(
        controller.detached,
        "_status",
        lambda _run_dir: {
            "terminal": False,
            "supervisor_state": "dead",
            "child_pid": 321,
            "child_process_group_id": 321,
            "child_start_token": "child-start",
            "containment_token": "a" * 64,
            "child_state": "alive",
            "receipt": None,
        },
    )
    monkeypatch.setattr(
        controller.detached,
        "_terminate_stale_target",
        lambda target: contained.append(dict(target)),
    )
    monkeypatch.setattr(controller.detached, "_identity_state", lambda *_args: "dead")
    monkeypatch.setattr(controller.detached, "_process_group_exists", lambda _pgid: False)
    monkeypatch.setattr(controller.detached, "_tagged_processes", lambda _token: [])

    controller._stop_detached(run_dir, code="containment_failed")

    assert contained == [
        {
            "child_pid": 321,
            "child_process_group_id": 321,
            "child_start_token": "child-start",
            "containment_token": "a" * 64,
        }
    ]


def test_dead_supervisor_never_accepts_uncontained_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "attempt"
    monkeypatch.setattr(
        controller.detached,
        "_stop",
        lambda _run_dir: {"stopped": False, "reason": "supervisor_not_alive"},
    )
    monkeypatch.setattr(
        controller.detached,
        "_status",
        lambda _run_dir: {
            "terminal": False,
            "supervisor_state": "dead",
            "child_pid": 321,
            "child_process_group_id": 321,
            "child_start_token": "child-start",
            "containment_token": "a" * 64,
            "child_state": "alive",
            "receipt": None,
        },
    )
    monkeypatch.setattr(controller.detached, "_terminate_stale_target", lambda _target: True)
    monkeypatch.setattr(controller.detached, "_identity_state", lambda *_args: "dead")
    monkeypatch.setattr(controller.detached, "_process_group_exists", lambda _pgid: False)
    monkeypatch.setattr(
        controller.detached,
        "_tagged_processes",
        lambda _token: [(322, SimpleNamespace(token="descendant"))],
    )

    with pytest.raises(
        controller.UnifiedResidentControllerError,
        match="containment_failed",
    ):
        controller._stop_detached(run_dir, code="containment_failed")


def test_caffeinate_liveness_binds_pid_parent_start_and_command(monkeypatch) -> None:
    evidence = {
        "caffeinate": {
            "pid": 501,
            "start_token": "caffeinate-start",
            "parent_pid": 500,
            "command": ["/usr/bin/caffeinate", "-i", "-w", "321"],
        }
    }
    process = SimpleNamespace(
        pid=501,
        ppid=500,
        cmdline=("/usr/bin/caffeinate", "-i", "-w", "321"),
    )
    monkeypatch.setattr(controller.detached, "_identity_state", lambda *_args: "alive")
    monkeypatch.setattr(
        controller,
        "get_resource_observer",
        lambda: SimpleNamespace(processes=lambda: [process]),
    )

    assert controller._caffeinate_is_live(evidence) is True
    process.ppid = 499
    assert controller._caffeinate_is_live(evidence) is False


def test_file_lock_waits_for_bounded_launch_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def flock(_descriptor: int, operation: int) -> None:
        nonlocal attempts
        if operation & controller.fcntl.LOCK_NB:
            attempts += 1
            if attempts < 3:
                raise BlockingIOError

    monkeypatch.setattr(controller.fcntl, "flock", flock)
    monkeypatch.setattr(controller.time, "sleep", lambda _seconds: None)

    with controller._file_lock(
        tmp_path / "host.lock",
        busy_code="host_busy",
        wait_s=1.0,
    ):
        pass

    assert attempts == 3


def test_controller_waits_for_installer_host_lease_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _raw = _config(tmp_path)
    observed_waits: list[float] = []

    @contextmanager
    def lease(_config, *, wait_s: float = 0.0):
        observed_waits.append(wait_s)
        yield {}

    monkeypatch.setattr(controller, "_host_lease", lease)
    monkeypatch.setattr(
        controller,
        "_verify_launchd",
        lambda *_args, **_kwargs: controller._fail("stop_after_handoff"),
    )

    with pytest.raises(
        controller.UnifiedResidentControllerError,
        match="stop_after_handoff",
    ):
        controller.run_controller(path, launchd_supervised=True)

    assert observed_waits == [controller.HOST_LEASE_HANDOFF_TIMEOUT_S]


def test_controller_stops_after_canonical_resource_guard_intervention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _raw = _config(tmp_path)
    sentinel = _private(tmp_path / "sentinel")
    tombstones = _private(sentinel / "tombstones")
    tombstone = {
        "schema": "aura.memory_sentinel.tombstone.v1",
        "reason": "external sentinel killed process tree at lethal ceiling",
        "guard_stage": "steady",
        "killed_pids": [4242],
        "final_sample": {"managed_mb": 51_515.0, "lethal_mb": 49_152.0},
    }
    tombstone_path = tombstones / "sentinel_tombstone_1.json"
    tombstone_path.write_bytes(canonical_bytes(tombstone) + b"\n")
    tombstone_path.chmod(0o400)
    snapshots = iter(
        (
            {"step": 0, "complete": False},
            {"step": 1, "complete": False},
        )
    )
    monkeypatch.setattr(controller, "_host_lease", lambda *_args, **_kwargs: nullcontext({}))
    monkeypatch.setattr(controller, "_verify_launchd", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(controller, "verify_package", lambda _config: {"valid": True})
    monkeypatch.setattr(controller, "_publish_status", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(controller, "_require_empty_model_lane", lambda: [])
    monkeypatch.setattr(controller, "_checkpoint_snapshot", lambda _config: next(snapshots))
    monkeypatch.setattr(
        controller,
        "_monitor_attempt",
        lambda *_args, **_kwargs: (
            {"plan_sha256": "a" * 64, "receipt": {"returncode": -9}},
            {"sentinel_run_dir": str(sentinel)},
        ),
    )

    with pytest.raises(
        controller.UnifiedResidentControllerError,
        match="resource_guard_intervention_requires_repair",
    ):
        controller.run_controller(path, launchd_supervised=True)

    result = json.loads(
        (path.parent / "attempt-results/attempt-0001.json").read_text(encoding="ascii")
    )
    intervention = result["resource_guard_intervention"]
    assert intervention["reason"] == tombstone["reason"]
    assert intervention["tombstone_sha256"] == canonical_sha256(tombstone)
    assert intervention["intervention_sha256"]


def test_install_publishes_intent_before_launch_and_receipts_exact_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _raw = _config(tmp_path)
    launch_agents = _private(tmp_path / "LaunchAgents")
    monkeypatch.setattr(controller, "LAUNCH_AGENTS_ROOT", launch_agents)
    monkeypatch.setattr(controller, "TRAINING_STATE_ROOT", _private(tmp_path / "state"))
    monkeypatch.setattr(controller, "verify_package", lambda _config: {"valid": True})
    monkeypatch.setattr(controller, "_require_empty_model_lane", lambda: [])
    monkeypatch.setattr(controller, "_retire_stale_launchd_jobs", lambda _label: [])
    monkeypatch.setattr(controller, "_host_lease", lambda _config: nullcontext({}))
    monkeypatch.setattr(
        controller,
        "_launchd_job",
        lambda label: {"target": f"gui/501/{label}", "pid": 777},
    )
    monkeypatch.setattr(
        controller.detached,
        "_process_start_token",
        lambda pid: "controller-start-token" if pid == 777 else "installer-token",
    )

    def run(command, **_kwargs):
        if command[1] == "bootstrap":
            root = path.parent
            assert (root / "launch-intent.json").is_file()
            assert next(launch_agents.glob("*.plist")).is_file()
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(controller.subprocess, "run", run)

    receipt = controller.install_launchd(path)

    assert receipt["pid"] == 777
    assert receipt["start_token"] == "controller-start-token"
    assert receipt["launch_intent_sha256"]
    assert (path.parent / "launchd-receipt-pointer.json").is_file()


def test_install_failure_after_bootstrap_rolls_back_exact_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _raw = _config(tmp_path)
    monkeypatch.setattr(controller, "LAUNCH_AGENTS_ROOT", _private(tmp_path / "agents"))
    monkeypatch.setattr(controller, "TRAINING_STATE_ROOT", _private(tmp_path / "state"))
    monkeypatch.setattr(controller, "verify_package", lambda _config: {"valid": True})
    monkeypatch.setattr(controller, "_require_empty_model_lane", lambda: [])
    monkeypatch.setattr(controller, "_retire_stale_launchd_jobs", lambda _label: [])
    monkeypatch.setattr(controller, "_host_lease", lambda _config: nullcontext({}))
    monkeypatch.setattr(
        controller,
        "_launchd_job",
        lambda label: {"target": f"gui/501/{label}", "pid": 777},
    )
    monkeypatch.setattr(controller.detached, "_process_start_token", lambda _pid: "token")
    monkeypatch.setattr(
        controller.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    rolled_back: list[str] = []
    monkeypatch.setattr(
        controller,
        "_rollback_launch",
        lambda config: rolled_back.append(str(config["campaign_id"])),
    )
    original_write_once = controller._write_once

    def fail_receipt(target: Path, value, **kwargs):
        if target.parent.name == "launchd-receipts":
            raise OSError("receipt disk failure")
        return original_write_once(target, value, **kwargs)

    monkeypatch.setattr(controller, "_write_once", fail_receipt)

    with pytest.raises(OSError, match="receipt disk failure"):
        controller.install_launchd(path)

    assert rolled_back == ["unit-canary"]


def test_detached_plan_hashes_inputs_but_excludes_generated_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    tools = source / "tools"
    tools.mkdir(parents=True)
    for name in (
        "train_unified_intrinsic_recurrence.py",
        "unified_intrinsic_preload_barrier.py",
        "verify_unified_intrinsic_resume.py",
    ):
        (tools / name).write_text("raise SystemExit(0)\n", encoding="ascii")
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "add", "tools"], cwd=source, check=True)
    path, _raw = _config(tmp_path)
    config = controller._load_config(path)
    run_dir = Path(config["paths"]["detached_attempts"]) / "attempt-0001"
    arguments = controller._main_launch_args(
        path,
        config,
        run_dir,
        invocation_steps=1,
        resume=False,
    )
    parsed = detached.build_parser().parse_args(arguments)
    verifier = json.loads(parsed.resume_verifier_json)

    plan = detached._build_plan(
        "unit-plan",
        parsed.command,
        Path(parsed.cwd),
        parsed.timeout,
        parsed.resume_contract,
        verifier,
        execution_exclusion_roots=(
            run_dir,
            Path(config["paths"]["training_output"]),
        ),
    )

    excluded = plan["target_execution_manifest"]["excluded_roots"]
    assert str(Path(config["paths"]["training_output"])) in excluded
    roots = plan["target_execution_manifest"]["roots"]
    assert any(row["path"] == str(source) for row in roots)
    assert any(row["path"] == config["paths"]["dataset"] for row in roots)
    assert any(row["path"] == config["paths"]["tokenized_dataset"] for row in roots)
