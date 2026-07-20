#!/usr/bin/env python3
"""Launch and verify certified resident-v3 checkpoint recovery phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.learning.recurrence_checkpoint_migration import (  # noqa: E402
    verify_migration,
)
from core.runtime.atomic_writer import atomic_write_bytes  # noqa: E402
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402
from tools import verify_resident_v3_training_admission as admission  # noqa: E402
from tools.launch_resident_recurrence_training import (  # noqa: E402
    _stop_unguarded_target,
    _validate_no_competing_model_process,
    _wait_for_running_target,
)

CALIBRATION_SCHEMA = "aura.resident_v3_recovery_calibration.v1"
_MAX_JSON_BYTES = 256 * 1024 * 1024
_STARTUP_LETHAL_MB = 73_728.0
_STEADY_LETHAL_MB = 59_392.0


class ResidentV3RecoveryError(RuntimeError):
    """Stable fail-closed recovery launch or verification error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise ResidentV3RecoveryError(code)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha(read_stable_bytes(path.resolve(strict=True), max_bytes=_MAX_JSON_BYTES))


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        value = json.loads(
            read_stable_bytes(path.expanduser().resolve(strict=True), max_bytes=_MAX_JSON_BYTES)
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResidentV3RecoveryError(f"{role}_invalid") from exc
    if not isinstance(value, dict):
        _fail(f"{role}_invalid")
    return value


def _migration(migration_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    migration = _read_json(migration_path, role="migration")
    destination = migration.get("destination")
    trainer = migration.get("new_trainer")
    if not isinstance(destination, Mapping) or not isinstance(trainer, Mapping):
        _fail("migration_invalid")
    summary = verify_migration(
        migration_path,
        expected_destination_root=Path(str(destination.get("root"))),
        expected_trainer_sha256=str(trainer.get("sha256")),
    )
    if trainer.get("sha256") != _sha_file(REPO_ROOT / "tools/recurrence_native_train_v2.py"):
        _fail("trainer_source_changed")
    return migration, summary


def _positive_int(value: Any, *, role: str) -> int:
    if type(value) is not int or value < 1:
        _fail(f"{role}_invalid")
    return value


def _finite(value: Any, *, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{role}_invalid")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{role}_invalid")
    return result


def _csv(value: Any, *, role: str) -> str:
    if not isinstance(value, list) or not value or any(
        isinstance(item, bool) or not isinstance(item, (str, int)) for item in value
    ):
        _fail(f"{role}_invalid")
    return ",".join(str(item) for item in value)


def _phase_paths(destination_root: Path, phase: str) -> dict[str, Path]:
    if phase not in {"calibration", "resume"}:
        _fail("phase_invalid")
    root = destination_root.parent
    label = root.name
    if not label.startswith("resident_32b_v3_cp") or not label.removeprefix(
        "resident_32b_v3_cp"
    ).isdigit():
        _fail("destination_checkpoint_label_invalid")
    return {
        "root": root,
        "run": root / f"detached-{phase}",
        "sentinel": root / f"sentinel-{phase}",
        "ring": destination_root / f"physical_footprint_{phase}_{label}.jsonl",
        "stage": destination_root / f"resource_stage_{phase}_{label}.json",
        "envelope": destination_root / f"launch_resource_envelope_{phase}_{label}.json",
    }


def build_commands(
    migration_path: Path,
    *,
    phase: str,
) -> tuple[list[str], list[str], dict[str, Path]]:
    migration, summary = _migration(migration_path)
    source = migration["source"]
    config = source.get("training_config_document")
    dataset = source.get("dataset_manifest_document")
    spec = source.get("execution_spec_document")
    if not all(isinstance(value, Mapping) for value in (config, dataset, spec)):
        _fail("migration_scientific_inputs_invalid")
    destination_root = Path(str(migration["destination"]["root"])).resolve(strict=True)
    paths = _phase_paths(destination_root, phase)
    checkpoint_number = destination_root.parent.name.removeprefix("resident_32b_v3_cp")
    latest = _read_json(destination_root / "latest.json", role="latest")
    if phase == "calibration":
        if latest.get("checkpoint") != summary["source_checkpoint"]:
            _fail("calibration_source_checkpoint_changed")
        max_minutes = 0.001
        timeout_s = 7_200
    else:
        calibration = verify_calibration(migration_path)
        if calibration.get("passed") is not True:
            _fail("calibration_not_admitted")
        max_minutes = 2_880.0
        timeout_s = 216_000
    objective_options = config.get("objective_options")
    bridge = config.get("bridge")
    holdout = config.get("holdout")
    lora = config.get("lora")
    optimizer = config.get("optimizer")
    if not all(
        isinstance(value, Mapping)
        for value in (objective_options, bridge, holdout, lora, optimizer)
    ):
        _fail("migration_training_config_invalid")
    branch_roles = spec.get("branch_roles")
    model = Path(str(config.get("model_path"))).resolve(strict=True)
    _validate_no_competing_model_process(model)
    python = REPO_ROOT / ".venv/bin/python"
    target = [
        str(python),
        str(REPO_ROOT / "tools/run_recurrence_training_envelope.py"),
        "--memory-limit-gb",
        "40",
        "--cache-limit-gb",
        "2",
        "--wired-limit-gb",
        "48",
        "--envelope-out",
        str(paths["envelope"]),
        "--trainer",
        str(REPO_ROOT / "tools/recurrence_native_train_v2.py"),
        "--",
        "--model",
        str(model),
        "--out-dir",
        str(destination_root),
        "--adapter-id",
        f"resident-32b-recurrence-v3-cp{checkpoint_number}",
        "--personality-adapter",
        str(config.get("personality_adapter_path") or "none"),
        "--train-seed",
        str(_positive_int(config.get("train_seed"), role="train_seed")),
        "--families",
        _csv(dataset.get("families"), role="families"),
        "--task-depths",
        _csv(dataset.get("task_depths"), role="task_depths"),
        "--per-cell",
        str(_positive_int(dataset.get("per_cell"), role="per_cell")),
        "--curriculum-depths",
        _csv(config.get("curriculum_depths"), role="curriculum_depths"),
        "--n-slots",
        str(_positive_int(spec.get("n_slots"), role="n_slots")),
        "--branch-roles",
        _csv(branch_roles, role="branch_roles"),
        "--exchange-interval",
        str(_positive_int(spec.get("exchange_interval"), role="exchange_interval")),
        "--alpha",
        str(_finite(spec.get("alpha"), role="alpha")),
        "--alpha-schedule",
        str(spec.get("alpha_schedule")),
        "--lora-rank",
        str(_positive_int(lora.get("rank"), role="lora_rank")),
        "--lora-targets",
        _csv(lora.get("targets"), role="lora_targets"),
        "--learning-rate",
        str(_finite(optimizer.get("learning_rate"), role="learning_rate")),
        "--monotonicity-weight",
        str(_finite(config.get("monotonicity_weight"), role="monotonicity_weight")),
        "--objective",
        "v3",
        "--depth-margin",
        str(_finite(objective_options.get("depth_margin"), role="depth_margin")),
        "--diversity-weight",
        str(_finite(objective_options.get("diversity_weight"), role="diversity_weight")),
        "--diversity-target-cos",
        str(_finite(objective_options.get("diversity_target_cos"), role="diversity_cos")),
        "--bridge-policy",
        str(bridge.get("policy")),
        "--holdout-per-cell",
        str(_positive_int(holdout.get("per_cell"), role="holdout_per_cell")),
        "--holdout-eval-samples",
        str(_positive_int(holdout.get("eval_samples"), role="holdout_eval_samples")),
        "--max-minutes",
        str(max_minutes),
        "--max-steps",
        str(_positive_int(config.get("max_steps"), role="max_steps")),
        "--checkpoint-every",
        "5",
        "--log-every",
        "5",
        "--resource-stage-path",
        str(paths["stage"]),
        "--resource-startup-lethal-mb",
        str(_STARTUP_LETHAL_MB),
        "--resource-steady-lethal-mb",
        str(_STEADY_LETHAL_MB),
        "--activation-checkpointing",
        "--resume",
        "--resume-migration-evidence",
        str(migration_path.resolve(strict=True)),
    ]
    launcher = [
        str(python),
        str(REPO_ROOT / "tools/run_detached_step.py"),
        "launch",
        "--run-dir",
        str(paths["run"]),
        "--name",
        f"cp{checkpoint_number}-resident-32b-v3-{phase}",
        "--cwd",
        str(REPO_ROOT),
        "--timeout",
        str(timeout_s),
        "--",
        *target,
    ]
    return launcher, target, paths


def _sentinel_command(paths: Mapping[str, Path], *, trainer_pid: int, timeout_s: int) -> list[str]:
    python = REPO_ROOT / ".venv/bin/python"
    return [
        str(python),
        str(REPO_ROOT / "tools/run_detached_step.py"),
        "launch",
        "--run-dir",
        str(paths["sentinel"]),
        "--name",
        f"{paths['run'].name}-memory-sentinel",
        "--cwd",
        str(REPO_ROOT),
        "--timeout",
        str(timeout_s),
        "--",
        str(python),
        str(REPO_ROOT / "tools/memory_sentinel.py"),
        "--pid",
        str(trainer_pid),
        "--lethal-mb",
        str(_STEADY_LETHAL_MB),
        "--startup-lethal-mb",
        str(_STARTUP_LETHAL_MB),
        "--steady-marker",
        str(paths["stage"]),
        "--interval",
        "0.5",
        "--immediate-kill-overshoot",
        "1.05",
        "--ring",
        str(paths["ring"]),
        "--ring-window-seconds",
        "46800.0",
        "--tombstone-dir",
        str(paths["sentinel"]),
    ]


def launch_phase(migration_path: Path, *, phase: str) -> dict[str, Any]:
    launcher, target, paths = build_commands(migration_path, phase=phase)
    for role in ("run", "sentinel"):
        if paths[role].exists() or paths[role].is_symlink():
            _fail(f"{phase}_{role}_already_exists")
    completed = subprocess.run(launcher, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        _fail("detached_launcher_failed")
    try:
        status = _wait_for_running_target(paths["run"], timeout_s=30.0)
    except BaseException:
        _stop_unguarded_target(paths["run"])
        raise
    trainer_pid = status.get("child_pid")
    if type(trainer_pid) is not int or trainer_pid < 1:
        _stop_unguarded_target(paths["run"])
        _fail("trainer_pid_invalid")
    timeout_s = 7_200 if phase == "calibration" else 216_000
    sentinel = subprocess.run(
        _sentinel_command(paths, trainer_pid=trainer_pid, timeout_s=timeout_s),
        cwd=REPO_ROOT,
        check=False,
    )
    if sentinel.returncode != 0:
        _stop_unguarded_target(paths["run"])
        _fail("sentinel_launcher_failed")
    return {
        "phase": phase,
        "trainer_pid": trainer_pid,
        "run_dir": str(paths["run"]),
        "sentinel_run_dir": str(paths["sentinel"]),
        "target_command_sha256": _sha(_canonical(target)),
    }


def _terminal_receipt(path: Path, *, role: str) -> dict[str, Any]:
    receipt = _read_json(path / "detached_receipt.json", role=role)
    if (
        receipt.get("containment_verified") is not True
        or receipt.get("process_group_empty") is not True
        or receipt.get("lineage_empty") is not True
        or receipt.get("restart_count") != 0
        or receipt.get("timed_out") is not False
    ):
        _fail(f"{role}_containment_invalid")
    return receipt


def verify_calibration(migration_path: Path) -> dict[str, Any]:
    migration, summary = _migration(migration_path)
    destination = Path(str(migration["destination"]["root"])).resolve(strict=True)
    paths = _phase_paths(destination, "calibration")
    output = paths["root"] / "calibration_verdict.json"
    if output.exists():
        verdict = _read_json(output, role="calibration_verdict")
        claimed = verdict.get("verdict_sha256")
        material = dict(verdict)
        material.pop("verdict_sha256", None)
        if verdict.get("schema") != CALIBRATION_SCHEMA or claimed != _sha(_canonical(material)):
            _fail("calibration_verdict_invalid")
        return verdict
    trainer_receipt = _terminal_receipt(paths["run"], role="calibration_trainer")
    sentinel_receipt = _terminal_receipt(paths["sentinel"], role="calibration_sentinel")
    tombstones = sorted(paths["sentinel"].glob("sentinel_tombstone_*.json"))
    training_receipt = _read_json(destination / "receipt.json", role="training_receipt")
    latest = _read_json(destination / "latest.json", role="latest")
    checkpoint = destination / str(latest.get("checkpoint", ""))
    complete = _read_json(checkpoint / "complete.json", role="checkpoint_complete")
    try:
        resource = admission._verify_footprint(  # noqa: SLF001 - shared strict proof primitive
            paths["ring"],
            paths["sentinel"],
            trainer_pid=int(trainer_receipt.get("child_pid", 0)),
            stage_path=paths["stage"],
            expected_trainer_sha256=summary["new_trainer_sha256"],
        )
    except admission.ResidentV3TrainingAdmissionError as exc:
        raise ResidentV3RecoveryError(f"calibration_resource_evidence_invalid:{exc.code}") from exc
    max_compute_mb = float(resource["stage_peak_managed_mb"]["compute"])
    expected_step = int(summary["source_step"]) + 1
    failure_points: list[str] = []
    if trainer_receipt.get("returncode") != 75:
        failure_points.append("calibration_trainer_did_not_exit_bounded_partial")
    if sentinel_receipt.get("returncode") != 0:
        failure_points.append("calibration_sentinel_failed")
    if tombstones:
        failure_points.append("calibration_sentinel_tombstone_present")
    if (
        training_receipt.get("complete") is not False
        or training_receipt.get("halt_reason") != "wall_clock"
        or training_receipt.get("steps") != expected_step
        or training_receipt.get("gradient_execution", {}).get("activation_rematerialization")
        != "transformer_layer_group_checkpoint"
        or training_receipt.get("gradient_execution", {}).get("layer_group_size") != 4
        or training_receipt.get("resume_migration") != summary
    ):
        failure_points.append("calibration_training_receipt_invalid")
    if complete.get("step") != expected_step or complete.get("config_sha256") != training_receipt.get(
        "config_sha256"
    ):
        failure_points.append("calibration_checkpoint_did_not_advance_exactly_once")
    if max_compute_mb >= _STARTUP_LETHAL_MB:
        failure_points.append("calibration_compute_memory_not_below_lethal_ceiling")
    material = {
        "schema": CALIBRATION_SCHEMA,
        "passed": not failure_points,
        "failure_points": failure_points,
        "source_step": summary["source_step"],
        "calibration_step": complete.get("step"),
        "max_compute_managed_mb": max_compute_mb,
        "compute_lethal_mb": _STARTUP_LETHAL_MB,
        "memory_reduction_vs_cp189_terminal_mb": round(74_846.3 - max_compute_mb, 3),
        "resource_evidence": resource,
        "trainer_receipt_sha256": trainer_receipt.get("receipt_sha256"),
        "sentinel_receipt_sha256": sentinel_receipt.get("receipt_sha256"),
        "checkpoint_complete_sha256": _sha(
            read_stable_bytes(checkpoint / "complete.json", max_bytes=_MAX_JSON_BYTES)
        ),
        "migration_sha256": summary["migration_sha256"],
        "verified_at": time.time(),
    }
    verdict = {**material, "verdict_sha256": _sha(_canonical(material))}
    atomic_write_bytes(output, _canonical(verdict) + b"\n", mode=0o600)
    if failure_points:
        _fail("calibration_failed")
    return verdict


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--migration", type=Path, required=True)
    parser.add_argument("--phase", choices=("calibration", "resume"))
    parser.add_argument("--verify-calibration", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_calibration:
        if args.phase is not None:
            _fail("cli_mode_invalid")
        print(json.dumps(verify_calibration(args.migration), sort_keys=True))
        return 0
    if args.phase is None:
        _fail("phase_required")
    if args.dry_run:
        launcher, target, paths = build_commands(args.migration, phase=args.phase)
        print(
            json.dumps(
                {
                    "launcher": launcher,
                    "target": target,
                    "paths": {key: str(value) for key, value in paths.items()},
                },
                sort_keys=True,
            )
        )
        return 0
    print(json.dumps(launch_phase(args.migration, phase=args.phase), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
