#!/usr/bin/env python3
"""Strict admission for migration-recovered resident-v3 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Never

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (  # noqa: E402
    canonical_json_bytes,
    strict_json_loads,
)
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402
from tools import launch_resident_v3_recovery as recovery  # noqa: E402
from tools import verify_resident_v3_training_admission as admission  # noqa: E402

SCHEMA = "aura.resident_v3_recovery_training_admission.v1"
CALIBRATION_SCHEMA = "aura.resident_v3_recovery_calibration.v1"
CONTROLLER_SCHEMA = "aura.resident_v3_recovery_controller_verdict.v1"
ARCHIVE_RECEIPT_SCHEMA = "aura.memory_sentinel_ring_archive_receipt.v1"
_MAX_JSON_BYTES = 512 * 1024 * 1024


class ResidentV3RecoveryAdmissionError(RuntimeError):
    """Stable fail-closed recovery admission error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise ResidentV3RecoveryAdmissionError(code)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path, *, role: str) -> tuple[bytes, dict[str, Any]]:
    supplied = path.expanduser()
    if not supplied.is_absolute() or supplied.is_symlink():
        _fail(f"{role}_path_invalid")
    resolved = supplied.resolve(strict=True)
    if resolved != supplied or not resolved.is_file():
        _fail(f"{role}_path_invalid")
    raw = read_stable_bytes(resolved, max_bytes=_MAX_JSON_BYTES)
    try:
        value = strict_json_loads(raw, role=role)
    except ValueError as exc:
        raise ResidentV3RecoveryAdmissionError(f"{role}_invalid") from exc
    if not isinstance(value, dict):
        _fail(f"{role}_invalid")
    return raw, value


def _absolute_file(path: Path, *, role: str) -> Path:
    supplied = path.expanduser()
    if not supplied.is_absolute() or supplied.is_symlink():
        _fail(f"{role}_path_invalid")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise ResidentV3RecoveryAdmissionError(f"{role}_path_invalid") from exc
    if resolved != supplied or not resolved.is_file():
        _fail(f"{role}_path_invalid")
    return resolved


def _verify_hashed_document(
    path: Path,
    *,
    role: str,
    schema: str,
    hash_key: str,
) -> tuple[bytes, dict[str, Any]]:
    raw, value = _read_json(path, role=role)
    material = dict(value)
    claimed = material.pop(hash_key, None)
    if value.get("schema") != schema or claimed != _sha(canonical_json_bytes(material)):
        _fail(f"{role}_invalid")
    return raw, value


def _protocol(migration: Mapping[str, Any], adapter_root: Path) -> dict[str, Any]:
    source = migration.get("source")
    if not isinstance(source, Mapping):
        _fail("migration_source_invalid")
    config = source.get("training_config_document")
    spec = source.get("execution_spec_document")
    if not isinstance(config, Mapping) or not isinstance(spec, Mapping):
        _fail("migration_source_invalid")
    base = config.get("base_checkpoint")
    options = config.get("objective_options")
    holdout = config.get("holdout")
    lora = config.get("lora")
    optimizer = config.get("optimizer")
    if not all(
        isinstance(value, Mapping)
        for value in (base, options, holdout, lora, optimizer)
    ):
        _fail("migration_source_invalid")
    return {
        "model": {
            "path": config.get("model_path"),
            "expected_full_weight_sha256": base.get("fingerprint"),
        },
        "training": {
            "output_dir": str(adapter_root),
            "adapter_id": (
                "resident-32b-recurrence-v3-"
                + adapter_root.parent.name.removeprefix("resident_32b_v3_")
            ),
            "train_seed": config.get("train_seed"),
            "max_steps": config.get("max_steps"),
            "curriculum_depths": config.get("curriculum_depths"),
            "monotonicity_weight": config.get("monotonicity_weight"),
            "depth_margin": options.get("depth_margin"),
            "diversity_weight": options.get("diversity_weight"),
            "diversity_target_cos": options.get("diversity_target_cos"),
            "holdout_per_cell": holdout.get("per_cell"),
            "holdout_count": holdout.get("count"),
            "holdout_eval_samples": holdout.get("eval_samples"),
            "lora_rank": lora.get("rank"),
            "lora_targets": lora.get("targets"),
            "learning_rate": optimizer.get("learning_rate"),
            "n_slots": spec.get("n_slots"),
            "branch_roles": spec.get("branch_roles"),
            "exchange_interval": spec.get("exchange_interval"),
            "alpha": spec.get("alpha"),
            "alpha_schedule": spec.get("alpha_schedule"),
        },
    }


def _command_option(command: Sequence[Any], option: str) -> str:
    values = [str(value) for value in command]
    positions = [index for index, value in enumerate(values) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(values):
        _fail("resume_command_invalid")
    return values[positions[0] + 1]


def _verify_resume_command(
    plan: Mapping[str, Any],
    *,
    migration_path: Path,
    adapter_root: Path,
    model_path: Path,
    stage_path: Path,
) -> dict[str, Any]:
    command = plan.get("command")
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        _fail("resume_command_invalid")
    try:
        separator = command.index("--")
    except ValueError:
        _fail("resume_command_invalid")
    if (
        command.count("--resume") != 1
        or command[1] != str(ROOT / "tools/run_recurrence_training_envelope.py")
        or _command_option(command[:separator], "--trainer")
        != str(ROOT / "tools/recurrence_native_train_v2.py")
        or _command_option(command[:separator], "--memory-limit-gb") != "40"
        or _command_option(command[:separator], "--cache-limit-gb") != "2"
        or _command_option(command[:separator], "--wired-limit-gb") != "48"
        or _command_option(command[separator + 1 :], "--model") != str(model_path)
        or _command_option(command[separator + 1 :], "--out-dir") != str(adapter_root)
        or _command_option(command[separator + 1 :], "--max-minutes") != "2880.0"
        or _command_option(command[separator + 1 :], "--resource-stage-path")
        != str(stage_path)
        or _command_option(command[separator + 1 :], "--resume-migration-evidence")
        != str(migration_path)
    ):
        _fail("resume_command_invalid")
    return {
        "plan_sha256": plan.get("plan_sha256"),
        "command_sha256": plan.get("command_sha256"),
    }


def _verify_archive(
    *,
    archive_run_dir: Path,
    archive_ring: Path,
    archive_receipt_path: Path,
    operational_ring: Path,
    trainer_pid: int,
) -> dict[str, Any]:
    plan, detached_receipt = admission._detached_terminal(  # noqa: SLF001
        archive_run_dir,
        role="sentinel_archive",
    )
    if (
        detached_receipt.get("returncode") != 0
        or detached_receipt.get("status") != "passed"
    ):
        _fail("sentinel_archive_failed")
    command = plan.get("command")
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        _fail("sentinel_archive_command_invalid")
    options = admission._option_map(command[2:], role="sentinel_archive")  # noqa: SLF001
    try:
        interval_s = float(options.get("--interval", "nan"))
        command_pid = int(options.get("--target-pid", "-1"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResidentV3RecoveryAdmissionError("sentinel_archive_command_invalid") from exc
    expected_state = archive_receipt_path.parent / "archive_state.json"
    if (
        command[1] != str(ROOT / "tools/archive_memory_sentinel_ring.py")
        or options.get("--source") != str(operational_ring)
        or options.get("--archive") != str(archive_ring)
        or options.get("--state") != str(expected_state)
        or options.get("--receipt") != str(archive_receipt_path)
        or command_pid != trainer_pid
        or interval_s != 5.0
    ):
        _fail("sentinel_archive_command_invalid")
    _raw, receipt = _verify_hashed_document(
        archive_receipt_path,
        role="sentinel_archive_receipt",
        schema=ARCHIVE_RECEIPT_SCHEMA,
        hash_key="receipt_sha256",
    )
    archive_path = _absolute_file(archive_ring, role="sentinel_archive")
    archive_raw = read_stable_bytes(archive_path, max_bytes=_MAX_JSON_BYTES)
    if (
        receipt.get("status") != "passed"
        or receipt.get("source") != str(operational_ring)
        or receipt.get("archive") != str(archive_ring)
        or receipt.get("archive_sha256") != _sha(archive_raw)
        or receipt.get("target_pid") != trainer_pid
        or type(receipt.get("sample_count")) is not int
        or receipt["sample_count"] < 1
        or receipt["sample_count"] != sum(1 for line in archive_raw.splitlines() if line)
    ):
        _fail("sentinel_archive_receipt_invalid")
    return {
        "archive_sha256": receipt["archive_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "sample_count": receipt["sample_count"],
        "plan_sha256": plan.get("plan_sha256"),
        "detached_receipt_sha256": detached_receipt.get("receipt_sha256"),
    }


def _verify_controller_terminal_binding(
    controller_details: Mapping[str, Any],
    *,
    calibration_verdict_sha256: str,
    trainer_receipt_sha256: str,
    sentinel_receipt_sha256: str,
) -> None:
    controller_receipts = controller_details.get("training_receipts")
    controller_trainer = (
        controller_receipts.get("trainer")
        if isinstance(controller_receipts, Mapping)
        else None
    )
    controller_sentinel = (
        controller_receipts.get("sentinel")
        if isinstance(controller_receipts, Mapping)
        else None
    )
    if (
        controller_details.get("calibration_verdict_sha256")
        != calibration_verdict_sha256
        or not isinstance(controller_trainer, Mapping)
        or not isinstance(controller_sentinel, Mapping)
        or controller_trainer.get("receipt_sha256") != trainer_receipt_sha256
        or controller_sentinel.get("receipt_sha256") != sentinel_receipt_sha256
    ):
        _fail("controller_terminal_binding_mismatch")


def verify(args: argparse.Namespace) -> dict[str, Any]:
    migration_raw, migration = _read_json(args.migration, role="migration")
    migration_path = args.migration.expanduser()
    destination = migration.get("destination")
    if not isinstance(destination, Mapping):
        _fail("migration_destination_invalid")
    adapter_root = Path(str(destination.get("root"))).resolve(strict=True)
    root = adapter_root.parent
    if not root.name.startswith("resident_32b_v3_cp"):
        _fail("migration_destination_invalid")
    try:
        _verified_migration, migration_summary = recovery._migration(  # noqa: SLF001
            migration_path,
            allow_destination_pointer_advance=True,
        )
    except Exception as exc:
        raise ResidentV3RecoveryAdmissionError("migration_verification_failed") from exc

    _calibration_raw, calibration = _verify_hashed_document(
        args.calibration_verdict,
        role="calibration_verdict",
        schema=CALIBRATION_SCHEMA,
        hash_key="verdict_sha256",
    )
    _controller_raw, controller = _verify_hashed_document(
        args.controller_verdict,
        role="controller_verdict",
        schema=CONTROLLER_SCHEMA,
        hash_key="verdict_sha256",
    )
    controller_details = controller.get("details")
    if (
        calibration.get("passed") is not True
        or calibration.get("failure_points") != []
        or calibration.get("migration_sha256") != migration_summary["migration_sha256"]
        or controller.get("decision") != "training_terminal_pending_strict_admission"
        or controller.get("migration_sha256") != migration_summary["migration_sha256"]
        or not isinstance(controller_details, Mapping)
    ):
        _fail("recovery_chain_not_admissible")

    resume_plan, resume_receipt = admission._detached_terminal(  # noqa: SLF001
        args.resume_run_dir,
        role="recovery_resume",
    )
    if resume_receipt.get("returncode") != 0:
        _fail("recovery_resume_failed")
    config_document = migration["source"]["training_config_document"]
    model_path = Path(str(config_document["model_path"])).resolve(strict=True)
    stage_path = adapter_root / f"resource_stage_resume_{root.name}.json"
    command = _verify_resume_command(
        resume_plan,
        migration_path=migration_path,
        adapter_root=adapter_root,
        model_path=model_path,
        stage_path=stage_path,
    )
    protocol = _protocol(migration, adapter_root)
    identity, training_receipt, training_config = admission._validate_adapter_identity(  # noqa: SLF001
        adapter_root,
        protocol,
        allow_bounded_partial=False,
    )
    admission._verify_protocol_match(protocol, training_receipt, training_config)  # noqa: SLF001
    state = admission.evaluate_training_state(training_receipt, training_config)
    if state.get("scope") != "complete_training" or identity.get("complete") is not True:
        _fail("training_not_complete")
    if (
        training_receipt.get("resume_migration") != migration_summary
        or training_receipt.get("gradient_execution", {}).get("adjoint_schema")
        != "aura.recurrence_exact_discrete_adjoint.v1"
    ):
        _fail("training_migration_identity_mismatch")
    terminal = admission._verify_terminal_checkpoint_state(  # noqa: SLF001
        adapter_root,
        training_receipt,
        log_every=5,
    )
    _sentinel_plan, sentinel_receipt = admission._detached_terminal(  # noqa: SLF001
        args.sentinel_run_dir,
        role="memory_sentinel",
    )
    _verify_controller_terminal_binding(
        controller_details,
        calibration_verdict_sha256=str(calibration["verdict_sha256"]),
        trainer_receipt_sha256=str(resume_receipt.get("receipt_sha256")),
        sentinel_receipt_sha256=str(sentinel_receipt.get("receipt_sha256")),
    )

    operational_ring = _absolute_file(args.operational_ring, role="operational_ring")
    archive_ring = _absolute_file(args.archive_ring, role="sentinel_archive")

    archive_evidence = _verify_archive(
        archive_run_dir=args.archive_run_dir,
        archive_ring=archive_ring,
        archive_receipt_path=args.archive_receipt,
        operational_ring=operational_ring,
        trainer_pid=int(resume_receipt["child_pid"]),
    )
    footprint = admission._verify_footprint(  # noqa: SLF001
        archive_ring,
        args.sentinel_run_dir,
        trainer_pid=int(resume_receipt["child_pid"]),
        stage_path=stage_path,
        expected_trainer_sha256=str(migration_summary["new_trainer_sha256"]),
        command_ring_path=operational_ring,
    )
    if footprint["sample_count"] != archive_evidence["sample_count"]:
        _fail("sentinel_archive_sample_count_mismatch")
    elapsed = training_receipt.get("elapsed_training_s")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(
        float(elapsed)
    ):
        _fail("training_elapsed_invalid")
    payload = {
        "schema": SCHEMA,
        "decision": "admit_to_freeze_and_mechanics",
        "claim_scope": "resident_v3_recovery_training_mechanics_admission_only",
        "migration": {
            "sha256": _sha(migration_raw),
            "migration_sha256": migration_summary["migration_sha256"],
        },
        "calibration_verdict_sha256": calibration["verdict_sha256"],
        "controller_verdict_sha256": controller["verdict_sha256"],
        "resume": {
            **command,
            "receipt_sha256": resume_receipt.get("receipt_sha256"),
            "returncode": resume_receipt.get("returncode"),
        },
        "archive": archive_evidence,
        "footprint": footprint,
        "training_state": state,
        "terminal_checkpoint": terminal,
        "identity_receipt": identity,
        "claim_flags": {
            "training_admitted": True,
            "adapter_freeze_eligible": True,
            "mechanics_proven": False,
            "reasoning_gain": False,
            "same_checkpoint_interaction": False,
            "frontier_level": False,
            "frontier_plus": False,
            "installed_desktop_gain": False,
        },
    }
    try:
        return admission._write_once(args.output, payload)  # noqa: SLF001
    except admission.ResidentV3TrainingAdmissionError as exc:
        raise ResidentV3RecoveryAdmissionError(exc.code) from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--migration", type=Path, required=True)
    parser.add_argument("--calibration-verdict", type=Path, required=True)
    parser.add_argument("--controller-verdict", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path, required=True)
    parser.add_argument("--sentinel-run-dir", type=Path, required=True)
    parser.add_argument("--operational-ring", type=Path, required=True)
    parser.add_argument("--archive-run-dir", type=Path, required=True)
    parser.add_argument("--archive-ring", type=Path, required=True)
    parser.add_argument("--archive-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify(args), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            f"verify_resident_v3_recovery_training_admission: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
