#!/usr/bin/env python3
"""Launch a resident recurrence phase only from a frozen resource amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402

PROTOCOL_SCHEMA = "aura.recurrence_native_resident_protocol.v2"
AMENDMENT_SCHEMA = "aura.recurrence_native_resident_protocol_amendment.v1"
FAILURE_SCHEMA = "aura.recurrence_resident_resource_failure.v1"
_MAX_JSON_BYTES = 8 * 1024 * 1024


class ResidentRecurrenceLaunchError(RuntimeError):
    """Stable fail-closed launch error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise ResidentRecurrenceLaunchError(code)


def _read_json(path: Path, *, role: str) -> tuple[bytes, dict[str, Any]]:
    supplied = path.expanduser()
    if not supplied.is_absolute() or supplied.is_symlink():
        _fail(f"{role}_path_invalid")
    resolved = supplied.resolve(strict=True)
    if resolved != supplied or not resolved.is_file():
        _fail(f"{role}_path_invalid")
    raw = read_stable_bytes(resolved, max_bytes=_MAX_JSON_BYTES)
    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ResidentRecurrenceLaunchError(f"{role}_invalid") from exc
    if not isinstance(parsed, dict):
        _fail(f"{role}_invalid")
    return raw, parsed


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Never:
    raise ValueError(f"non-finite JSON value: {value}")


def _binding_matches(raw: bytes, binding: Mapping[str, Any]) -> bool:
    return (
        binding.get("sha256") == hashlib.sha256(raw).hexdigest()
        and binding.get("size_bytes") == len(raw)
    )


def _source_path(value: Any, *, role: str) -> Path:
    if not isinstance(value, str) or not value:
        _fail(f"{role}_path_invalid")
    supplied = REPO_ROOT / value
    if supplied.is_symlink():
        _fail(f"{role}_path_invalid")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise ResidentRecurrenceLaunchError(f"{role}_path_invalid") from exc
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        _fail(f"{role}_path_invalid")
    if not resolved.is_file():
        _fail(f"{role}_path_invalid")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _absolute_path(value: Any, *, role: str, kind: str | None = None) -> Path:
    if not isinstance(value, str) or not value:
        _fail(f"{role}_path_invalid")
    supplied = Path(value).expanduser()
    if not supplied.is_absolute() or supplied.is_symlink():
        _fail(f"{role}_path_invalid")
    resolved = supplied.resolve(strict=kind is not None)
    if resolved != supplied:
        _fail(f"{role}_path_invalid")
    if kind == "file" and not resolved.is_file():
        _fail(f"{role}_path_invalid")
    if kind == "directory" and not resolved.is_dir():
        _fail(f"{role}_path_invalid")
    return resolved


def _positive_int(value: Any, *, role: str) -> int:
    if type(value) is not int or value < 1:
        _fail(f"{role}_invalid")
    return value


def _finite_number(value: Any, *, role: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        _fail(f"{role}_invalid")
    return float(value)


def _csv(values: Any, *, role: str) -> str:
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, (str, int)) or isinstance(value, bool) for value in values)
    ):
        _fail(f"{role}_invalid")
    return ",".join(str(value) for value in values)


def _validate_no_competing_model_process(model: Path) -> None:
    try:
        import psutil
    except ImportError as exc:
        raise ResidentRecurrenceLaunchError("psutil_required") from exc

    excluded: set[int] = {os.getpid()}
    current = psutil.Process(os.getpid())
    while True:
        try:
            current = current.parent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        if current is None:
            break
        excluded.add(current.pid)
    model_text = str(model)
    conflicts: list[int] = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = int(process.info["pid"])
            command = process.info.get("cmdline") or []
        except (TypeError, ValueError, psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if pid in excluded or not command:
            continue
        tokens = [str(token) for token in command]
        if (
            model_text in tokens
            or any(token.endswith("recurrence_native_train_v2.py") for token in tokens)
            or any(token.endswith("run_recurrence_training_envelope.py") for token in tokens)
            or any(token.endswith("aura_main.py") for token in tokens)
        ):
            conflicts.append(pid)
    if conflicts:
        _fail("competing_model_process_detected")


def _validate_contract(
    protocol_path: Path,
    amendment_path: Path,
    *,
    phase: str = "resume",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if phase not in {"partial", "resume"}:
        _fail("launch_phase_invalid")
    protocol_raw, protocol = _read_json(protocol_path, role="protocol")
    _amendment_raw, amendment = _read_json(amendment_path, role="amendment")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        _fail("protocol_schema_invalid")
    if amendment.get("schema") != AMENDMENT_SCHEMA:
        _fail("amendment_schema_invalid")
    parent = amendment.get("parent_protocol")
    if not isinstance(parent, Mapping) or not _binding_matches(protocol_raw, parent):
        _fail("parent_protocol_binding_mismatch")
    if parent.get("path") != protocol_path.name:
        _fail("parent_protocol_path_mismatch")

    failure_ref = amendment.get("triggering_failure")
    if not isinstance(failure_ref, Mapping):
        _fail("triggering_failure_invalid")
    failure_path = (amendment_path.parent / str(failure_ref.get("path", ""))).absolute()
    _failure_raw, failure = _read_json(
        failure_path, role="triggering_failure"
    )
    kernel = failure.get("kernel_evidence")
    process = failure.get("process")
    detached_binding = failure.get("detached_receipt")
    if (
        failure.get("schema") != FAILURE_SCHEMA
        or not isinstance(kernel, Mapping)
        or not isinstance(process, Mapping)
        or not isinstance(detached_binding, Mapping)
        or kernel.get("termination_classification")
        != failure_ref.get("required_classification")
        or kernel.get("compressed_process_mb")
        != failure_ref.get("required_compressed_process_mb")
        or process.get("detached_returncode") != -9
        or process.get("timed_out") is not False
        or process.get("restart_count") != 0
        or process.get("containment_verified") is not True
        or process.get("process_group_empty") is not True
        or process.get("lineage_empty") is not True
    ):
        _fail("triggering_failure_mismatch")
    receipt_candidate = failure_path.parent / str(detached_binding.get("path", ""))
    if receipt_candidate.is_symlink():
        _fail("triggering_failure_receipt_path_invalid")
    try:
        receipt_path = receipt_candidate.resolve(strict=True)
        receipt_path.relative_to(amendment_path.parent.parent)
    except (OSError, ValueError) as exc:
        raise ResidentRecurrenceLaunchError(
            "triggering_failure_receipt_path_invalid"
        ) from exc
    receipt_raw, failed_receipt = _read_json(
        receipt_path,
        role="triggering_failure_receipt",
    )
    if (
        not _binding_matches(receipt_raw, detached_binding)
        or failed_receipt.get("returncode") != process.get("detached_returncode")
        or failed_receipt.get("timed_out") != process.get("timed_out")
        or failed_receipt.get("restart_count") != process.get("restart_count")
        or failed_receipt.get("containment_verified")
        != process.get("containment_verified")
        or failed_receipt.get("process_group_empty")
        != process.get("process_group_empty")
        or failed_receipt.get("lineage_empty") != process.get("lineage_empty")
    ):
        _fail("triggering_failure_receipt_mismatch")

    envelope = amendment.get("resource_envelope")
    if not isinstance(envelope, Mapping):
        _fail("resource_envelope_invalid")
    wrapper = _source_path(envelope.get("wrapper"), role="resource_wrapper")
    trainer = _source_path(envelope.get("trainer"), role="trainer")
    limits = (
        _finite_number(envelope.get("memory_limit_gb"), role="memory_limit"),
        _finite_number(envelope.get("cache_limit_gb"), role="cache_limit"),
        _finite_number(envelope.get("wired_limit_gb"), role="wired_limit"),
    )
    if limits != (40.0, 2.0, 48.0):
        _fail("resource_envelope_limits_not_preregistered")
    if (
        envelope.get("cache_cleared_before_model_load") is not True
        or envelope.get("wrapper_sha256") != _sha256_file(wrapper)
        or envelope.get("trainer_sha256") != _sha256_file(trainer)
    ):
        _fail("resource_envelope_source_mismatch")

    training = protocol.get("training")
    phase_contract = amendment.get(phase)
    if not isinstance(training, Mapping) or not isinstance(phase_contract, Mapping):
        _fail("training_contract_invalid")
    adapter = _absolute_path(
        training.get("output_dir"),
        role="adapter",
        kind="directory" if phase == "resume" else None,
    )
    if phase == "partial":
        if adapter.exists():
            _fail("partial_adapter_path_already_exists")
    else:
        receipt_path = adapter / "receipt.json"
        _receipt_raw, receipt = _read_json(receipt_path, role="training_receipt")
        if (
            receipt.get("complete") is not False
            or receipt.get("halt_reason") != "wall_clock"
            or receipt.get("steps") != phase_contract.get("expected_resume_step")
            or receipt.get("objective_schema") != "aura.recurrence_native_objective.v3"
        ):
            _fail("resume_checkpoint_state_mismatch")
    run_dir = _absolute_path(phase_contract.get("run_dir"), role="run_dir")
    try:
        run_dir.relative_to(amendment_path.parent)
    except ValueError:
        _fail("run_dir_outside_protocol_root")
    if run_dir.exists():
        _fail("run_dir_already_exists")
    envelope_out = _absolute_path(envelope.get("envelope_out"), role="envelope_out")
    try:
        envelope_out.relative_to(adapter)
    except ValueError:
        _fail("envelope_out_outside_adapter")
    if phase == "partial" and envelope_out.exists():
        _fail("envelope_out_already_exists")
    return protocol, amendment, {"wrapper": wrapper, "trainer": trainer}


def _target_command(
    protocol: Mapping[str, Any],
    amendment: Mapping[str, Any],
    sources: Mapping[str, Path],
    *,
    phase: str = "resume",
) -> list[str]:
    training = protocol["training"]
    envelope = amendment["resource_envelope"]
    phase_contract = amendment[phase]
    max_minutes = phase_contract.get("max_minutes")
    if max_minutes is None:
        max_minutes = (
            training.get("hard_training_minutes_after_resume")
            if phase == "resume"
            else protocol.get("detached_execution", {}).get("partial_max_minutes")
        )
    python = REPO_ROOT / ".venv/bin/python"
    if not python.exists():
        _fail("python_launcher_missing")
    model = _absolute_path(training["model_path"] if "model_path" in training else protocol["model"]["path"], role="model", kind="directory")
    adapter = _absolute_path(
        training["output_dir"],
        role="adapter",
        kind="directory" if phase == "resume" else None,
    )
    command = [
        str(python),
        str(sources["wrapper"]),
        "--memory-limit-gb",
        str(envelope["memory_limit_gb"]),
        "--cache-limit-gb",
        str(envelope["cache_limit_gb"]),
        "--wired-limit-gb",
        str(envelope["wired_limit_gb"]),
        "--envelope-out",
        str(envelope["envelope_out"]),
        "--trainer",
        str(sources["trainer"]),
        "--",
        "--model",
        str(model),
        "--out-dir",
        str(adapter),
        "--adapter-id",
        str(training["adapter_id"]),
        "--personality-adapter",
        "none",
        "--train-seed",
        str(_positive_int(training["train_seed"], role="train_seed")),
        "--families",
        _csv(training["families"], role="families"),
        "--task-depths",
        _csv(training["task_depths"], role="task_depths"),
        "--per-cell",
        str(_positive_int(training["per_cell"], role="per_cell")),
        "--curriculum-depths",
        _csv(training["curriculum_depths"], role="curriculum_depths"),
        "--n-slots",
        str(_positive_int(training["n_slots"], role="n_slots")),
        "--branch-roles",
        _csv(training["branch_roles"], role="branch_roles"),
        "--exchange-interval",
        str(_positive_int(training["exchange_interval"], role="exchange_interval")),
        "--alpha",
        str(_finite_number(training["alpha"], role="alpha")),
        "--alpha-schedule",
        str(training["alpha_schedule"]),
        "--lora-rank",
        str(_positive_int(training["lora_rank"], role="lora_rank")),
        "--lora-targets",
        _csv(training["lora_targets"], role="lora_targets"),
        "--learning-rate",
        str(_finite_number(training["learning_rate"], role="learning_rate")),
        "--monotonicity-weight",
        str(_finite_number(training["monotonicity_weight"], role="monotonicity_weight")),
        "--objective",
        str(training["objective"]),
        "--depth-margin",
        str(_finite_number(training["depth_margin"], role="depth_margin")),
        "--diversity-weight",
        str(_finite_number(training["diversity_weight"], role="diversity_weight")),
        "--diversity-target-cos",
        str(_finite_number(training["diversity_target_cos"], role="diversity_target_cos")),
        "--bridge-policy",
        str(training["bridge_policy"]),
        "--holdout-per-cell",
        str(_positive_int(training["holdout_per_cell"], role="holdout_per_cell")),
        "--holdout-eval-samples",
        str(_positive_int(training["holdout_eval_samples"], role="holdout_eval_samples")),
        "--max-minutes",
        str(_finite_number(max_minutes, role="max_minutes")),
        "--max-steps",
        str(_positive_int(training["max_steps"], role="max_steps")),
        "--checkpoint-every",
        str(_positive_int(phase_contract["checkpoint_every_steps"], role="checkpoint_every")),
        "--log-every",
        str(_positive_int(phase_contract["log_every_steps"], role="log_every")),
    ]
    if phase == "resume":
        command.append("--resume")
    return command


def build_launch_command(
    protocol_path: Path,
    amendment_path: Path,
    *,
    phase: str = "resume",
) -> tuple[list[str], list[str]]:
    protocol, amendment, sources = _validate_contract(
        protocol_path,
        amendment_path,
        phase=phase,
    )
    model = _absolute_path(protocol["model"]["path"], role="model", kind="directory")
    _validate_no_competing_model_process(model)
    target = _target_command(protocol, amendment, sources, phase=phase)
    phase_contract = amendment[phase]
    launcher = [
        str(REPO_ROOT / ".venv/bin/python"),
        str((REPO_ROOT / "tools/run_detached_step.py").resolve(strict=True)),
        "launch",
        "--run-dir",
        str(phase_contract["run_dir"]),
        "--name",
        str(phase_contract["name"]),
        "--cwd",
        str(REPO_ROOT),
        "--timeout",
        str(_positive_int(phase_contract["timeout_seconds"], role="timeout")),
        "--",
        *target,
    ]
    return launcher, target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--phase", choices=("partial", "resume"), default="resume")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    launcher, target = build_launch_command(
        args.protocol,
        args.amendment,
        phase=args.phase,
    )
    if args.dry_run:
        print(
            json.dumps(
                {"launcher": launcher, "target": target},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    completed = subprocess.run(launcher, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        _fail("detached_launcher_failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
