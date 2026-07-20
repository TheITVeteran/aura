#!/usr/bin/env python3
"""Verify resident-v3 training before adapter freeze or capability evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.adapter_identity import (  # noqa: E402
    inspect_mlx_tensor_metadata,
)
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (  # noqa: E402
    OBJECTIVE_SCHEMA_V3,
    canonical_json_bytes,
    full_weight_checkpoint_identity,
    personality_bundle_identity,
    runtime_environment_identity,
    strict_json_loads,
    validate_v2_adapter_identity,
)
from core.runtime.atomic_writer import atomic_write_bytes  # noqa: E402
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402
from core.runtime.resource_stage_guard import (  # noqa: E402
    ResourceStageGuardError,
    ack_path,
    lease_request_path,
    read_armed_ack,
    read_compute_lease_ack,
    read_compute_lease_request,
    read_ready_marker,
)
from tools import run_detached_step as detached  # noqa: E402
from tools.run_latent_cortex_paired_campaign import (  # noqa: E402
    model_behavior_bundle_identity,
)

SCHEMA = "aura.resident_v3_training_admission.v1"
PROTOCOL_SCHEMA = "aura.recurrence_native_resident_protocol.v2"
AMENDMENT_SCHEMA = "aura.recurrence_native_resident_protocol_amendment.v1"
RESOURCE_SCHEMA = "aura.recurrence_training_resource_envelope.v1"
PARTIAL_CHECKPOINT_EVIDENCE_SCHEMA = "aura.recurrence_partial_checkpoint_evidence.v1"
TRAINING_CHECKPOINT_SCHEMA = "aura.recurrence_native_checkpoint.v3"
_MAX_JSON_BYTES = 256 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 1 << 40
_GIB = 1024**3


class ResidentV3TrainingAdmissionError(RuntimeError):
    """Stable fail-closed admission error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise ResidentV3TrainingAdmissionError(code)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Never:
    raise ValueError(f"non-finite JSON value: {value}")


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
        raise ResidentV3TrainingAdmissionError(f"{role}_invalid") from exc
    if not isinstance(parsed, dict):
        _fail(f"{role}_invalid")
    return raw, parsed


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_binding(path: Path, *, role: str) -> dict[str, Any]:
    if path.is_symlink():
        _fail(f"{role}_symlink_rejected")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ResidentV3TrainingAdmissionError(f"{role}_unavailable") from exc
    if not resolved.is_file():
        _fail(f"{role}_unavailable")
    raw = read_stable_bytes(resolved, max_bytes=_MAX_ARTIFACT_BYTES)
    return {
        "path": str(resolved),
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
    }


def _binding_matches(raw: bytes, binding: Mapping[str, Any]) -> bool:
    return binding.get("sha256") == _sha256(raw) and binding.get("size_bytes") == len(raw)


def _project_loss_trail(
    committed: Any,
    pending_losses: Any,
    pending_cosines: Any,
    *,
    step: int,
) -> list[dict[str, Any]]:
    if (
        not isinstance(committed, list)
        or any(not isinstance(entry, dict) for entry in committed)
        or not isinstance(pending_losses, list)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in pending_losses
        )
        or not isinstance(pending_cosines, list)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in pending_cosines
        )
    ):
        _fail("training_checkpoint_exact_evidence_invalid")
    projected = [dict(entry) for entry in committed]
    if pending_losses:
        terminal: dict[str, Any] = {
            "step": step,
            "mean_loss": round(sum(pending_losses) / len(pending_losses), 6),
            "window_steps": len(pending_losses),
            "partial_window": True,
        }
        if pending_cosines:
            terminal["pairwise_cos_mean"] = round(sum(pending_cosines) / len(pending_cosines), 6)
        projected.append(terminal)
    return projected


def _checkpoint_complete(
    adapter_dir: Path,
    checkpoint_relative: Any,
    *,
    expected_binding: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        not isinstance(checkpoint_relative, str)
        or not checkpoint_relative.startswith("checkpoints/")
        or Path(checkpoint_relative).parent != Path("checkpoints")
    ):
        _fail("training_checkpoint_path_invalid")
    try:
        checkpoint_dir = (adapter_dir / checkpoint_relative).resolve(strict=True)
        checkpoint_dir.relative_to((adapter_dir / "checkpoints").resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ResidentV3TrainingAdmissionError("training_checkpoint_path_invalid") from exc
    complete_path = checkpoint_dir / "complete.json"
    complete_raw, complete = _read_json(
        complete_path,
        role="training_checkpoint_complete",
    )
    binding = _file_binding(complete_path, role="training_checkpoint_complete")
    if expected_binding is not None and (
        expected_binding.get("sha256") != binding["sha256"]
        or expected_binding.get("size_bytes") != binding["size_bytes"]
        or expected_binding.get("path") != binding["path"]
    ):
        _fail("partial_checkpoint_completion_binding_mismatch")
    if (
        complete.get("schema") != TRAINING_CHECKPOINT_SCHEMA
        or complete.get("checkpoint_id") != checkpoint_dir.name
    ):
        _fail("training_checkpoint_schema_invalid")
    for role in ("adapter", "optimizer"):
        declared = complete.get(role)
        if not isinstance(declared, Mapping) or set(declared) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            _fail(f"training_checkpoint_{role}_binding_invalid")
        artifact = checkpoint_dir / str(declared["path"])
        observed = _file_binding(artifact, role=f"training_checkpoint_{role}")
        if (
            artifact.parent != checkpoint_dir
            or declared.get("sha256") != observed["sha256"]
            or declared.get("size_bytes") != observed["size_bytes"]
        ):
            _fail(f"training_checkpoint_{role}_binding_mismatch")
    return complete, binding


def _verify_partial_checkpoint_evidence(
    *,
    evidence_path: Path,
    adapter_dir: Path,
    protocol_raw: bytes,
    amendment_raw: bytes,
    trainer_sha256: str,
    expected_step: int,
    partial_finished_at: float,
    resume_started_at: float,
) -> dict[str, Any]:
    _raw, evidence = _read_json(
        evidence_path,
        role="partial_checkpoint_evidence",
    )
    claimed = evidence.get("evidence_sha256")
    material = dict(evidence)
    material.pop("evidence_sha256", None)
    captured_at = evidence.get("captured_at")
    complete, binding = _checkpoint_complete(
        adapter_dir,
        evidence.get("checkpoint"),
        expected_binding=(
            evidence.get("checkpoint_complete_binding")
            if isinstance(evidence.get("checkpoint_complete_binding"), Mapping)
            else None
        ),
    )
    receipt = evidence.get("training_receipt")
    if (
        evidence.get("schema") != PARTIAL_CHECKPOINT_EVIDENCE_SCHEMA
        or claimed != _sha256(canonical_json_bytes(material))
        or evidence.get("protocol_sha256") != _sha256(protocol_raw)
        or evidence.get("amendment_sha256") != _sha256(amendment_raw)
        or evidence.get("trainer_sha256") != trainer_sha256
        or not isinstance(captured_at, (int, float))
        or isinstance(captured_at, bool)
        or not partial_finished_at <= float(captured_at) <= resume_started_at
        or evidence.get("checkpoint_complete") != complete
        or not isinstance(receipt, Mapping)
        or receipt.get("complete") is not False
        or receipt.get("halt_reason") != "wall_clock"
        or receipt.get("steps") != expected_step
        or complete.get("step") != expected_step
        or complete.get("loss_trail") != []
        or len(complete.get("pending_window_losses") or []) != expected_step
        or complete.get("holdout_trail") != []
        or complete.get("holdout_eval_count") != 0
        or receipt.get("loss_trail")
        != _project_loss_trail(
            complete.get("loss_trail"),
            complete.get("pending_window_losses"),
            complete.get("pending_window_cosines"),
            step=expected_step,
        )
        or not isinstance(receipt.get("holdout_trail"), list)
        or len(receipt["holdout_trail"]) != 1
        or receipt["holdout_trail"][0].get("step") != expected_step
    ):
        _fail("partial_checkpoint_evidence_invalid")
    checkpoint_dir = (adapter_dir / str(evidence["checkpoint"])).resolve(strict=True)
    tensor_evidence = evidence.get("checkpoint_tensors")
    if not isinstance(tensor_evidence, Mapping):
        _fail("partial_checkpoint_tensor_evidence_invalid")
    for role in ("adapter", "optimizer"):
        declared = complete[role]
        observed = _file_binding(
            checkpoint_dir / str(declared["path"]),
            role=f"partial_checkpoint_{role}",
        )
        if tensor_evidence.get(role) != observed:
            _fail("partial_checkpoint_tensor_evidence_mismatch")
    return {
        "evidence_sha256": claimed,
        "checkpoint": evidence["checkpoint"],
        "checkpoint_complete_sha256": binding["sha256"],
        "pending_loss_count": len(complete["pending_window_losses"]),
        "durable_holdout_eval_count": complete["holdout_eval_count"],
        "captured_at": captured_at,
    }


def _verify_terminal_checkpoint_state(
    adapter_dir: Path,
    receipt: Mapping[str, Any],
    *,
    log_every: int,
) -> dict[str, Any]:
    latest_raw, latest = _read_json(
        adapter_dir / "latest.json",
        role="training_latest",
    )
    complete, binding = _checkpoint_complete(
        adapter_dir,
        latest.get("checkpoint"),
    )
    steps = receipt.get("steps")
    if type(steps) is not int or steps < 1:
        _fail("training_checkpoint_step_invalid")
    pending_losses = complete.get("pending_window_losses")
    holdout_trail = complete.get("holdout_trail")
    holdout_count = complete.get("holdout_eval_count")
    complete_run = receipt.get("complete") is True
    if (
        latest.get("schema") != "aura.recurrence_native_checkpoint_pointer.v1"
        or latest.get("complete_sha256") != binding["sha256"]
        or receipt.get("final_checkpoint") != complete.get("checkpoint_id")
        or complete.get("step") != steps
        or type(holdout_count) is not int
        or holdout_count < 0
        or not isinstance(holdout_trail, list)
        or holdout_count != len(holdout_trail)
        or (complete_run and pending_losses != [])
        or (
            not complete_run
            and isinstance(pending_losses, list)
            and len(pending_losses) != steps % log_every
        )
        or receipt.get("loss_trail")
        != _project_loss_trail(
            complete.get("loss_trail"),
            pending_losses,
            complete.get("pending_window_cosines"),
            step=steps,
        )
    ):
        _fail("terminal_checkpoint_exact_evidence_mismatch")
    receipt_holdout = receipt.get("holdout_trail")
    if not isinstance(receipt_holdout, list):
        _fail("terminal_checkpoint_holdout_evidence_invalid")
    if complete_run:
        if receipt_holdout != holdout_trail:
            _fail("terminal_checkpoint_holdout_evidence_mismatch")
    elif receipt_holdout != holdout_trail:
        if (
            len(receipt_holdout) != len(holdout_trail) + 1
            or receipt_holdout[:-1] != holdout_trail
            or receipt_holdout[-1].get("step") != steps
        ):
            _fail("terminal_checkpoint_holdout_evidence_mismatch")
    return {
        "checkpoint": latest["checkpoint"],
        "checkpoint_complete_sha256": binding["sha256"],
        "latest_sha256": _sha256(latest_raw),
        "pending_loss_count": len(pending_losses or []),
        "holdout_eval_count": holdout_count,
    }


def _contained_file(root: Path, relative: Any, *, role: str) -> Path:
    if not isinstance(relative, str) or not relative:
        _fail(f"{role}_path_invalid")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        _fail(f"{role}_path_invalid")
    candidate = root.joinpath(*pure.parts)
    if candidate.is_symlink():
        _fail(f"{role}_symlink_rejected")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ResidentV3TrainingAdmissionError(f"{role}_path_invalid") from exc
    if not resolved.is_file():
        _fail(f"{role}_path_invalid")
    return resolved


def _artifact_payload(
    root: Path,
    binding: Mapping[str, Any],
    *,
    role: str,
) -> tuple[Path, bytes]:
    path = _contained_file(root, binding.get("path"), role=role)
    raw = read_stable_bytes(path, max_bytes=_MAX_ARTIFACT_BYTES)
    if not _binding_matches(raw, binding):
        _fail(f"{role}_binding_mismatch")
    return path, raw


def _adapter_artifacts(
    adapter_dir: Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, bytes], Path]:
    artifacts: dict[str, bytes] = {}
    adapter_path: Path | None = None
    for role in (
        "adapter",
        "adapter_alias",
        "loader_config",
        "training_receipt",
        "training_config",
        "dataset_manifest",
        "execution_spec",
    ):
        binding = manifest.get(role)
        if not isinstance(binding, Mapping):
            _fail(f"manifest_{role}_invalid")
        path, raw = _artifact_payload(adapter_dir, binding, role=role)
        relative = str(binding["path"])
        if relative in artifacts:
            _fail("manifest_artifact_path_duplicate")
        artifacts[relative] = raw
        if role == "adapter":
            adapter_path = path
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        _fail("manifest_sources_invalid")
    for source_role, value in sources.items():
        if not isinstance(source_role, str) or not isinstance(value, Mapping):
            _fail("manifest_sources_invalid")
        binding = {
            "path": value.get("snapshot_path"),
            "sha256": value.get("sha256"),
            "size_bytes": value.get("size_bytes"),
        }
        _path, raw = _artifact_payload(
            adapter_dir,
            binding,
            role=f"source_{source_role}",
        )
        relative = str(binding["path"])
        if relative in artifacts:
            _fail("manifest_artifact_path_duplicate")
        artifacts[relative] = raw
    completion = _contained_file(
        adapter_dir,
        "training_completion.json",
        role="training_completion",
    )
    artifacts["training_completion.json"] = read_stable_bytes(
        completion,
        max_bytes=_MAX_JSON_BYTES,
    )
    if adapter_path is None:
        _fail("adapter_artifact_missing")
    return artifacts, adapter_path


def _detached_terminal(run_dir: Path, *, role: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if run_dir.is_symlink():
        _fail(f"{role}_run_dir_invalid")
    try:
        resolved = run_dir.resolve(strict=True)
    except OSError as exc:
        raise ResidentV3TrainingAdmissionError(f"{role}_run_dir_invalid") from exc
    if not resolved.is_dir():
        _fail(f"{role}_run_dir_invalid")
    inspection = detached._status(resolved)
    receipt = inspection.get("receipt")
    if (
        inspection.get("terminal") is not True
        or inspection.get("completion_indeterminate") is not False
        or inspection.get("supervisor_alive") is not False
        or inspection.get("child_state") != "dead"
        or not isinstance(receipt, dict)
    ):
        _fail(f"{role}_not_terminal")
    plan_path = resolved / detached.PLAN_FILE
    _plan_raw, plan = _read_json(plan_path, role=f"{role}_plan")
    try:
        detached._verify_plan(plan, plan_path)
        persisted = detached._verified_receipt(resolved / detached.RECEIPT_FILE)
    except Exception as exc:
        raise ResidentV3TrainingAdmissionError(f"{role}_journal_invalid") from exc
    if persisted != receipt:
        _fail(f"{role}_receipt_mismatch")
    if (
        receipt.get("command") != plan.get("command")
        or receipt.get("command_sha256") != plan.get("command_sha256")
        or receipt.get("plan_sha256") != plan.get("plan_sha256")
        or receipt.get("containment_verified") is not True
        or receipt.get("restart_count") != 0
        or receipt.get("timed_out") is not False
        or receipt.get("process_group_empty") is not True
        or receipt.get("lineage_empty") is not True
        or receipt.get("supervisor_error") is not None
        or receipt.get("supervisor_error_type") is not None
    ):
        _fail(f"{role}_terminal_contract_invalid")
    return plan, receipt


def _option_map(argv: Sequence[Any], *, role: str) -> dict[str, str]:
    values = [str(value) for value in argv]
    result: dict[str, str] = {}
    index = 0
    while index < len(values):
        token = values[index]
        if token == "--resume" and index == len(values) - 1:
            result[token] = "true"
            index += 1
            continue
        if not token.startswith("--") or index + 1 >= len(values):
            _fail(f"{role}_argv_invalid")
        if token in result or values[index + 1].startswith("--"):
            _fail(f"{role}_argv_invalid")
        result[token] = values[index + 1]
        index += 2
    return result


def _verify_resume_command(
    plan: Mapping[str, Any],
    protocol: Mapping[str, Any],
    amendment: Mapping[str, Any],
    *,
    phase: str = "resume",
    source_root: Path | None = None,
) -> dict[str, Any]:
    source_root = source_root or REPO_ROOT
    if phase not in {"partial", "resume"}:
        _fail("training_phase_invalid")
    command = plan.get("command")
    if not isinstance(command, list) or len(command) < 16:
        _fail("resume_command_invalid")
    envelope = amendment.get("resource_envelope")
    training = protocol.get("training")
    phase_contract = amendment.get(phase)
    sentinel = amendment.get("sentinel")
    if (
        not isinstance(envelope, Mapping)
        or not isinstance(training, Mapping)
        or not isinstance(phase_contract, Mapping)
        or not isinstance(sentinel, Mapping)
    ):
        _fail("resume_contract_invalid")
    try:
        separator = command.index("--")
    except ValueError:
        _fail("resume_command_not_enveloped")
    wrapper_options = _option_map(command[2:separator], role="resource_wrapper")
    trainer_options = _option_map(command[separator + 1 :], role="trainer")
    try:
        memory_limit = float(wrapper_options.get("--memory-limit-gb", "nan"))
        cache_limit = float(wrapper_options.get("--cache-limit-gb", "nan"))
        wired_limit = float(wrapper_options.get("--wired-limit-gb", "nan"))
        startup_lethal_mb = float(trainer_options.get("--resource-startup-lethal-mb", "nan"))
        steady_lethal_mb = float(trainer_options.get("--resource-steady-lethal-mb", "nan"))
        max_steps = int(trainer_options.get("--max-steps", "-1"))
        checkpoint_every = int(trainer_options.get("--checkpoint-every", "-1"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResidentV3TrainingAdmissionError("resume_command_contract_mismatch") from exc
    if (
        command[1] != str(source_root / envelope.get("wrapper", ""))
        or wrapper_options.get("--trainer") != str(source_root / envelope.get("trainer", ""))
        or memory_limit != 40.0
        or cache_limit != 2.0
        or wired_limit != 48.0
        or wrapper_options.get("--envelope-out") != envelope.get("envelope_out")
        or (trainer_options.get("--resume") == "true") != (phase == "resume")
        or trainer_options.get("--out-dir") != training.get("output_dir")
        or trainer_options.get("--adapter-id") != training.get("adapter_id")
        or trainer_options.get("--objective") != "v3"
        or trainer_options.get("--resource-stage-path") != sentinel.get(f"{phase}_stage_path")
        or startup_lethal_mb != 73728.0
        or steady_lethal_mb != 59392.0
        or max_steps != training.get("max_steps")
        or checkpoint_every != phase_contract.get("checkpoint_every_steps")
    ):
        _fail("resume_command_contract_mismatch")
    return {
        "plan_sha256": plan.get("plan_sha256"),
        "command_sha256": plan.get("command_sha256"),
        "phase": phase,
        "resource_limits_gb": {"active": 40, "cache": 2, "wired": 48},
    }


def _verify_resource_envelope(
    path: Path,
    amendment: Mapping[str, Any],
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    source_root = source_root or REPO_ROOT
    raw, resource = _read_json(path, role="resource_envelope")
    expected = amendment.get("resource_envelope")
    if not isinstance(expected, Mapping):
        _fail("resource_envelope_contract_invalid")
    wrapper = source_root / str(expected.get("wrapper", ""))
    trainer = source_root / str(expected.get("trainer", ""))
    limits = {
        "memory_limit_bytes": 40 * _GIB,
        "cache_limit_bytes": 2 * _GIB,
        "wired_limit_bytes": 48 * _GIB,
    }
    if (
        resource.get("schema") != RESOURCE_SCHEMA
        or resource.get("cache_cleared_before_model_load") is not True
        or any(resource.get(key) != value for key, value in limits.items())
        or resource.get("wrapper_sha256") != _file_binding(wrapper, role="wrapper")["sha256"]
        or resource.get("trainer_sha256") != _file_binding(trainer, role="trainer")["sha256"]
    ):
        _fail("resource_envelope_mismatch")
    return {**limits, "sha256": _sha256(raw)}


def _finite_trail(value: Any, *, role: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        _fail(f"{role}_missing")
    result: list[dict[str, Any]] = []
    previous_step = -1
    for entry in value:
        if not isinstance(entry, Mapping):
            _fail(f"{role}_invalid")
        step = entry.get("step")
        loss = entry.get("mean_loss")
        if (
            type(step) is not int
            or step < previous_step
            or isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
            or float(loss) < 0.0
        ):
            _fail(f"{role}_invalid")
        previous_step = step
        result.append(dict(entry))
    return result


def evaluate_training_state(
    receipt: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    minimum_partial_steps: int = 200,
    holdout_ratio_limit: float = 1.5,
) -> dict[str, Any]:
    max_steps = config.get("max_steps")
    steps = receipt.get("steps")
    if type(max_steps) is not int or max_steps < 1 or type(steps) is not int:
        _fail("training_step_state_invalid")
    if receipt.get("complete") is True:
        if receipt.get("halt_reason") != "max_steps" or steps != max_steps:
            _fail("training_completion_state_invalid")
        scope = "complete_training"
    elif receipt.get("complete") is False:
        if (
            receipt.get("halt_reason") != "wall_clock"
            or not minimum_partial_steps <= steps < max_steps
        ):
            _fail("bounded_partial_not_admissible")
        scope = "bounded_partial_training"
    else:
        _fail("training_completion_state_invalid")
    loss_trail = _finite_trail(receipt.get("loss_trail"), role="loss_trail")
    holdout_trail = _finite_trail(receipt.get("holdout_trail"), role="holdout_trail")
    if loss_trail[-1]["step"] != steps or holdout_trail[-1]["step"] != steps:
        _fail("terminal_trail_step_mismatch")
    holdout_losses = [float(entry["mean_loss"]) for entry in holdout_trail]
    minimum = min(holdout_losses)
    final = holdout_losses[-1]
    threshold = minimum * holdout_ratio_limit
    if final > threshold:
        _fail("holdout_overfitting_guard_failed")
    return {
        "scope": scope,
        "complete": scope == "complete_training",
        "steps": steps,
        "max_steps": max_steps,
        "loss_observations": len(loss_trail),
        "holdout_observations": len(holdout_trail),
        "holdout_minimum_mean_loss": minimum,
        "holdout_final_mean_loss": final,
        "holdout_limit_ratio": holdout_ratio_limit,
        "holdout_observed_ratio": final / minimum
        if minimum > 0.0
        else (1.0 if final == 0.0 else math.inf),
    }


def _verify_footprint(
    ring_path: Path,
    sentinel_dir: Path,
    *,
    trainer_pid: int,
    stage_path: Path,
    expected_trainer_sha256: str,
    command_ring_path: Path | None = None,
) -> dict[str, Any]:
    plan, receipt = _detached_terminal(sentinel_dir, role="memory_sentinel")
    if receipt.get("returncode") != 0 or receipt.get("status") != "passed":
        _fail("memory_sentinel_failed")
    command = plan.get("command")
    if not isinstance(command, list):
        _fail("memory_sentinel_command_invalid")
    options = _option_map(command[2:], role="memory_sentinel")
    try:
        lethal_mb = float(options.get("--lethal-mb", "nan"))
        startup_lethal_mb = float(options.get("--startup-lethal-mb", "nan"))
        interval_s = float(options.get("--interval", "nan"))
        overshoot_factor = float(options.get("--immediate-kill-overshoot", "nan"))
        ring_window_s = float(options.get("--ring-window-seconds", "nan"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResidentV3TrainingAdmissionError("memory_sentinel_command_mismatch") from exc
    if (
        command[1] != str(REPO_ROOT / "tools/memory_sentinel.py")
        or int(options.get("--pid", "-1")) != trainer_pid
        or lethal_mb != 59392.0
        or startup_lethal_mb != 73728.0
        or interval_s != 0.5
        or overshoot_factor != 1.05
        or ring_window_s != 46800.0
        or Path(options.get("--steady-marker", "")) != stage_path
        or Path(options.get("--ring", "")) != (command_ring_path or ring_path)
    ):
        _fail("memory_sentinel_command_mismatch")
    if any(sentinel_dir.glob("sentinel_tombstone_*.json")):
        _fail("memory_sentinel_lethal_abort")
    raw = read_stable_bytes(ring_path, max_bytes=_MAX_JSON_BYTES)
    samples: list[dict[str, Any]] = []
    try:
        for line in raw.splitlines():
            sample = json.loads(
                line,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_nonfinite,
            )
            if not isinstance(sample, dict):
                _fail("memory_sample_invalid")
            samples.append(sample)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ResidentV3TrainingAdmissionError("memory_sample_invalid") from exc
    if not samples:
        _fail("memory_samples_missing")
    stages: list[str] = []
    values_by_stage: dict[str, list[float]] = {
        "startup": [],
        "steady": [],
        "compute": [],
        "draining": [],
    }
    lease_workloads: dict[int, str] = {}
    for sample in samples:
        managed = sample.get("managed_mb")
        stage = sample.get("guard_stage")
        active_lethal = sample.get("active_lethal_mb")
        marker_observed = sample.get("marker_observed")
        lease_sequence = sample.get("lease_sequence")
        lease_workload = sample.get("lease_workload")
        expected_lethal = lethal_mb if stage == "steady" else startup_lethal_mb
        if (
            isinstance(managed, bool)
            or not isinstance(managed, (int, float))
            or not math.isfinite(float(managed))
            or float(managed) < 0.0
            or stage not in values_by_stage
            or active_lethal != expected_lethal
            or type(marker_observed) is not bool
            or (stage != "startup" and marker_observed is not True)
            or type(lease_sequence) is not int
            or lease_sequence < 1
            or not isinstance(lease_workload, str)
            or (stage in {"compute", "draining"} and not lease_workload)
            or (stage in {"startup", "steady"} and lease_workload)
        ):
            _fail("memory_sample_invalid")
        if float(managed) >= expected_lethal:
            _fail("memory_envelope_exceeded")
        stages.append(stage)
        values_by_stage[stage].append(float(managed))
        if stage in {"compute", "draining"}:
            prior = lease_workloads.setdefault(lease_sequence, lease_workload)
            if prior != lease_workload:
                _fail("memory_lease_workload_changed")
    if any(not values_by_stage[stage] for stage in values_by_stage):
        _fail("memory_guard_stage_evidence_missing")
    allowed_transitions = {
        "startup": {"startup", "steady"},
        "steady": {"steady", "compute"},
        "compute": {"compute", "draining"},
        "draining": {"draining", "steady"},
    }
    if any(
        following not in allowed_transitions[current]
        for current, following in zip(stages, stages[1:], strict=False)
    ):
        _fail("memory_guard_stage_regressed")
    try:
        marker, marker_raw = read_ready_marker(
            stage_path,
            expected_target_pid=trainer_pid,
        )
        acknowledgement, acknowledgement_raw = read_armed_ack(
            stage_path,
            marker_raw=marker_raw,
            expected_target_pid=trainer_pid,
            startup_lethal_mb=startup_lethal_mb,
            steady_lethal_mb=lethal_mb,
        )
    except ResourceStageGuardError as exc:
        raise ResidentV3TrainingAdmissionError("memory_guard_handshake_invalid") from exc
    if marker.get("trainer_sha256") != expected_trainer_sha256 or acknowledgement.get(
        "sentinel_pid"
    ) != receipt.get("child_pid"):
        _fail("memory_guard_handshake_binding_mismatch")
    sequences = sorted(lease_workloads)
    if not sequences or sequences != list(range(1, sequences[-1] + 1)):
        _fail("memory_compute_lease_sequence_invalid")
    predecessor_ack_raw = acknowledgement_raw
    lease_chain = hashlib.sha256()
    workload_counts: dict[str, int] = {}
    for sequence in sequences:
        try:
            acquire_path, acquire, acquire_raw = read_compute_lease_request(
                stage_path,
                marker_raw=marker_raw,
                expected_target_pid=trainer_pid,
                sequence=sequence,
                workload=None,
                action="acquire",
                predecessor_ack_raw=predecessor_ack_raw,
            )
            workload = str(acquire["workload"])
            acquire_ack, acquire_ack_raw = read_compute_lease_ack(
                acquire_path,
                request_raw=acquire_raw,
                expected_target_pid=trainer_pid,
                sequence=sequence,
                workload=workload,
                action="acquire",
                active_lethal_mb=startup_lethal_mb,
            )
            release_path, _release, release_raw = read_compute_lease_request(
                stage_path,
                marker_raw=marker_raw,
                expected_target_pid=trainer_pid,
                sequence=sequence,
                workload=workload,
                action="release",
                predecessor_ack_raw=acquire_ack_raw,
            )
            release_ack, release_ack_raw = read_compute_lease_ack(
                release_path,
                request_raw=release_raw,
                expected_target_pid=trainer_pid,
                sequence=sequence,
                workload=workload,
                action="release",
                active_lethal_mb=lethal_mb,
            )
        except ResourceStageGuardError as exc:
            raise ResidentV3TrainingAdmissionError("memory_compute_lease_invalid") from exc
        if (
            lease_workloads.get(sequence) != workload
            or acquire_ack.get("sentinel_pid") != receipt.get("child_pid")
            or release_ack.get("sentinel_pid") != receipt.get("child_pid")
        ):
            _fail("memory_compute_lease_binding_mismatch")
        for artifact in (acquire_raw, acquire_ack_raw, release_raw, release_ack_raw):
            lease_chain.update(len(artifact).to_bytes(8, "big"))
            lease_chain.update(artifact)
        predecessor_ack_raw = release_ack_raw
        workload_counts[workload] = workload_counts.get(workload, 0) + 1
    if workload_counts.get("training_step", 0) < 1:
        _fail("memory_training_compute_lease_missing")
    if lease_request_path(
        stage_path,
        sequence=sequences[-1] + 1,
        action="acquire",
    ).exists():
        _fail("memory_compute_lease_incomplete")
    return {
        "sample_count": len(samples),
        "stage_sample_counts": {stage: len(values) for stage, values in values_by_stage.items()},
        "stage_peak_managed_mb": {stage: max(values) for stage, values in values_by_stage.items()},
        "compute_lease_count": len(sequences),
        "compute_lease_workloads": workload_counts,
        "compute_lease_chain_sha256": lease_chain.hexdigest(),
        "startup_lethal_mb": startup_lethal_mb,
        "steady_lethal_mb": lethal_mb,
        "head_at": samples[0].get("at"),
        "tail_at": samples[-1].get("at"),
        "ring_sha256": _sha256(raw),
        "marker_sha256": _sha256(marker_raw),
        "ack_sha256": _sha256(acknowledgement_raw),
        "ack_path": str(ack_path(stage_path)),
        "sentinel_plan_sha256": plan.get("plan_sha256"),
        "sentinel_receipt_sha256": receipt.get("receipt_sha256"),
    }


def _validate_adapter_identity(
    adapter_dir: Path,
    protocol: Mapping[str, Any],
    *,
    allow_bounded_partial: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = adapter_dir / "recurrence_adapter_manifest.json"
    manifest_raw = read_stable_bytes(manifest_path, max_bytes=_MAX_JSON_BYTES)
    try:
        manifest = strict_json_loads(manifest_raw, role="resident_v3_manifest")
    except ValueError as exc:
        raise ResidentV3TrainingAdmissionError("resident_v3_manifest_invalid") from exc
    artifacts, adapter_path = _adapter_artifacts(adapter_dir, manifest)
    receipt = strict_json_loads(
        artifacts[str(manifest["training_receipt"]["path"])],
        role="resident_v3_training_receipt",
    )
    config = strict_json_loads(
        artifacts[str(manifest["training_config"]["path"])],
        role="resident_v3_training_config",
    )
    model = Path(str(protocol["model"]["path"])).expanduser().resolve(strict=True)
    identity = validate_v2_adapter_identity(
        manifest_raw,
        adapter_id=str(protocol["training"]["adapter_id"]),
        actual_base_checkpoint=full_weight_checkpoint_identity(model),
        actual_model_behavior_bundle=model_behavior_bundle_identity(model),
        actual_personality_adapter=personality_bundle_identity(None),
        actual_runtime_environment=runtime_environment_identity(),
        artifacts=artifacts,
        tensor_metadata=inspect_mlx_tensor_metadata(adapter_path),
        allow_bounded_partial=allow_bounded_partial,
    )
    return identity, receipt, config


def _verify_protocol_match(
    protocol: Mapping[str, Any],
    receipt: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    training = protocol.get("training")
    model = protocol.get("model")
    options = config.get("objective_options")
    holdout = config.get("holdout")
    lora = config.get("lora")
    optimizer = config.get("optimizer")
    spec = config.get("execution_spec")
    if not all(
        isinstance(value, Mapping)
        for value in (training, model, options, holdout, lora, optimizer, spec)
    ):
        _fail("protocol_training_contract_invalid")
    if (
        receipt.get("objective_schema") != OBJECTIVE_SCHEMA_V3
        or config.get("objective_schema") != OBJECTIVE_SCHEMA_V3
        or receipt.get("base_checkpoint", {}).get("fingerprint")
        != model.get("expected_full_weight_sha256")
        or config.get("model_path") != model.get("path")
        or config.get("train_seed") != training.get("train_seed")
        or config.get("max_steps") != training.get("max_steps")
        or config.get("curriculum_depths") != training.get("curriculum_depths")
        or config.get("monotonicity_weight") != training.get("monotonicity_weight")
        or options.get("depth_margin") != training.get("depth_margin")
        or options.get("diversity_weight") != training.get("diversity_weight")
        or options.get("diversity_target_cos") != training.get("diversity_target_cos")
        or holdout.get("per_cell") != training.get("holdout_per_cell")
        or holdout.get("count") != training.get("holdout_count")
        or holdout.get("eval_samples") != training.get("holdout_eval_samples")
        or lora.get("rank") != training.get("lora_rank")
        or lora.get("targets") != training.get("lora_targets")
        or optimizer.get("learning_rate") != training.get("learning_rate")
        or spec.get("n_slots") != training.get("n_slots")
        or spec.get("branch_roles") != training.get("branch_roles")
        or spec.get("exchange_interval") != training.get("exchange_interval")
        or spec.get("alpha") != training.get("alpha")
        or spec.get("alpha_schedule") != training.get("alpha_schedule")
    ):
        _fail("protocol_training_contract_mismatch")


def _write_once(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("admission_sha256", None)
    digest = _sha256(canonical_json_bytes(unsigned))
    document = {**unsigned, "admission_sha256": digest}
    raw = canonical_json_bytes(document) + b"\n"
    if path.exists():
        if path.is_symlink() or read_stable_bytes(path, max_bytes=_MAX_JSON_BYTES) != raw:
            _fail("admission_output_exists_different")
        return document
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink():
        _fail("admission_output_parent_symlink")
    atomic_write_bytes(path, raw)
    return document


def verify(args: argparse.Namespace) -> dict[str, Any]:
    protocol_raw, protocol = _read_json(args.protocol, role="protocol")
    amendment_raw, amendment = _read_json(args.amendment, role="amendment")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        _fail("protocol_schema_invalid")
    if amendment.get("schema") != AMENDMENT_SCHEMA:
        _fail("amendment_schema_invalid")
    parent = amendment.get("parent_protocol")
    if not isinstance(parent, Mapping) or not _binding_matches(protocol_raw, parent):
        _fail("parent_protocol_binding_mismatch")

    source_root = Path(
        getattr(args, "training_source_root", None) or REPO_ROOT
    ).expanduser().resolve(strict=True)
    if not source_root.is_dir():
        _fail("training_source_root_invalid")
    partial_plan, partial_receipt = _detached_terminal(
        Path(protocol["detached_execution"]["partial_run_dir"]),
        role="forced_partial",
    )
    if (
        partial_receipt.get("returncode")
        != protocol["detached_execution"].get("expected_partial_returncode")
        or partial_receipt.get("passed") is not False
    ):
        _fail("forced_partial_result_invalid")
    resume_dir = Path(amendment["resume"]["run_dir"])
    resume_plan, resume_receipt = _detached_terminal(resume_dir, role="resume")
    if resume_receipt.get("returncode") not in {0, 75}:
        _fail("resume_terminal_code_invalid")
    partial_command_evidence = _verify_resume_command(
        partial_plan,
        protocol,
        amendment,
        phase="partial",
        source_root=source_root,
    )
    command_evidence = _verify_resume_command(
        resume_plan,
        protocol,
        amendment,
        phase="resume",
        source_root=source_root,
    )
    resource_path = Path(amendment["resource_envelope"]["envelope_out"])
    resource_evidence = _verify_resource_envelope(
        resource_path,
        amendment,
        source_root=source_root,
    )

    adapter_dir = Path(protocol["training"]["output_dir"]).resolve(strict=True)
    receipt_preview = strict_json_loads(
        read_stable_bytes(adapter_dir / "receipt.json", max_bytes=_MAX_JSON_BYTES),
        role="training_receipt_preview",
    )
    allow_partial = receipt_preview.get("complete") is False
    identity, training_receipt, training_config = _validate_adapter_identity(
        adapter_dir,
        protocol,
        allow_bounded_partial=allow_partial,
    )
    _verify_protocol_match(protocol, training_receipt, training_config)
    state = evaluate_training_state(training_receipt, training_config)
    if state["scope"] == "bounded_partial_training":
        if (
            identity.get("complete") is not False
            or identity.get("load_eligible") is not False
            or identity.get("training_scope") != state["scope"]
        ):
            _fail("bounded_partial_identity_scope_mismatch")
    elif identity.get("complete") is not True:
        _fail("complete_identity_scope_mismatch")
    if resume_receipt.get("returncode") != (0 if state["complete"] else 75):
        _fail("resume_result_training_state_mismatch")
    try:
        partial_finished_at = float(partial_receipt["finished_at"])
        resume_started_at = float(resume_receipt["started_at"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ResidentV3TrainingAdmissionError("detached_training_chronology_invalid") from exc
    if (
        not math.isfinite(partial_finished_at)
        or not math.isfinite(resume_started_at)
        or partial_finished_at > resume_started_at
    ):
        _fail("detached_training_chronology_invalid")
    exact_partial_evidence = _verify_partial_checkpoint_evidence(
        evidence_path=Path(amendment["resume"]["partial_checkpoint_evidence_path"]),
        adapter_dir=adapter_dir,
        protocol_raw=protocol_raw,
        amendment_raw=amendment_raw,
        trainer_sha256=str(amendment["resource_envelope"]["trainer_sha256"]),
        expected_step=int(amendment["resume"]["expected_resume_step"]),
        partial_finished_at=partial_finished_at,
        resume_started_at=resume_started_at,
    )
    terminal_checkpoint_evidence = _verify_terminal_checkpoint_state(
        adapter_dir,
        training_receipt,
        log_every=int(amendment["resume"]["log_every_steps"]),
    )

    partial_footprint = _verify_footprint(
        args.partial_footprint_ring,
        args.partial_sentinel_run_dir,
        trainer_pid=int(partial_receipt["child_pid"]),
        stage_path=Path(amendment["sentinel"]["partial_stage_path"]),
        expected_trainer_sha256=str(amendment["resource_envelope"]["trainer_sha256"]),
    )
    resume_footprint = _verify_footprint(
        args.resume_footprint_ring,
        args.resume_sentinel_run_dir,
        trainer_pid=int(resume_receipt["child_pid"]),
        stage_path=Path(amendment["sentinel"]["resume_stage_path"]),
        expected_trainer_sha256=str(amendment["resource_envelope"]["trainer_sha256"]),
    )
    freeze_eligible = state["scope"] == "complete_training"
    payload = {
        "schema": SCHEMA,
        "decision": (
            "admit_to_freeze_and_mechanics"
            if freeze_eligible
            else "retain_bounded_partial_training_evidence"
        ),
        "claim_scope": "resident_v3_training_mechanics_admission_only",
        "protocol": {"sha256": _sha256(protocol_raw), "path": str(args.protocol)},
        "amendment": {"sha256": _sha256(amendment_raw), "path": str(args.amendment)},
        "forced_partial": {
            **partial_command_evidence,
            "plan_sha256": partial_plan.get("plan_sha256"),
            "receipt_sha256": partial_receipt.get("receipt_sha256"),
            "returncode": partial_receipt.get("returncode"),
            "footprint": partial_footprint,
            "exact_checkpoint": exact_partial_evidence,
        },
        "resume": {
            **command_evidence,
            "receipt_sha256": resume_receipt.get("receipt_sha256"),
            "returncode": resume_receipt.get("returncode"),
        },
        "resource_envelope": resource_evidence,
        "footprint": resume_footprint,
        "training_state": state,
        "terminal_checkpoint": terminal_checkpoint_evidence,
        "identity_receipt": identity,
        "claim_flags": {
            "training_admitted": True,
            "adapter_freeze_eligible": freeze_eligible,
            "mechanics_proven": False,
            "reasoning_gain": False,
            "same_checkpoint_interaction": False,
            "frontier_level": False,
            "frontier_plus": False,
            "installed_desktop_gain": False,
        },
    }
    return _write_once(args.output, payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--partial-sentinel-run-dir", type=Path, required=True)
    parser.add_argument("--partial-footprint-ring", type=Path, required=True)
    parser.add_argument("--resume-sentinel-run-dir", type=Path, required=True)
    parser.add_argument("--resume-footprint-ring", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-source-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    try:
        result = verify(args)
    except ResidentV3TrainingAdmissionError as exc:
        print(json.dumps({"decision": "reject", "reason": exc.code}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
