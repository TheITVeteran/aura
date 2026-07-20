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
from tools import run_detached_step as detached  # noqa: E402
from tools.run_latent_cortex_paired_campaign import (  # noqa: E402
    model_behavior_bundle_identity,
)

SCHEMA = "aura.resident_v3_training_admission.v1"
PROTOCOL_SCHEMA = "aura.recurrence_native_resident_protocol.v2"
AMENDMENT_SCHEMA = "aura.recurrence_native_resident_protocol_amendment.v1"
RESOURCE_SCHEMA = "aura.recurrence_training_resource_envelope.v1"
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
) -> dict[str, Any]:
    if phase not in {"partial", "resume"}:
        _fail("training_phase_invalid")
    command = plan.get("command")
    if not isinstance(command, list) or len(command) < 16:
        _fail("resume_command_invalid")
    envelope = amendment.get("resource_envelope")
    training = protocol.get("training")
    phase_contract = amendment.get(phase)
    if (
        not isinstance(envelope, Mapping)
        or not isinstance(training, Mapping)
        or not isinstance(phase_contract, Mapping)
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
        max_steps = int(trainer_options.get("--max-steps", "-1"))
        checkpoint_every = int(trainer_options.get("--checkpoint-every", "-1"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResidentV3TrainingAdmissionError("resume_command_contract_mismatch") from exc
    if (
        command[1] != str(REPO_ROOT / envelope.get("wrapper", ""))
        or wrapper_options.get("--trainer")
        != str(REPO_ROOT / envelope.get("trainer", ""))
        or memory_limit != 40.0
        or cache_limit != 2.0
        or wired_limit != 48.0
        or wrapper_options.get("--envelope-out") != envelope.get("envelope_out")
        or (trainer_options.get("--resume") == "true") != (phase == "resume")
        or trainer_options.get("--out-dir") != training.get("output_dir")
        or trainer_options.get("--adapter-id") != training.get("adapter_id")
        or trainer_options.get("--objective") != "v3"
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


def _verify_resource_envelope(path: Path, amendment: Mapping[str, Any]) -> dict[str, Any]:
    raw, resource = _read_json(path, role="resource_envelope")
    expected = amendment.get("resource_envelope")
    if not isinstance(expected, Mapping):
        _fail("resource_envelope_contract_invalid")
    wrapper = REPO_ROOT / str(expected.get("wrapper", ""))
    trainer = REPO_ROOT / str(expected.get("trainer", ""))
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
        "holdout_observed_ratio": final / minimum if minimum > 0.0 else (1.0 if final == 0.0 else math.inf),
    }


def _verify_footprint(
    ring_path: Path,
    sentinel_dir: Path,
    *,
    trainer_pid: int,
) -> dict[str, Any]:
    plan, receipt = _detached_terminal(sentinel_dir, role="memory_sentinel")
    if receipt.get("returncode") != 0 or receipt.get("status") != "passed":
        _fail("memory_sentinel_failed")
    command = plan.get("command")
    if not isinstance(command, list):
        _fail("memory_sentinel_command_invalid")
    options = _option_map(command[2:], role="memory_sentinel")
    lethal_mb = float(options.get("--lethal-mb", "nan"))
    if (
        command[1] != str(REPO_ROOT / "tools/memory_sentinel.py")
        or int(options.get("--pid", "-1")) != trainer_pid
        or lethal_mb != 59392.0
        or Path(options.get("--ring", "")) != ring_path
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
    managed = [sample.get("managed_mb") for sample in samples]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in managed
    ):
        _fail("memory_sample_invalid")
    peak = max(float(value) for value in managed)
    if peak >= lethal_mb:
        _fail("memory_envelope_exceeded")
    return {
        "sample_count": len(samples),
        "peak_managed_mb": peak,
        "lethal_mb": lethal_mb,
        "head_at": samples[0].get("at"),
        "tail_at": samples[-1].get("at"),
        "ring_sha256": _sha256(raw),
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
    )
    command_evidence = _verify_resume_command(
        resume_plan,
        protocol,
        amendment,
        phase="resume",
    )
    resource_path = Path(amendment["resource_envelope"]["envelope_out"])
    resource_evidence = _verify_resource_envelope(resource_path, amendment)

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

    partial_footprint = _verify_footprint(
        args.partial_footprint_ring,
        args.partial_sentinel_run_dir,
        trainer_pid=int(partial_receipt["child_pid"]),
    )
    resume_footprint = _verify_footprint(
        args.resume_footprint_ring,
        args.resume_sentinel_run_dir,
        trainer_pid=int(resume_receipt["child_pid"]),
    )
    payload = {
        "schema": SCHEMA,
        "decision": "admit_to_freeze_and_mechanics",
        "claim_scope": "resident_v3_training_mechanics_admission_only",
        "protocol": {"sha256": _sha256(protocol_raw), "path": str(args.protocol)},
        "amendment": {"sha256": _sha256(amendment_raw), "path": str(args.amendment)},
        "forced_partial": {
            **partial_command_evidence,
            "plan_sha256": partial_plan.get("plan_sha256"),
            "receipt_sha256": partial_receipt.get("receipt_sha256"),
            "returncode": partial_receipt.get("returncode"),
            "footprint": partial_footprint,
        },
        "resume": {
            **command_evidence,
            "receipt_sha256": resume_receipt.get("receipt_sha256"),
            "returncode": resume_receipt.get("returncode"),
        },
        "resource_envelope": resource_evidence,
        "footprint": resume_footprint,
        "training_state": state,
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
