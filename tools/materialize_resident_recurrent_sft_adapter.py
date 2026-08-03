#!/usr/bin/env python3
"""Materialize a verified resident recurrent-SFT checkpoint as an adapter package."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, Never, cast

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.adapter_identity import (  # noqa: E402
    TensorIdentity,
    inspect_mlx_tensor_metadata,
)
from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    CampaignJournal,
    CampaignPlan,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.execution_spec import (  # noqa: E402
    RLCExecutionSpec,
)
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (  # noqa: E402
    full_weight_checkpoint_identity,
    model_behavior_bundle_identity,
    runtime_environment_identity,
    strict_json_loads,
)
from core.brain.llm.latent_cortex.resident_recurrent_sft_adapter_identity import (  # noqa: E402
    LEGACY_MANIFEST_SCHEMA,
    MANIFEST_SCHEMA,
    declared_bindings,
    topology_sha256,
    validate_resident_recurrent_sft_adapter_identity,
)
from core.learning.recurrent_sft_execution import (  # noqa: E402
    adapter_tensor_fingerprint,
)
from core.learning.resident_recurrent_sft_bootstrap_authority import (  # noqa: E402
    authorize_bound_artifacts,
    sha256_bytes,
    sha256_json,
    validate_authority,
)
from core.learning.resident_recurrent_sft_bootstrap_execution import (  # noqa: E402
    adapter_topology_sha256,
)
from core.learning.resident_recurrent_sft_bootstrap_state import (  # noqa: E402
    authority_state_bindings,
    inspect_checkpoint,
    validate_checkpoint_state,
)
from core.runtime.atomic_writer import interprocess_file_lock  # noqa: E402
from tools.resident_recurrent_sft_bootstrap_identity import (  # noqa: E402
    absent_personality_identity,
)

PACKAGE_COMPLETION_SCHEMA: Final = "aura.resident_recurrent_sft_adapter_package_completion.v1"
TRAINING_ADMISSION_SCHEMA: Final = "aura.resident_recurrent_sft_training_admission.v1"
LOADER_CONFIG_SCHEMA: Final = "aura.resident_recurrent_sft_adapter_config.v1"
MAX_JSON_BYTES: Final = 64 * 1024 * 1024
MAX_ARTIFACT_BYTES: Final = 1 << 50
_INVOCATION_NAME = re.compile(r"invocation-([0-9]{4})\.json\Z")
_CHECKPOINT_NAME = re.compile(r"sequence-([0-9]{8})-step-([0-9]{8})-([0-9a-f]{32})\Z")


class ResidentRecurrentSFTMaterializationError(RuntimeError):
    """The source campaign cannot be admitted as a complete adapter package."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise ResidentRecurrentSFTMaterializationError(code)


def _reject_symlink_chain(path: Path, *, role: str, require_exists: bool = True) -> Path:
    lexical = path.expanduser().absolute()
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            if require_exists:
                _fail(f"resident_sft_materialize_{role}_missing")
            break
        if stat.S_ISLNK(mode):
            _fail(f"resident_sft_materialize_{role}_symlink_forbidden")
    try:
        return lexical.resolve(strict=require_exists)
    except OSError as exc:
        raise ResidentRecurrentSFTMaterializationError(
            f"resident_sft_materialize_{role}_unavailable"
        ) from exc


def _directory(path: Path, *, role: str) -> Path:
    resolved = _reject_symlink_chain(path, role=role)
    if not resolved.is_dir():
        _fail(f"resident_sft_materialize_{role}_not_directory")
    return resolved


def _contained(root: Path, relative: Any, *, role: str, directory: bool = False) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        _fail(f"resident_sft_materialize_{role}_path_invalid")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or str(pure) != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        _fail(f"resident_sft_materialize_{role}_path_invalid")
    resolved = _reject_symlink_chain(root.joinpath(*pure.parts), role=role)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ResidentRecurrentSFTMaterializationError(
            f"resident_sft_materialize_{role}_outside_capsule"
        ) from exc
    if resolved.is_dir() is not directory:
        _fail(f"resident_sft_materialize_{role}_type_invalid")
    return resolved


def _stable_bytes(path: Path, *, role: str, maximum: int = MAX_ARTIFACT_BYTES) -> bytes:
    resolved = _reject_symlink_chain(path, role=role)
    try:
        before = resolved.stat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _fail(f"resident_sft_materialize_{role}_file_invalid")
        if not 0 < before.st_size <= maximum:
            _fail(f"resident_sft_materialize_{role}_size_invalid")
        payload = resolved.read_bytes()
        after = resolved.stat()
    except OSError as exc:
        raise ResidentRecurrentSFTMaterializationError(
            f"resident_sft_materialize_{role}_read_failed"
        ) from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or len(payload) != before.st_size:
        _fail(f"resident_sft_materialize_{role}_changed_while_reading")
    return payload


def _json(payload: bytes, *, role: str) -> dict[str, Any]:
    if len(payload) > MAX_JSON_BYTES:
        _fail(f"resident_sft_materialize_{role}_size_invalid")
    value = strict_json_loads(payload, role=f"resident_sft_materialize_{role}")
    if not isinstance(value, Mapping):
        _fail(f"resident_sft_materialize_{role}_schema_invalid")
    return dict(value)


def _binding(path: str, payload: bytes) -> dict[str, Any]:
    return {"path": path, "sha256": sha256_bytes(payload), "size_bytes": len(payload)}


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail("resident_sft_materialize_package_short_write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_into(staging: Path, relative: str, payload: bytes) -> dict[str, Any]:
    _write_exclusive(staging / relative, payload)
    return _binding(relative, payload)


def _normalize_training_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(runtime)
    body.pop("identity_sha256", None)
    body.pop("interpreter", None)
    return {**body, "identity_sha256": sha256_json(body)}


def _verify_controller_journal(
    *,
    plan_payload: bytes,
    journal_payload: bytes,
    manifest_payload: bytes,
) -> dict[str, Any]:
    plan = CampaignPlan.from_dict(_json(plan_payload, role="campaign_plan"))
    expected_manifest = _json(manifest_payload, role="campaign_manifest")
    with tempfile.TemporaryDirectory(prefix="aura-resident-sft-journal-replay-") as raw:
        root = Path(raw)
        journal_path = root / "campaign.journal.jsonl"
        _write_exclusive(journal_path, journal_payload)
        with CampaignJournal(journal_path, plan) as journal:
            resume = journal.resume()
            if resume.committed_cell_ids != plan.cell_ids or resume.runnable_cell_ids:
                _fail("resident_sft_materialize_controller_journal_incomplete")
            replayed = journal.finalize(root / "campaign-manifest.json")
    if replayed != expected_manifest:
        _fail("resident_sft_materialize_controller_manifest_mismatch")
    return cast(dict[str, Any], replayed)


def _verify_controller_config(
    config: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    body = dict(config)
    claimed = body.pop("config_sha256", None)
    source = config.get("source")
    if (
        claimed != sha256_json(body)
        or config.get("campaign_id") != authority["campaign_id"]
        or config.get("profile") != "full"
        or config.get("authority", {}).get("semantic_sha256") != authority["authority_sha256"]
        or config.get("plan", {}).get("semantic_sha256") != plan.get("plan_sha256")
        or not isinstance(source, Mapping)
        or source.get("commit") != source.get("origin_main")
    ):
        _fail("resident_sft_materialize_controller_config_invalid")


def _verify_status(
    status: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    state: Mapping[str, Any],
) -> None:
    body = dict(status)
    claimed = body.pop("status_sha256", None)
    if (
        claimed != sha256_json(body)
        or status.get("schema") != "aura.resident_recurrent_sft_bootstrap_status.v1"
        or status.get("authority_sha256") != authority["authority_sha256"]
        or status.get("step") != state["step"]
        or status.get("max_steps") != authority["trainer"]["max_steps"]
        or status.get("latest_invocation") != state["invocation_count"]
        or status.get("terminal") is not True
        or status.get("halt_reason") != "max_steps"
    ):
        _fail("resident_sft_materialize_terminal_status_invalid")


def _verify_invocations(
    training_root: Path,
    *,
    authority: Mapping[str, Any],
    state: Mapping[str, Any],
    base_identity: Mapping[str, Any],
) -> list[tuple[int, Path, bytes, dict[str, Any]]]:
    observed: list[tuple[int, Path, bytes, dict[str, Any]]] = []
    for path in sorted(training_root.glob("invocation-*.json")):
        match = _INVOCATION_NAME.fullmatch(path.name)
        if match is None:
            _fail("resident_sft_materialize_invocation_name_invalid")
        ordinal = int(match.group(1))
        payload = _stable_bytes(path, role=f"invocation_{ordinal:04d}")
        receipt = _json(payload, role=f"invocation_{ordinal:04d}")
        body = dict(receipt)
        claimed = body.pop("receipt_sha256", None)
        if (
            claimed != sha256_json(body)
            or receipt.get("schema") != "aura.resident_recurrent_sft_bootstrap_invocation.v1"
            or receipt.get("authority_sha256") != authority["authority_sha256"]
            or receipt.get("campaign_id") != authority["campaign_id"]
            or receipt.get("campaign_scope") != "full_bootstrap"
            or receipt.get("invocation_count") != ordinal
            or receipt.get("base_checkpoint_before") != dict(base_identity)
            or receipt.get("base_checkpoint_after") != dict(base_identity)
            or receipt.get("base_checkpoint_immutable") is not True
        ):
            _fail("resident_sft_materialize_invocation_invalid")
        observed.append((ordinal, path, payload, receipt))
    expected_count = state["invocation_count"]
    if [row[0] for row in observed] != list(range(1, expected_count + 1)):
        _fail("resident_sft_materialize_invocation_sequence_incomplete")
    terminal = observed[-1][3]
    if (
        terminal.get("step") != state["step"]
        or terminal.get("checkpoint_sequence") != state["checkpoint_sequence"]
        or terminal.get("checkpoint_complete_sha256") is None
        or terminal.get("terminal") is not True
        or terminal.get("halt_reason") != "max_steps"
        or terminal.get("bootstrap_complete") is not True
        or terminal.get("claim_state", {}).get("resident_sft_complete") is not True
    ):
        _fail("resident_sft_materialize_terminal_invocation_incomplete")
    return observed


def _adapter_value_identity(path: Path, *, role: str) -> tuple[str, str]:
    try:
        import mlx.core as mx

        tensors = mx.load(str(path))
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ResidentRecurrentSFTMaterializationError(
            f"resident_sft_materialize_{role}_tensor_load_failed"
        ) from exc
    if not isinstance(tensors, dict) or not tensors:
        _fail(f"resident_sft_materialize_{role}_tensor_container_invalid")
    return adapter_tensor_fingerprint(tensors), adapter_topology_sha256(tensors)


def _verify_checkpoint_chain(
    training_root: Path,
    *,
    authority: Mapping[str, Any],
    final_complete_sha256: str,
) -> dict[str, Any]:
    checkpoints = _directory(training_root / "checkpoints", role="checkpoint_root")
    expected_bindings = authority_state_bindings(authority)
    generations: list[tuple[int, int, Path]] = []
    for path in checkpoints.iterdir():
        resolved = _directory(path, role="checkpoint_generation")
        match = _CHECKPOINT_NAME.fullmatch(resolved.name)
        if match is None:
            _fail("resident_sft_materialize_checkpoint_generation_name_invalid")
        generations.append((int(match.group(1)), int(match.group(2)), resolved))
    generations.sort(key=lambda row: row[0])
    max_steps = authority["trainer"]["max_steps"]
    if (
        len(generations) != max_steps + 1
        or [sequence for sequence, _step, _path in generations] != list(range(1, max_steps + 2))
        or [step for _sequence, step, _path in generations] != list(range(0, max_steps + 1))
    ):
        _fail("resident_sft_materialize_checkpoint_chain_incomplete")

    prior_state: dict[str, Any] | None = None
    prior_adapter_value_sha256: str | None = None
    final: dict[str, Any] | None = None
    evaluate_every = authority["trainer"]["evaluate_every"]
    for sequence, step, generation in generations:
        complete_payload = _stable_bytes(
            generation / "complete.json",
            role=f"checkpoint_{sequence:08d}_complete",
            maximum=MAX_JSON_BYTES,
        )
        complete = _json(complete_payload, role=f"checkpoint_{sequence:08d}_complete")
        if (
            set(complete)
            != {
                "schema",
                "checkpoint_id",
                "created_at_unix",
                "state",
                "adapter",
                "optimizer",
            }
            or complete.get("checkpoint_id") != generation.name
        ):
            _fail("resident_sft_materialize_checkpoint_complete_schema_invalid")
        raw_state = complete.get("state")
        if not isinstance(raw_state, Mapping):
            _fail("resident_sft_materialize_checkpoint_state_invalid")
        state = validate_checkpoint_state(raw_state)
        if (
            state["checkpoint_sequence"] != sequence
            or state["step"] != step
            or any(state[role] != digest for role, digest in expected_bindings.items())
            or state["terminal"] is not (sequence == len(generations))
            or state["halt_reason"] != ("max_steps" if sequence == len(generations) else None)
        ):
            _fail("resident_sft_materialize_checkpoint_transition_invalid")
        for role in ("adapter", "optimizer"):
            binding = complete.get(role)
            if (
                not isinstance(binding, Mapping)
                or set(binding)
                != {
                    "path",
                    "sha256",
                    "size_bytes",
                }
                or binding.get("path") != f"{role}.safetensors"
            ):
                _fail(f"resident_sft_materialize_checkpoint_{role}_binding_invalid")
            payload = _stable_bytes(
                generation / f"{role}.safetensors",
                role=f"checkpoint_{sequence:08d}_{role}",
            )
            if binding.get("sha256") != sha256_bytes(payload) or binding.get("size_bytes") != len(
                payload
            ):
                _fail(f"resident_sft_materialize_checkpoint_{role}_binding_mismatch")

        adapter_value_sha256, observed_topology_sha256 = _adapter_value_identity(
            generation / "adapter.safetensors",
            role=f"checkpoint_{sequence:08d}_adapter",
        )
        loss_trail = state["loss_trail"]
        validation_trail = state["validation_trail"]
        expected_validation_steps = list(range(evaluate_every, step + 1, evaluate_every))
        if state["terminal"] and step % evaluate_every:
            expected_validation_steps.append(step)
        if (
            len(loss_trail) != step
            or [entry.get("step") for entry in loss_trail] != list(range(1, step + 1))
            or [entry.get("step") for entry in validation_trail] != expected_validation_steps
            or state["pending_losses"]
            or observed_topology_sha256 != state["adapter_topology_sha256"]
        ):
            _fail("resident_sft_materialize_checkpoint_trail_invalid")
        if step == 0:
            if adapter_value_sha256 != state["initial_adapter_sha256"]:
                _fail("resident_sft_materialize_initial_adapter_identity_mismatch")
        else:
            latest_loss = loss_trail[-1]
            if (
                latest_loss.get("adapter_after_sha256") != adapter_value_sha256
                or latest_loss.get("adapter_before_sha256") != prior_adapter_value_sha256
            ):
                _fail("resident_sft_materialize_adapter_value_continuity_invalid")
        if prior_state is not None:
            if (
                state["loss_trail"][:-1] != prior_state["loss_trail"]
                or state["validation_trail"][: len(prior_state["validation_trail"])]
                != prior_state["validation_trail"]
                or state["baseline_validation"] != prior_state["baseline_validation"]
                or state["initial_adapter_sha256"] != prior_state["initial_adapter_sha256"]
                or state["adapter_topology_sha256"] != prior_state["adapter_topology_sha256"]
                or state["sample_history_sha256"] == prior_state["sample_history_sha256"]
                or state["invocation_count"] < prior_state["invocation_count"]
                or state["invocation_count"] > prior_state["invocation_count"] + 1
            ):
                _fail("resident_sft_materialize_checkpoint_prefix_drift")
        prior_state = state
        prior_adapter_value_sha256 = adapter_value_sha256
        final = {
            "sequence": sequence,
            "step": step,
            "checkpoint_id": generation.name,
            "complete_sha256": sha256_bytes(complete_payload),
            "adapter_value_sha256": adapter_value_sha256,
        }
    if final is None or final["complete_sha256"] != final_complete_sha256:
        _fail("resident_sft_materialize_checkpoint_final_digest_mismatch")

    pointer_payload = _stable_bytes(training_root / "latest.json", role="checkpoint_pointer")
    pointer = _json(pointer_payload, role="checkpoint_pointer")
    if (
        pointer.get("checkpoint_sequence") != final["sequence"]
        or PurePosixPath(str(pointer.get("checkpoint"))).name != final["checkpoint_id"]
        or pointer.get("complete_sha256") != final["complete_sha256"]
    ):
        _fail("resident_sft_materialize_checkpoint_pointer_not_final")
    return {
        "generation_count": len(generations),
        "terminal_sequence": final["sequence"],
        "terminal_step": final["step"],
        "terminal_complete_sha256": final["complete_sha256"],
        "terminal_adapter_value_sha256": final["adapter_value_sha256"],
    }


def _lora_metadata(
    *,
    tensors: Sequence[TensorIdentity],
    authority: Mapping[str, Any],
    spec: RLCExecutionSpec,
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    trainer = authority["trainer"]
    layer_count = model_config.get("num_hidden_layers")
    if type(layer_count) is not int or layer_count < 1:
        _fail("resident_sft_materialize_model_layer_count_invalid")
    prelude_end = max(1, int(layer_count * spec.prelude_frac))
    coda_start = min(
        layer_count - 1,
        layer_count - max(1, int(layer_count * spec.coda_frac)),
    )
    lora_layers = trainer["lora_layers"]
    targets = list(trainer["lora_targets"])
    indices = list(range(coda_start - lora_layers, coda_start))
    if len(indices) != lora_layers or indices[0] < prelude_end:
        _fail("resident_sft_materialize_recurrent_layer_window_invalid")
    expected_projections = [
        f"model.layers.{index}.self_attn.{target}" for index in indices for target in targets
    ]
    actual_keys = {tensor.key for tensor in tensors}
    has_depth_bank = any(".depth_a." in key or ".depth_b." in key for key in actual_keys)
    depth_bank_size = max(authority["dataset"]["depths"]) if has_depth_bank else 0
    expected_keys = {
        f"{projection}.{suffix}"
        for projection in expected_projections
        for suffix in ("lora_a", "lora_b")
    }
    expected_keys.update(
        f"{projection}.{suffix}.{depth}"
        for projection in expected_projections
        for suffix in ("depth_a", "depth_b")
        for depth in range(depth_bank_size)
    )
    expected_tensor_count = len(expected_projections) * (2 + 2 * depth_bank_size)
    if len(tensors) != expected_tensor_count or actual_keys != expected_keys:
        _fail("resident_sft_materialize_exact_lora_topology_mismatch")
    rank = trainer["lora_rank"]
    for tensor in tensors:
        if len(tensor.shape) != 2 or rank not in tensor.shape:
            _fail("resident_sft_materialize_lora_rank_shape_mismatch")
    trainable = sum(
        dimension for tensor in tensors for dimension in [tensor.shape[0] * tensor.shape[1]]
    )
    metadata = {
        "rank": rank,
        "scale": float(trainer["lora_scale"]),
        "dropout": float(trainer["lora_dropout"]),
        "layers": lora_layers,
        "targets": targets,
        "wrapped_projections": len(expected_projections),
        "projection_paths": expected_projections,
        "trainable_params": trainable,
    }
    if depth_bank_size:
        metadata.update(
            {
                "conditioning_schema": "aura.depth_conditioned_lora.v1",
                "depth_bank_size": depth_bank_size,
            }
        )
    return metadata


def _package_artifacts(root: Path, manifest: Mapping[str, Any]) -> dict[str, bytes]:
    artifacts: dict[str, bytes] = {}
    for role, binding in declared_bindings(manifest):
        path = root.joinpath(*PurePosixPath(binding["path"]).parts)
        artifacts[binding["path"]] = _stable_bytes(path, role=f"package_{role}")
    completion = root / "training_completion.json"
    if completion.exists():
        artifacts["training_completion.json"] = _stable_bytes(
            completion, role="package_training_completion"
        )
    return artifacts


def _validate_completion(
    payload: bytes,
    *,
    manifest_payload: bytes,
    authority: Mapping[str, Any],
    state: Mapping[str, Any],
    adapter_sha256: str,
    checkpoint_complete_sha256: str,
) -> dict[str, Any]:
    completion = _json(payload, role="training_completion")
    expected = {
        "schema": PACKAGE_COMPLETION_SCHEMA,
        "complete": True,
        "halt_reason": "max_steps",
        "step": state["step"],
        "adapter_sha256": adapter_sha256,
        "checkpoint_complete_sha256": checkpoint_complete_sha256,
        "authority_sha256": authority["authority_sha256"],
        "manifest_sha256": sha256_bytes(manifest_payload),
    }
    if completion != expected or payload != canonical_json_bytes(completion):
        _fail("resident_sft_materialize_package_completion_invalid")
    return completion


def _training_admission(
    *,
    identity_receipt: Mapping[str, Any],
    authority: Mapping[str, Any],
    state: Mapping[str, Any],
    adapter_sha256: str,
    checkpoint_complete_sha256: str,
) -> dict[str, Any]:
    mechanics_identity = {
        **dict(identity_receipt),
        "complete": True,
        "load_eligible": True,
        "training_scope": "resident_recurrent_sft",
    }
    body = {
        "schema": TRAINING_ADMISSION_SCHEMA,
        "decision": "admit_to_freeze_and_mechanics",
        "claim_scope": "resident_recurrent_sft_training_mechanics_admission_only",
        "training_state": {
            "complete": True,
            "terminal": True,
            "halt_reason": "max_steps",
            "step": state["step"],
            "checkpoint_sequence": state["checkpoint_sequence"],
            "invocation_count": state["invocation_count"],
            "authority_sha256": authority["authority_sha256"],
            "adapter_sha256": adapter_sha256,
            "checkpoint_complete_sha256": checkpoint_complete_sha256,
        },
        "identity_receipt": mechanics_identity,
        "claim_flags": {
            "reasoning_gain_proven": False,
            "causal_gain_proven": False,
            "frontier_level_proven": False,
            "grpo_admission": False,
            "promotion_allowed": False,
            "wow_signal": False,
        },
    }
    return {**body, "admission_sha256": sha256_json(body)}


def materialize_resident_recurrent_sft_adapter(
    *,
    campaign_root: Path,
    source_capsule_root: Path,
    destination: Path,
    adapter_id: str,
    model_path: Path,
) -> dict[str, Any]:
    if not isinstance(adapter_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:-]{0,159}", adapter_id
    ):
        _fail("resident_sft_materialize_adapter_id_invalid")
    capsule = _directory(source_capsule_root, role="source_capsule")
    campaign = _directory(campaign_root, role="campaign_root")
    try:
        campaign.relative_to(capsule)
    except ValueError as exc:
        raise ResidentRecurrentSFTMaterializationError(
            "resident_sft_materialize_campaign_outside_capsule"
        ) from exc
    model = _directory(model_path, role="model")
    destination_lexical = destination.expanduser().absolute()
    parent = _directory(destination_lexical.parent, role="destination_parent")
    admission_path = parent / f"{destination_lexical.name}.training_admission.json"
    if destination_lexical.exists() or destination_lexical.is_symlink():
        _fail("resident_sft_materialize_destination_exists")
    if admission_path.exists() or admission_path.is_symlink():
        _fail("resident_sft_materialize_admission_destination_exists")

    authority_path = campaign / "inputs" / "authority.json"
    authority_payload = _stable_bytes(authority_path, role="authority", maximum=MAX_JSON_BYTES)
    authority = validate_authority(
        _json(authority_payload, role="authority"),
        allow_expired_resume=True,
    )
    if authority["campaign_scope"] != "full_bootstrap":
        _fail("resident_sft_materialize_campaign_not_full_bootstrap")
    expected_model = _contained(
        capsule,
        authority["model"]["path"],
        role="authority_model",
        directory=True,
    )
    if model != expected_model:
        _fail("resident_sft_materialize_model_path_mismatch")
    base_identity = full_weight_checkpoint_identity(model)
    behavior_identity = model_behavior_bundle_identity(model)
    personality_identity = absent_personality_identity()
    evaluation_runtime = runtime_environment_identity()
    if (
        base_identity != authority["model"]["base_checkpoint"]
        or behavior_identity != authority["model"]["behavior_bundle"]
        or personality_identity != authority["model"]["personality_bundle"]
        or _normalize_training_runtime(authority["runtime"]) != evaluation_runtime
    ):
        _fail("resident_sft_materialize_effective_stack_mismatch")

    train_path = _contained(
        capsule, authority["dataset_artifacts"]["train"]["path"], role="train_dataset"
    )
    validation_path = _contained(
        capsule,
        authority["dataset_artifacts"]["validation"]["path"],
        role="validation_dataset",
    )
    train_payload = _stable_bytes(train_path, role="train_dataset")
    validation_payload = _stable_bytes(validation_path, role="validation_dataset")
    source_paths = {
        role: _contained(capsule, authority["sources"][role]["path"], role=f"source_{role}")
        for role in authority["sources"]
    }
    source_payloads = {
        role: _stable_bytes(path, role=f"source_{role}") for role, path in source_paths.items()
    }
    authorize_bound_artifacts(
        authority,
        train_payload=train_payload,
        validation_payload=validation_payload,
        source_payloads=source_payloads,
        expected_authority_sha256=authority["authority_sha256"],
    )
    execution_path = _contained(capsule, authority["execution_spec"]["path"], role="execution_spec")
    execution_payload = _stable_bytes(execution_path, role="execution_spec")
    spec = RLCExecutionSpec.from_dict(_json(execution_payload, role="execution_spec"))
    if (
        sha256_bytes(execution_payload) != authority["execution_spec"]["sha256"]
        or len(execution_payload) != authority["execution_spec"]["size_bytes"]
        or spec.sha256 != authority["execution_spec"]["semantic_sha256"]
    ):
        _fail("resident_sft_materialize_execution_spec_mismatch")
    trust_path = _contained(capsule, authority["trust_policy"]["path"], role="trust_policy")
    trust_payload = _stable_bytes(trust_path, role="trust_policy")
    if (
        sha256_bytes(trust_payload) != authority["trust_policy"]["sha256"]
        or len(trust_payload) != authority["trust_policy"]["size_bytes"]
    ):
        _fail("resident_sft_materialize_trust_policy_mismatch")

    training = campaign / "training"
    inspected = inspect_checkpoint(
        training,
        expected_bindings=authority_state_bindings(authority),
    )
    state = inspected.state
    if (
        state["terminal"] is not True
        or state["halt_reason"] != "max_steps"
        or state["step"] != authority["trainer"]["max_steps"]
    ):
        _fail("resident_sft_materialize_terminal_checkpoint_incomplete")
    chain_receipt = _verify_checkpoint_chain(
        training,
        authority=authority,
        final_complete_sha256=inspected.complete_sha256,
    )
    pointer_path = training / "latest.json"
    complete_path = inspected.checkpoint_dir / "complete.json"
    adapter_path = inspected.checkpoint_dir / "adapter.safetensors"
    optimizer_path = inspected.checkpoint_dir / "optimizer.safetensors"
    pointer_payload = _stable_bytes(pointer_path, role="checkpoint_pointer")
    complete_payload = _stable_bytes(complete_path, role="checkpoint_complete")
    adapter_payload = _stable_bytes(adapter_path, role="adapter")
    optimizer_payload = _stable_bytes(optimizer_path, role="optimizer")

    invocations = _verify_invocations(
        training,
        authority=authority,
        state=state,
        base_identity=base_identity,
    )
    if invocations[-1][3]["checkpoint_complete_sha256"] != inspected.complete_sha256:
        _fail("resident_sft_materialize_terminal_invocation_checkpoint_mismatch")
    status_payload = _stable_bytes(training / "status.json", role="terminal_status")
    _verify_status(_json(status_payload, role="terminal_status"), authority=authority, state=state)

    controller = campaign / "controller"
    controller_config_payload = _stable_bytes(
        campaign / "controller-config.json", role="controller_config"
    )
    plan_payload = _stable_bytes(campaign / "inputs" / "campaign-plan.json", role="campaign_plan")
    plan = _json(plan_payload, role="campaign_plan")
    _verify_controller_config(
        _json(controller_config_payload, role="controller_config"),
        authority=authority,
        plan=plan,
    )
    journal_payload = _stable_bytes(controller / "campaign.journal.jsonl", role="journal")
    campaign_manifest_payload = _stable_bytes(
        controller / "campaign-manifest.json", role="campaign_manifest"
    )
    campaign_manifest = _verify_controller_journal(
        plan_payload=plan_payload,
        journal_payload=journal_payload,
        manifest_payload=campaign_manifest_payload,
    )
    completion_payload = _stable_bytes(
        controller / "completion-receipt.json", role="controller_completion"
    )
    controller_completion = _json(completion_payload, role="controller_completion")
    if (
        controller_completion.get("journal_manifest_sha256") != campaign_manifest["manifest_sha256"]
        or controller_completion.get("plan_sha256") != plan["plan_sha256"]
    ):
        _fail("resident_sft_materialize_controller_completion_incomplete")

    tensors = inspect_mlx_tensor_metadata(adapter_path)
    model_config_payload = _stable_bytes(model / "config.json", role="model_config")
    lora = _lora_metadata(
        tensors=tensors,
        authority=authority,
        spec=spec,
        model_config=_json(model_config_payload, role="model_config"),
    )
    if (
        topology_sha256([tensor.to_dict() for tensor in tensors])
        != state["adapter_topology_sha256"]
    ):
        _fail("resident_sft_materialize_adapter_topology_digest_mismatch")

    lock_path = parent / f".{destination_lexical.name}.materialize.lock"
    staging = parent / f".{destination_lexical.name}.staging-{uuid.uuid4().hex}"
    admission_staging = parent / f".{admission_path.name}.staging-{uuid.uuid4().hex}"
    with interprocess_file_lock(lock_path):
        if destination_lexical.exists() or destination_lexical.is_symlink():
            _fail("resident_sft_materialize_destination_exists")
        if admission_path.exists() or admission_path.is_symlink():
            _fail("resident_sft_materialize_admission_destination_exists")
        try:
            staging.mkdir(mode=0o700)
            bindings: dict[str, Any] = {}
            bindings["authority"] = _copy_into(
                staging, "evidence/authority.json", authority_payload
            )
            bindings["train_dataset"] = _copy_into(
                staging, "evidence/datasets/train.json", train_payload
            )
            bindings["validation_dataset"] = _copy_into(
                staging, "evidence/datasets/validation.json", validation_payload
            )
            bindings["execution_spec"] = _copy_into(
                staging, "execution_spec.json", execution_payload
            )
            bindings["trust_policy"] = _copy_into(
                staging, "evidence/trust-policy.json", trust_payload
            )
            bindings["checkpoint_pointer"] = _copy_into(
                staging, "evidence/checkpoint/latest.json", pointer_payload
            )
            bindings["checkpoint_complete"] = _copy_into(
                staging, "evidence/checkpoint/complete.json", complete_payload
            )
            # Preserve the checkpoint leaf name because the signed complete
            # record binds it in addition to the content digest.
            bindings["adapter"] = _copy_into(staging, "adapter.safetensors", adapter_payload)
            bindings["optimizer"] = _copy_into(
                staging, "evidence/checkpoint/optimizer.safetensors", optimizer_payload
            )
            bindings["controller_config"] = _copy_into(
                staging, "evidence/controller/config.json", controller_config_payload
            )
            bindings["campaign_plan"] = _copy_into(
                staging, "evidence/controller/campaign-plan.json", plan_payload
            )
            bindings["campaign_journal"] = _copy_into(
                staging, "evidence/controller/campaign.journal.jsonl", journal_payload
            )
            bindings["campaign_manifest"] = _copy_into(
                staging,
                "evidence/controller/campaign-manifest.json",
                campaign_manifest_payload,
            )
            bindings["controller_completion"] = _copy_into(
                staging, "evidence/controller/completion-receipt.json", completion_payload
            )
            bindings["terminal_status"] = _copy_into(
                staging, "evidence/training/status.json", status_payload
            )
            chain_payload = canonical_json_bytes(
                {
                    "schema": "aura.resident_recurrent_sft_checkpoint_chain_replay.v1",
                    "authority_sha256": authority["authority_sha256"],
                    **chain_receipt,
                }
            )
            bindings["checkpoint_chain_replay"] = _copy_into(
                staging, "evidence/checkpoint/chain-replay.json", chain_payload
            )
            for ordinal, _path, payload, _receipt in invocations[:-1]:
                bindings[f"invocation_{ordinal:04d}"] = _copy_into(
                    staging,
                    f"evidence/invocations/invocation-{ordinal:04d}.json",
                    payload,
                )
            terminal_ordinal, _terminal_path, terminal_payload, _terminal = invocations[-1]
            bindings["terminal_invocation"] = _copy_into(
                staging,
                f"evidence/invocations/invocation-{terminal_ordinal:04d}.json",
                terminal_payload,
            )
            source_bindings: dict[str, Any] = {}
            for role in sorted(authority["sources"]):
                source_bindings[role] = _copy_into(
                    staging,
                    f"evidence/sources/{role}.snapshot",
                    source_payloads[role],
                )
            bindings["source_snapshots"] = source_bindings
            loader_config = {
                "schema": LOADER_CONFIG_SCHEMA,
                "fine_tune_type": "recurrence_scoped_lora",
                "loader": "aura_custom_loader_required",
                "adapter_id": adapter_id,
                "base_checkpoint_fingerprint": base_identity["fingerprint"],
                "execution_spec_sha256": spec.sha256,
                "lora": lora,
            }
            loader_payload = canonical_json_bytes(loader_config)
            bindings["loader_config"] = _copy_into(staging, "adapter_config.json", loader_payload)
            manifest = {
                "schema": (
                    MANIFEST_SCHEMA if lora.get("depth_bank_size") else LEGACY_MANIFEST_SCHEMA
                ),
                "adapter_id": adapter_id,
                "training_protocol": authority["training_authority"],
                "base_checkpoint": base_identity,
                "model_behavior_bundle": behavior_identity,
                "personality_adapter": personality_identity,
                "training_runtime": authority["runtime"],
                "bindings": bindings,
                "lora": lora,
                "tensors": [tensor.to_dict() for tensor in tensors],
                "claim_boundary": {
                    "training_objective_learned": True,
                    "reasoning_gain_proven": False,
                    "causal_gain_proven": False,
                    "frontier_level_proven": False,
                    "promotion_allowed": False,
                },
            }
            manifest_payload = canonical_json_bytes(manifest)
            _write_exclusive(staging / "recurrence_adapter_manifest.json", manifest_payload)
            package_completion = {
                "schema": PACKAGE_COMPLETION_SCHEMA,
                "complete": True,
                "halt_reason": "max_steps",
                "step": state["step"],
                "adapter_sha256": inspected.adapter_binding["sha256"],
                "checkpoint_complete_sha256": inspected.complete_sha256,
                "authority_sha256": authority["authority_sha256"],
                "manifest_sha256": sha256_bytes(manifest_payload),
            }
            package_completion_payload = canonical_json_bytes(package_completion)
            _write_exclusive(staging / "training_completion.json", package_completion_payload)
            _validate_completion(
                package_completion_payload,
                manifest_payload=manifest_payload,
                authority=authority,
                state=state,
                adapter_sha256=inspected.adapter_binding["sha256"],
                checkpoint_complete_sha256=inspected.complete_sha256,
            )
            identity_receipt = validate_resident_recurrent_sft_adapter_identity(
                manifest_payload,
                adapter_id=adapter_id,
                actual_base_checkpoint=base_identity,
                actual_model_behavior_bundle=behavior_identity,
                actual_personality_adapter=personality_identity,
                actual_runtime_environment=evaluation_runtime,
                artifacts=_package_artifacts(staging, manifest),
                tensor_metadata=inspect_mlx_tensor_metadata(staging / "adapter.safetensors"),
            )
            _write_exclusive(
                staging / "identity_receipt.json",
                canonical_json_bytes(identity_receipt),
            )
            admission = _training_admission(
                identity_receipt=identity_receipt,
                authority=authority,
                state=state,
                adapter_sha256=inspected.adapter_binding["sha256"],
                checkpoint_complete_sha256=inspected.complete_sha256,
            )
            admission_payload = canonical_json_bytes(admission)
            _write_exclusive(admission_staging, admission_payload)
            if _json(admission_payload, role="training_admission") != admission or admission[
                "admission_sha256"
            ] != sha256_json(
                {key: value for key, value in admission.items() if key != "admission_sha256"}
            ):
                _fail("resident_sft_materialize_training_admission_invalid")
            for path in sorted(staging.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                if path.is_file():
                    path.chmod(0o400)
                elif path.is_dir():
                    path.chmod(0o500)
            staging.chmod(0o500)
            admission_staging.chmod(0o400)
            os.rename(staging, destination_lexical)
            try:
                os.rename(admission_staging, admission_path)
            except BaseException:
                for path in destination_lexical.rglob("*"):
                    try:
                        path.chmod(0o700 if path.is_dir() else 0o600)
                    except OSError:
                        pass
                destination_lexical.chmod(0o700)
                shutil.rmtree(destination_lexical, ignore_errors=True)
                raise
            parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except BaseException:
            if staging.exists() and not staging.is_symlink():
                for path in staging.rglob("*"):
                    try:
                        path.chmod(0o700 if path.is_dir() else 0o600)
                    except OSError:
                        pass
                try:
                    staging.chmod(0o700)
                except OSError:
                    pass
                shutil.rmtree(staging, ignore_errors=True)
            if admission_staging.exists() and not admission_staging.is_symlink():
                try:
                    admission_staging.chmod(0o600)
                    admission_staging.unlink()
                except OSError:
                    pass
            raise
    return {
        "destination": str(destination_lexical),
        "adapter_id": adapter_id,
        "manifest_sha256": sha256_bytes(manifest_payload),
        "adapter_sha256": inspected.adapter_binding["sha256"],
        "checkpoint_complete_sha256": inspected.complete_sha256,
        "authority_sha256": authority["authority_sha256"],
        "terminal_step": state["step"],
        "checkpoint_generation_count": chain_receipt["generation_count"],
        "training_runtime_identity_sha256": authority["runtime"]["identity_sha256"],
        "evaluation_runtime_identity_sha256": evaluation_runtime["identity_sha256"],
        "training_admission_path": str(admission_path),
        "identity_receipt": identity_receipt,
        "training_completion": package_completion,
        "training_admission": admission,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--source-capsule-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--adapter-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = materialize_resident_recurrent_sft_adapter(
            campaign_root=args.campaign_root,
            source_capsule_root=args.source_capsule_root,
            destination=args.destination,
            adapter_id=args.adapter_id,
            model_path=args.model,
        )
    except (OSError, TypeError, ValueError, ResidentRecurrentSFTMaterializationError) as exc:
        code = getattr(exc, "code", type(exc).__name__)
        print(f"materialization failed: {code}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
