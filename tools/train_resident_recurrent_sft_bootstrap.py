#!/usr/bin/env python3
"""Train or exactly resume the resident cached recurrent SFT bootstrap.

This command cannot fuse, promote, admit GRPO, or evaluate reasoning gain. It
only mutates the authority-bound recurrent adapter in memory and publishes
quarantined, crash-consistent checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import stat
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final, Never, cast

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec  # noqa: E402
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (  # noqa: E402
    full_weight_checkpoint_identity,
    model_behavior_bundle_identity,
)
from core.learning.recurrence_native_objective_v2 import (  # noqa: E402
    cached_supervised_live_path_loss,
    cached_supervised_live_path_value_and_grad,
)
from core.learning.recurrent_grpo import (  # noqa: E402
    attach_recurrent_policy_adapters,
)
from core.learning.recurrent_sft_execution import (  # noqa: E402
    adapter_tensor_dict,
    adapter_tensor_fingerprint,
    assert_adapter_tensor_topology,
)
from core.learning.resident_recurrent_sft_bootstrap_authority import (  # noqa: E402
    REQUIRED_SOURCE_ROLES,
    ResidentSFTBootstrapConfig,
    authorize_bound_artifacts,
    sha256_bytes,
    sha256_json,
    validate_authority,
)
from core.learning.resident_recurrent_sft_bootstrap_execution import (  # noqa: E402
    adapter_topology_sha256,
    advance_sample_history,
    family_depth_balanced_order,
    initial_sample_history,
    project_rows,
    sampling_receipt,
    validate_family_depth_balanced_order,
)
from core.learning.resident_recurrent_sft_bootstrap_state import (  # noqa: E402
    authority_state_bindings,
    inspect_checkpoint,
    load_checkpoint,
    order_sha256,
    save_checkpoint,
)
from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_bytes_if_absent,
    ensure_private_directory,
    interprocess_file_lock,
)
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402
from core.runtime.mlx_memory_guard import mlx_memory_envelope  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402
from core.runtime.secure_path_custody import (  # noqa: E402
    DirectoryCustody,
    SecurePathCustodyError,
)
from tools.resident_recurrent_sft_bootstrap_identity import (  # noqa: E402
    absent_personality_identity,
    resident_bootstrap_runtime_identity,
    resident_bootstrap_tokenizer_identity,
)

INVOCATION_SCHEMA: Final = "aura.resident_recurrent_sft_bootstrap_invocation.v1"
STATUS_SCHEMA: Final = "aura.resident_recurrent_sft_bootstrap_status.v1"
RESUME_POLICIES: Final = frozenset({"never", "auto", "required"})
MAX_DOCUMENT_BYTES: Final = 256 * 1024 * 1024
INTERRUPTED = False


class ResidentSFTBootstrapTrainingError(RuntimeError):
    """The source-bound resident bootstrap could not continue honestly."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise ResidentSFTBootstrapTrainingError(code)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError) as exc:
        raise ResidentSFTBootstrapTrainingError("resident_sft_trainer_noncanonical_value") from exc


def _read_json_bytes(payload: bytes, *, role: str) -> Any:
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ResidentSFTBootstrapTrainingError(
            f"resident_sft_trainer_{role}_json_invalid"
        ) from exc
    if _canonical_json_bytes(value) != payload:
        _fail(f"resident_sft_trainer_{role}_noncanonical")
    return value


def _resolve_repo_path(relative: Any, *, role: str, directory: bool = False) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        _fail(f"resident_sft_trainer_{role}_path_invalid")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or str(pure) != relative or ".." in pure.parts:
        _fail(f"resident_sft_trainer_{role}_path_invalid")
    lexical = REPO_ROOT / pure
    current = REPO_ROOT
    for component in pure.parts:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            _fail(f"resident_sft_trainer_{role}_path_unavailable")
        if stat.S_ISLNK(mode):
            _fail(f"resident_sft_trainer_{role}_symlink_forbidden")
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(REPO_ROOT.resolve(strict=True))
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ResidentSFTBootstrapTrainingError(
            f"resident_sft_trainer_{role}_path_unavailable"
        ) from exc
    if directory != resolved.is_dir() or (not directory and not resolved.is_file()):
        _fail(f"resident_sft_trainer_{role}_type_invalid")
    return resolved


def _resolve_repo_output_path(relative: Any, *, role: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        _fail(f"resident_sft_trainer_{role}_path_invalid")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or str(pure) != relative or ".." in pure.parts:
        _fail(f"resident_sft_trainer_{role}_path_invalid")
    current = REPO_ROOT
    for component in pure.parts:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            _fail(f"resident_sft_trainer_{role}_symlink_forbidden")
    resolved = (REPO_ROOT / pure).resolve(strict=False)
    try:
        resolved.relative_to(REPO_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ResidentSFTBootstrapTrainingError(
            f"resident_sft_trainer_{role}_path_unavailable"
        ) from exc
    return resolved


def _bound_bytes(binding: Mapping[str, Any], *, role: str) -> bytes:
    required = {"path", "sha256", "size_bytes"}
    if not isinstance(binding, Mapping) or not required.issubset(binding):
        _fail(f"resident_sft_trainer_{role}_binding_invalid")
    path = _resolve_repo_path(binding["path"], role=role)
    try:
        payload = read_stable_bytes(path, max_bytes=MAX_DOCUMENT_BYTES)
    except OSError as exc:
        raise ResidentSFTBootstrapTrainingError(f"resident_sft_trainer_{role}_unreadable") from exc
    if len(payload) != binding["size_bytes"] or sha256_bytes(payload) != binding["sha256"]:
        _fail(f"resident_sft_trainer_{role}_binding_drift")
    return cast(bytes, payload)


def _load_authority(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    try:
        payload = read_stable_bytes(
            path.expanduser().resolve(strict=True), max_bytes=MAX_DOCUMENT_BYTES
        )
    except OSError as exc:
        raise ResidentSFTBootstrapTrainingError(
            "resident_sft_trainer_authority_unreadable"
        ) from exc
    value = _read_json_bytes(payload, role="authority")
    if not isinstance(value, dict):
        _fail("resident_sft_trainer_authority_invalid")
    authority: dict[str, Any] = validate_authority(
        value,
        expected_authority_sha256=expected_sha256,
    )
    return authority


def _load_spec(authority: Mapping[str, Any]) -> RLCExecutionSpec:
    binding = authority["execution_spec"]
    payload = _bound_bytes(binding, role="execution_spec")
    value = _read_json_bytes(payload, role="execution_spec")
    if not isinstance(value, dict):
        _fail("resident_sft_trainer_execution_spec_invalid")
    try:
        spec = RLCExecutionSpec.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise ResidentSFTBootstrapTrainingError(
            "resident_sft_trainer_execution_spec_invalid"
        ) from exc
    if spec.sha256 != binding["semantic_sha256"]:
        _fail("resident_sft_trainer_execution_spec_semantic_drift")
    return spec


def _load_trust_policy(authority: Mapping[str, Any]) -> dict[str, Any]:
    binding = authority["trust_policy"]
    payload = _bound_bytes(binding, role="trust_policy")
    value = _read_json_bytes(payload, role="trust_policy")
    if not isinstance(value, dict) or sha256_json(value) != binding["semantic_sha256"]:
        _fail("resident_sft_trainer_trust_policy_semantic_drift")
    return value


def _load_dataset_and_sources(
    authority: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, bytes]]:
    train_payload = _bound_bytes(authority["dataset_artifacts"]["train"], role="train")
    validation_payload = _bound_bytes(
        authority["dataset_artifacts"]["validation"],
        role="validation",
    )
    train = _read_json_bytes(train_payload, role="train")
    validation = _read_json_bytes(validation_payload, role="validation")
    if (
        not isinstance(train, list)
        or not train
        or any(not isinstance(row, dict) for row in train)
        or not isinstance(validation, list)
        or not validation
        or any(not isinstance(row, dict) for row in validation)
    ):
        _fail("resident_sft_trainer_dataset_invalid")
    sources = {
        role: _bound_bytes(authority["sources"][role], role=f"source_{role}")
        for role in sorted(REQUIRED_SOURCE_ROLES)
    }
    authorize_bound_artifacts(
        authority,
        train_payload=train_payload,
        validation_payload=validation_payload,
        source_payloads=sources,
        expected_authority_sha256=authority["authority_sha256"],
    )
    return list(train), list(validation), sources


def _validation_summary(
    model: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    spec: RLCExecutionSpec,
    config: ResidentSFTBootstrapConfig,
) -> dict[str, Any]:
    order = family_depth_balanced_order(
        rows,
        seed=config.seed ^ 0x5A17,
        epoch=0,
    )
    if config.validation_examples > len(order):
        _fail("resident_sft_trainer_validation_budget_exceeds_split")
    selected = order[: config.validation_examples]
    records: list[dict[str, Any]] = []
    for index in selected:
        row = rows[index]
        result = cached_supervised_live_path_loss(
            model,
            row["prompt_tokens"],
            row["answer_tokens"],
            spec=spec,
            bridge_tokens=row["bridge_tokens"],
            branch_indices=config.branch_indices,
        )
        if (
            result.execution_spec_sha256 != spec.sha256
            or result.prompt_tokens_sha256 != sha256_json(row["prompt_tokens"])
            or result.answer_tokens_sha256 != sha256_json(row["answer_tokens"])
            or not math.isfinite(result.value)
        ):
            _fail("resident_sft_trainer_validation_objective_drift")
        records.append(
            {
                "example_id": row["example_id"],
                "loss": result.value,
                "branch_values": list(result.branch_values),
                "answer_token_count": result.answer_token_count,
            }
        )
    mean = sum(record["loss"] for record in records) / len(records)
    body = {
        "examples": len(records),
        "mean_loss": mean,
        "records": records,
        "execution_spec_sha256": spec.sha256,
        "branch_indices": list(config.branch_indices),
    }
    return {**body, "receipt_sha256": sha256_json(body)}


def _publish_sampling_receipt(
    out_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    order: Sequence[int],
    *,
    seed: int,
    epoch: int,
    custody: DirectoryCustody | None = None,
) -> dict[str, Any]:
    receipt = sampling_receipt(rows, order, seed=seed, epoch=epoch)
    payload = _canonical_json_bytes(receipt)
    directory = (
        custody.ensure_directory("sampling")
        if custody is not None
        else ensure_private_directory(out_dir / "sampling")
    )
    path = directory / f"epoch-{epoch:08d}.json"
    published = (
        custody.write_bytes_once(
            f"sampling/epoch-{epoch:08d}.json",
            payload,
            mode=0o600,
        )
        if custody is not None
        else atomic_write_bytes_if_absent(path, payload, mode=0o600)
    )
    if not published:
        try:
            observed = (
                custody.read_bytes(
                    f"sampling/epoch-{epoch:08d}.json",
                    max_bytes=MAX_DOCUMENT_BYTES,
                )
                if custody is not None
                else read_stable_bytes(path, max_bytes=MAX_DOCUMENT_BYTES)
            )
        except (OSError, SecurePathCustodyError) as exc:
            raise ResidentSFTBootstrapTrainingError(
                "resident_sft_trainer_sampling_receipt_unreadable"
            ) from exc
        if observed != payload:
            _fail("resident_sft_trainer_sampling_receipt_drift")
    return cast(dict[str, Any], receipt)


def _state_document(
    bindings: Mapping[str, str],
    *,
    sequence: int,
    step: int,
    epoch: int,
    cursor: int,
    order: list[int],
    config: ResidentSFTBootstrapConfig,
    train_count: int,
    validation_count: int,
    elapsed_s: float,
    invocation_count: int,
    sample_history_sha256: str,
    initial_adapter_sha256: str,
    adapter_topology_identity: str,
    loss_trail: list[dict[str, Any]],
    validation_trail: list[dict[str, Any]],
    baseline_validation: dict[str, Any],
    terminal: bool,
    halt_reason: str | None,
) -> dict[str, Any]:
    return {
        **dict(bindings),
        "checkpoint_sequence": sequence,
        "step": step,
        "optimizer_updates": step,
        "epoch": epoch,
        "cursor": cursor,
        "order": list(order),
        "order_sha256": order_sha256(order=order, seed=config.seed, epoch=epoch),
        "sampler": config.sampler,
        "seed": config.seed,
        "train_example_count": train_count,
        "validation_example_count": validation_count,
        "elapsed_training_s": round(float(elapsed_s), 6),
        "invocation_count": invocation_count,
        "sample_history_sha256": sample_history_sha256,
        "initial_adapter_sha256": initial_adapter_sha256,
        "adapter_topology_sha256": adapter_topology_identity,
        "loss_trail": list(loss_trail),
        "validation_trail": list(validation_trail),
        "pending_losses": [],
        "baseline_validation": dict(baseline_validation),
        "last_step_committed": True,
        "terminal": terminal,
        "halt_reason": halt_reason,
    }


def _publish_receipt(
    out_dir: Path,
    *,
    authority: Mapping[str, Any],
    state: Mapping[str, Any],
    checkpoint_sha256: str,
    base_before: Mapping[str, Any],
    base_after: Mapping[str, Any],
    halt_reason: str,
    required_end_step: int,
    custody: DirectoryCustody | None = None,
) -> dict[str, Any]:
    terminal_target_reached = bool(
        state["terminal"]
        and halt_reason == "max_steps"
        and state["step"] == authority["trainer"]["max_steps"]
    )
    campaign_scope = authority["campaign_scope"]
    bootstrap_complete = terminal_target_reached and campaign_scope == "full_bootstrap"
    canary_lifecycle_complete = terminal_target_reached and campaign_scope == "canary_lifecycle"
    body = {
        "schema": INVOCATION_SCHEMA,
        "authority_sha256": authority["authority_sha256"],
        "campaign_id": authority["campaign_id"],
        "campaign_scope": campaign_scope,
        "invocation_count": state["invocation_count"],
        "checkpoint_sequence": state["checkpoint_sequence"],
        "checkpoint_complete_sha256": checkpoint_sha256,
        "step": state["step"],
        "max_steps": authority["trainer"]["max_steps"],
        "required_end_step": required_end_step,
        "terminal": state["terminal"],
        "halt_reason": halt_reason,
        "canary_lifecycle_complete": canary_lifecycle_complete,
        "bootstrap_complete": bootstrap_complete,
        "base_checkpoint_before": dict(base_before),
        "base_checkpoint_after": dict(base_after),
        "base_checkpoint_immutable": dict(base_before) == dict(base_after),
        "claim_state": {
            "resident_sft_complete": bootstrap_complete,
            "causal_gain_proven": False,
            "reasoning_gain_proven": False,
            "frontier_level_proven": False,
            "grpo_admission": False,
            "promotion_allowed": False,
        },
    }
    receipt = {**body, "receipt_sha256": sha256_json(body)}
    if not receipt["base_checkpoint_immutable"]:
        _fail("resident_sft_trainer_base_checkpoint_changed")
    payload = _canonical_json_bytes(receipt)
    invocation_name = f"invocation-{state['invocation_count']:04d}.json"
    if custody is None:
        atomic_write_bytes(out_dir / invocation_name, payload, mode=0o600)
    else:
        custody.atomic_write_bytes(invocation_name, payload, mode=0o600)
    status_body = {
        "schema": STATUS_SCHEMA,
        "authority_sha256": authority["authority_sha256"],
        "latest_invocation": state["invocation_count"],
        "latest_receipt_sha256": receipt["receipt_sha256"],
        "step": state["step"],
        "max_steps": authority["trainer"]["max_steps"],
        "terminal": state["terminal"],
        "halt_reason": halt_reason,
    }
    status_payload = _canonical_json_bytes(
        {**status_body, "status_sha256": sha256_json(status_body)}
    )
    if custody is None:
        atomic_write_bytes(out_dir / "status.json", status_payload, mode=0o600)
    else:
        custody.atomic_write_bytes("status.json", status_payload, mode=0o600)
    return receipt


def _run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    authority_preview = _load_authority(
        args.authority,
        expected_sha256=args.expected_authority_sha256,
    )
    out_dir = _resolve_repo_output_path(authority_preview["artifact_root"], role="artifact_root")
    out_custody = DirectoryCustody.acquire(
        out_dir,
        expected_identity=authority_preview["artifact_root_identity"],
        private=True,
    )
    checkpoint_exists = out_custody.file_exists("latest.json")
    allow_expired_resume = checkpoint_exists and args.resume_policy in {"auto", "required"}
    authority = validate_authority(
        authority_preview,
        expected_authority_sha256=args.expected_authority_sha256,
        now=datetime.now(UTC),
        allow_expired_resume=allow_expired_resume,
    )
    if args.resume_policy == "never" and checkpoint_exists:
        _fail("resident_sft_trainer_checkpoint_exists_resume_required")
    if args.resume_policy == "required" and not checkpoint_exists:
        _fail("resident_sft_trainer_resume_checkpoint_missing")
    if out_custody.identity != authority["artifact_root_identity"]:
        _fail("resident_sft_trainer_artifact_root_identity_drift")
    out_dir = out_custody.path
    if out_dir != _resolve_repo_output_path(
        authority_preview["artifact_root"], role="artifact_root"
    ):
        _fail("resident_sft_trainer_artifact_root_identity_drift")
    config = ResidentSFTBootstrapConfig.from_dict(authority["trainer"])
    invocation_step_budget = (
        config.max_invocation_steps
        if getattr(args, "invocation_step_budget", None) is None
        else args.invocation_step_budget
    )
    if (
        type(invocation_step_budget) is not int
        or not 1 <= invocation_step_budget <= config.max_invocation_steps
    ):
        _fail("resident_sft_trainer_invocation_step_budget_invalid")
    required_end_step_arg = getattr(args, "required_end_step", None)
    if required_end_step_arg is not None and (
        type(required_end_step_arg) is not int or not 0 <= required_end_step_arg <= config.max_steps
    ):
        _fail("resident_sft_trainer_required_end_step_invalid")
    train_rows, validation_rows, sources = _load_dataset_and_sources(authority)
    spec = _load_spec(authority)
    _load_trust_policy(authority)
    if any(index >= len(spec.branch_roles) for index in config.branch_indices):
        _fail("resident_sft_trainer_branch_index_outside_spec")
    model_dir = _resolve_repo_path(authority["model"]["path"], role="model", directory=True)
    base_before = full_weight_checkpoint_identity(model_dir)
    behavior = model_behavior_bundle_identity(model_dir)
    personality = absent_personality_identity()
    runtime = resident_bootstrap_runtime_identity()
    if (
        base_before != authority["model"]["base_checkpoint"]
        or behavior != authority["model"]["behavior_bundle"]
        or personality != authority["model"]["personality_bundle"]
        or runtime != authority["runtime"]
    ):
        _fail("resident_sft_trainer_preload_identity_drift")
    bindings = authority_state_bindings(authority)
    out_custody.verify()

    trainer_lock = (
        out_custody.file_lock(".trainer.lock")
        if out_custody is not None
        else interprocess_file_lock(out_dir / ".trainer.lock")
    )
    with trainer_lock:
        import mlx.core as mx
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten, tree_unflatten
        from mlx_lm import load

        with (
            standalone_model_lane(
                owner_id=f"resident-sft:{authority['campaign_id']}",
                model_path=str(model_dir),
                purpose="training",
                preemptible=False,
                metadata={
                    "tool": "train_resident_recurrent_sft_bootstrap",
                    "authority_sha256": authority["authority_sha256"],
                },
            ),
            mlx_memory_envelope(
                fraction=config.memory_fraction,
                restore_limits_on_exit=False,
            ),
        ):
            model, tokenizer = load(str(model_dir))
            tokenizer_identity = resident_bootstrap_tokenizer_identity(model_dir, tokenizer)
            validate_authority(
                authority,
                expected_authority_sha256=args.expected_authority_sha256,
                observed_model_identity=base_before,
                observed_behavior_identity=behavior,
                observed_personality_identity=personality,
                observed_tokenizer_identity=tokenizer_identity,
                observed_execution_spec=authority["execution_spec"],
                observed_sources=authority["sources"],
                now=datetime.now(UTC),
                allow_expired_resume=allow_expired_resume,
            )
            projected_train = project_rows(
                train_rows,
                tokenizer=tokenizer,
                max_seq_length=config.max_seq_length,
            )
            projected_validation = project_rows(
                validation_rows,
                tokenizer=tokenizer,
                max_seq_length=config.max_seq_length,
            )
            attached = attach_recurrent_policy_adapters(
                model,
                spec,
                lora_rank=config.lora_rank,
                lora_layers=config.lora_layers,
                lora_targets=config.lora_targets,
                initialization_seed=config.lora_initialization_seed,
                lora_dropout=config.lora_dropout,
                lora_scale=config.lora_scale,
            )
            if len(attached) != config.lora_layers * len(config.lora_targets):
                _fail("resident_sft_trainer_adapter_attachment_count_drift")
            expected_adapter = adapter_tensor_dict(model)
            mx.eval(expected_adapter)
            initial_adapter_sha = adapter_tensor_fingerprint(expected_adapter)
            topology_sha = adapter_topology_sha256(expected_adapter)
            optimizer = optim.AdamW(
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            optimizer.init(model.trainable_parameters())

            step = 0
            epoch = 0
            cursor = 0
            order = family_depth_balanced_order(
                projected_train,
                seed=config.seed,
                epoch=epoch,
            )
            _publish_sampling_receipt(
                out_dir,
                projected_train,
                order,
                seed=config.seed,
                epoch=epoch,
                custody=out_custody,
            )
            out_custody.verify()
            sample_history = initial_sample_history()
            invocation_count = 1
            prior_elapsed = 0.0
            loss_trail: list[dict[str, Any]] = []
            validation_trail: list[dict[str, Any]] = []
            baseline: dict[str, Any]
            sequence = 0

            if checkpoint_exists:
                loaded = load_checkpoint(
                    out_dir,
                    expected_bindings=bindings,
                    custody=out_custody,
                )
                state = loaded.state
                assert_adapter_tensor_topology(expected_adapter, loaded.adapter_tensors)
                if (
                    state["initial_adapter_sha256"] != initial_adapter_sha
                    or state["adapter_topology_sha256"] != topology_sha
                ):
                    _fail("resident_sft_trainer_resume_adapter_identity_drift")
                loaded_adapter_sha = adapter_tensor_fingerprint(loaded.adapter_tensors)
                model.load_weights(list(loaded.adapter_tensors.items()), strict=False)
                optimizer.state = tree_unflatten(list(loaded.optimizer_tensors.items()))
                optimizer.init(model.trainable_parameters())
                mx.eval(model.trainable_parameters(), optimizer.state)
                if adapter_tensor_fingerprint(adapter_tensor_dict(model)) != loaded_adapter_sha:
                    _fail("resident_sft_trainer_loaded_adapter_drift")
                if state["terminal"]:
                    required_end_step = (
                        config.max_steps if required_end_step_arg is None else required_end_step_arg
                    )
                    if state["step"] != required_end_step:
                        _fail("resident_sft_trainer_required_end_step_overshot")
                    inspected = inspect_checkpoint(
                        out_dir,
                        expected_bindings=bindings,
                        custody=out_custody,
                    )
                    base_after = full_weight_checkpoint_identity(model_dir)
                    receipt = _publish_receipt(
                        out_dir,
                        authority=authority,
                        state=state,
                        checkpoint_sha256=inspected.complete_sha256,
                        base_before=base_before,
                        base_after=base_after,
                        halt_reason=state["halt_reason"],
                        required_end_step=required_end_step,
                        custody=out_custody,
                    )
                    out_custody.verify()
                    print(json.dumps(receipt, sort_keys=True), flush=True)
                    return 0
                step = state["step"]
                epoch = state["epoch"]
                cursor = state["cursor"]
                order = validate_family_depth_balanced_order(
                    projected_train,
                    state["order"],
                    seed=config.seed,
                    epoch=epoch,
                )
                _publish_sampling_receipt(
                    out_dir,
                    projected_train,
                    order,
                    seed=config.seed,
                    epoch=epoch,
                    custody=out_custody,
                )
                sample_history = state["sample_history_sha256"]
                invocation_count = state["invocation_count"] + 1
                prior_elapsed = state["elapsed_training_s"]
                loss_trail = list(state["loss_trail"])
                validation_trail = list(state["validation_trail"])
                baseline = dict(state["baseline_validation"])
                sequence = state["checkpoint_sequence"]
            else:
                baseline = _validation_summary(
                    model,
                    projected_validation,
                    spec=spec,
                    config=config,
                )
                initial_terminal = time.monotonic() - started >= config.max_minutes * 60.0
                initial_state = _state_document(
                    bindings,
                    sequence=1,
                    step=0,
                    epoch=0,
                    cursor=0,
                    order=order,
                    config=config,
                    train_count=len(projected_train),
                    validation_count=len(projected_validation),
                    elapsed_s=time.monotonic() - started,
                    invocation_count=invocation_count,
                    sample_history_sha256=sample_history,
                    initial_adapter_sha256=initial_adapter_sha,
                    adapter_topology_identity=topology_sha,
                    loss_trail=loss_trail,
                    validation_trail=validation_trail,
                    baseline_validation=baseline,
                    terminal=initial_terminal,
                    halt_reason="wall_clock" if initial_terminal else None,
                )
                save_checkpoint(
                    out_dir,
                    adapter_tensors=expected_adapter,
                    optimizer_tensors=dict(tree_flatten(optimizer.state)),
                    state=initial_state,
                    custody=out_custody,
                )
                out_custody.verify()
                sequence = 1

            invocation_started_step = step
            required_end_step = (
                min(config.max_steps, step + invocation_step_budget)
                if required_end_step_arg is None
                else required_end_step_arg
            )
            if step > required_end_step:
                _fail("resident_sft_trainer_required_end_step_overshot")

            def elapsed() -> float:
                return prior_elapsed + time.monotonic() - started

            halt_reason = "invocation_step_limit"
            while step < config.max_steps and step < required_end_step:
                if INTERRUPTED:
                    halt_reason = "interrupted"
                    break
                if elapsed() >= config.max_minutes * 60.0:
                    halt_reason = "wall_clock"
                    break
                if step - invocation_started_step >= invocation_step_budget:
                    halt_reason = "invocation_step_limit"
                    break
                if cursor >= len(order):
                    epoch += 1
                    cursor = 0
                    order = family_depth_balanced_order(
                        projected_train,
                        seed=config.seed,
                        epoch=epoch,
                    )
                    _publish_sampling_receipt(
                        out_dir,
                        projected_train,
                        order,
                        seed=config.seed,
                        epoch=epoch,
                        custody=out_custody,
                    )
                    out_custody.verify()
                row = projected_train[order[cursor]]
                before_update = adapter_tensor_fingerprint(adapter_tensor_dict(model))
                result = cached_supervised_live_path_value_and_grad(
                    model,
                    row["prompt_tokens"],
                    row["answer_tokens"],
                    spec=spec,
                    bridge_tokens=row["bridge_tokens"],
                    branch_indices=config.branch_indices,
                )
                if (
                    result.execution_spec_sha256 != spec.sha256
                    or result.prompt_tokens_sha256 != sha256_json(row["prompt_tokens"])
                    or result.answer_tokens_sha256 != sha256_json(row["answer_tokens"])
                    or not math.isfinite(result.value)
                ):
                    _fail("resident_sft_trainer_objective_identity_drift")
                optimizer.update(model, result.gradients)
                mx.eval(model.trainable_parameters(), optimizer.state)
                adapter = adapter_tensor_dict(model)
                after_update = adapter_tensor_fingerprint(adapter)
                if after_update == before_update:
                    _fail("resident_sft_trainer_optimizer_update_noop")
                step += 1
                cursor += 1
                sample_history = advance_sample_history(
                    sample_history,
                    example_id=row["example_id"],
                    step=step,
                    epoch=epoch,
                    cursor=cursor,
                )
                loss_trail.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "cursor": cursor,
                        "example_id": row["example_id"],
                        "loss": result.value,
                        "branch_values": list(result.branch_values),
                        "adapter_before_sha256": before_update,
                        "adapter_after_sha256": after_update,
                    }
                )
                reached_max = step >= config.max_steps
                reached_wall = elapsed() >= config.max_minutes * 60.0
                terminal = reached_max or reached_wall
                halt = "max_steps" if reached_max else "wall_clock" if reached_wall else None
                if step % config.evaluate_every == 0 or terminal:
                    validation = _validation_summary(
                        model,
                        projected_validation,
                        spec=spec,
                        config=config,
                    )
                    validation_trail.append({"step": step, **validation})
                sequence += 1
                state = _state_document(
                    bindings,
                    sequence=sequence,
                    step=step,
                    epoch=epoch,
                    cursor=cursor,
                    order=order,
                    config=config,
                    train_count=len(projected_train),
                    validation_count=len(projected_validation),
                    elapsed_s=elapsed(),
                    invocation_count=invocation_count,
                    sample_history_sha256=sample_history,
                    initial_adapter_sha256=initial_adapter_sha,
                    adapter_topology_identity=topology_sha,
                    loss_trail=loss_trail,
                    validation_trail=validation_trail,
                    baseline_validation=baseline,
                    terminal=terminal,
                    halt_reason=halt,
                )
                save_checkpoint(
                    out_dir,
                    adapter_tensors=adapter,
                    optimizer_tensors=dict(tree_flatten(optimizer.state)),
                    state=state,
                    custody=out_custody,
                )
                out_custody.verify()
                if terminal:
                    halt_reason = str(halt)
                    break

            inspected = inspect_checkpoint(
                out_dir,
                expected_bindings=bindings,
                custody=out_custody,
            )
            final_state = inspected.state
            if final_state["step"] != required_end_step:
                _fail("resident_sft_trainer_required_end_step_not_reached")
            if halt_reason == "wall_clock" and not final_state["terminal"]:
                sequence += 1
                final_state = _state_document(
                    bindings,
                    sequence=sequence,
                    step=step,
                    epoch=epoch,
                    cursor=cursor,
                    order=order,
                    config=config,
                    train_count=len(projected_train),
                    validation_count=len(projected_validation),
                    elapsed_s=elapsed(),
                    invocation_count=invocation_count,
                    sample_history_sha256=sample_history,
                    initial_adapter_sha256=initial_adapter_sha,
                    adapter_topology_identity=topology_sha,
                    loss_trail=loss_trail,
                    validation_trail=validation_trail,
                    baseline_validation=baseline,
                    terminal=True,
                    halt_reason="wall_clock",
                )
                save_checkpoint(
                    out_dir,
                    adapter_tensors=adapter_tensor_dict(model),
                    optimizer_tensors=dict(tree_flatten(optimizer.state)),
                    state=final_state,
                    custody=out_custody,
                )
                out_custody.verify()
                inspected = inspect_checkpoint(
                    out_dir,
                    expected_bindings=bindings,
                    custody=out_custody,
                )
                final_state = inspected.state
            base_after = full_weight_checkpoint_identity(model_dir)
            receipt = _publish_receipt(
                out_dir,
                authority=authority,
                state=final_state,
                checkpoint_sha256=inspected.complete_sha256,
                base_before=base_before,
                base_after=base_after,
                halt_reason=halt_reason,
                required_end_step=required_end_step,
                custody=out_custody,
            )
            out_custody.verify()
            print(json.dumps(receipt, sort_keys=True), flush=True)
            del model, tokenizer, optimizer
            mx.synchronize()
            return 0


def _signal_handler(_signum: int, _frame: Any) -> None:
    global INTERRUPTED
    INTERRUPTED = True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--resume-policy", choices=sorted(RESUME_POLICIES), default="auto")
    parser.add_argument("--invocation-step-budget", type=int)
    parser.add_argument("--required-end-step", type=int)
    args = parser.parse_args(argv)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _signal_handler)
    try:
        return _run(args)
    except Exception as exc:
        error = {
            "schema": "aura.resident_recurrent_sft_bootstrap_error.v1",
            "error_type": type(exc).__name__,
            "error": str(exc) or "no_message",
            "claims_supported": [],
        }
        print(json.dumps(error, sort_keys=True), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
