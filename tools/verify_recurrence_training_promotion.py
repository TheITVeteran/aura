#!/usr/bin/env python
"""Verify terminal recurrence training and its immutable adapter freeze.

This gate certifies training completion, containment, resource limits, and
artifact identity.  It deliberately does not certify a reasoning gain or make
the adapter pilot-eligible; a fresh frozen-adapter mechanics campaign remains
mandatory after this receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes  # noqa: E402
from core.brain.llm.latent_cortex.campaign_launch_bundle import (  # noqa: E402
    adapter_artifact_inventory,
    inventory_root_sha256,
    read_canonical_json,
    sha256_bytes,
    verify_adapter_freeze,
)
from core.learning.recurrence_curriculum import (  # noqa: E402
    RECURRENCE_TRAINING_FAMILIES,
    task_battery,
)
from core.runtime.file_read_gateway import (  # noqa: E402
    open_stable_readonly_binary,
    read_stable_bytes,
)
from tools import prepare_latent_cortex_campaign as preparation  # noqa: E402
from tools import run_detached_step as detached  # noqa: E402
from tools import run_latent_cortex_paired_campaign as campaign_runner  # noqa: E402

SCHEMA = "aura.latent_cortex.recurrence_training_promotion.v1"
RESOURCE_SCHEMA = "aura.recurrence_training_resource_envelope.v1"
TRAINING_CONFIG_SCHEMA = "aura.recurrence_native_training_config.v2"
TRAINING_DATASET_SCHEMA = "aura.recurrence_native_dataset.v2"
TRAINING_RECEIPT_SCHEMA = "aura.recurrence_native_train.v2"
TRAINING_COMPLETION_SCHEMA = "aura.recurrence_native_training_completion.v1"
TRAINING_CHECKPOINT_SCHEMA = "aura.recurrence_native_checkpoint.v2"
TRAINING_POINTER_SCHEMA = "aura.recurrence_native_checkpoint_pointer.v1"
_MAX_FILE_BYTES = 1 << 40
_GIB = 1024**3
_CURRICULUM_PATH = REPO_ROOT / "core/learning/recurrence_curriculum.py"
_DEFAULT_CONTRACT_PATH = (
    REPO_ROOT / "config/latent_cortex/resident_32b_training_contract.json"
)
_SAFETENSOR_DTYPES = {
    "BOOL": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
}


class TrainingPromotionError(RuntimeError):
    """Stable fail-closed promotion error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise TrainingPromotionError(code)


def _sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _file_binding(path: Path, *, role: str) -> dict[str, Any]:
    resolved = path.expanduser()
    if resolved.is_symlink():
        _fail(f"{role}_symlink_rejected")
    resolved = resolved.resolve(strict=True)
    digest = hashlib.sha256()
    try:
        with open_stable_readonly_binary(resolved, max_bytes=_MAX_FILE_BYTES) as (
            handle,
            identity,
        ):
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    except (OSError, ValueError) as exc:
        raise TrainingPromotionError(f"{role}_unavailable") from exc
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "size_bytes": identity.size,
    }


def _option_map(argv: list[str], *, role: str) -> dict[str, str]:
    result: dict[str, str] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--") or "=" in token:
            _fail(f"{role}_argv_invalid")
        if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
            _fail(f"{role}_argv_invalid")
        if token in result:
            _fail(f"{role}_option_duplicate")
        result[token] = argv[index + 1]
        index += 2
    return result


def _canonical_path(
    value: str | Path,
    *,
    role: str,
    kind: str | None = None,
) -> Path:
    supplied = Path(value).expanduser()
    if not supplied.is_absolute() or supplied.is_symlink():
        _fail(f"{role}_path_invalid")
    resolved = supplied.resolve(strict=True)
    if str(supplied) != str(resolved):
        _fail(f"{role}_path_invalid")
    if kind == "file" and not resolved.is_file():
        _fail(f"{role}_path_invalid")
    if kind == "directory" and not resolved.is_dir():
        _fail(f"{role}_path_invalid")
    return resolved


def _integer_option(options: Mapping[str, str], option: str, *, role: str, minimum: int = 1) -> int:
    try:
        value = int(options[option])
    except (KeyError, ValueError, OverflowError) as exc:
        raise TrainingPromotionError(f"{role}_invalid") from exc
    if value < minimum:
        _fail(f"{role}_invalid")
    return value


def _float_option(
    options: Mapping[str, str], option: str, *, role: str, minimum: float = 0.0
) -> float:
    try:
        value = float(options[option])
    except (KeyError, ValueError, OverflowError) as exc:
        raise TrainingPromotionError(f"{role}_invalid") from exc
    if value < minimum or value == float("inf") or value == float("-inf") or value != value:
        _fail(f"{role}_invalid")
    return value


def _csv(value: str, *, role: str) -> list[str]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        _fail(f"{role}_invalid")
    return parts


def _csv_integers(value: str, *, role: str) -> list[int]:
    parts = _csv(value, role=role)
    try:
        parsed = [int(part) for part in parts]
    except (ValueError, OverflowError) as exc:
        raise TrainingPromotionError(f"{role}_invalid") from exc
    if any(number < 1 for number in parsed):
        _fail(f"{role}_invalid")
    return parsed


def _load_bound_tokenizer(model: Path) -> Any:
    try:
        from mlx_lm.utils import load_tokenizer

        config_path = model / "config.json"
        if config_path.is_symlink():
            _fail("training_tokenizer_config_invalid")
        config = json.loads(
            read_stable_bytes(config_path, max_bytes=4 * 1024 * 1024),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_nonfinite_json,
        )
        eos_token_ids = config.get("eos_token_id") if isinstance(config, Mapping) else None
        eos_values = eos_token_ids if isinstance(eos_token_ids, list) else [eos_token_ids]
        if not eos_values or any(
            type(token_id) is not int or not 0 <= token_id < 2**31
            for token_id in eos_values
        ):
            _fail("training_tokenizer_eos_contract_invalid")
        return load_tokenizer(model, eos_token_ids=eos_token_ids)
    except TrainingPromotionError:
        raise
    except (
        ImportError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
    ) as exc:
        raise TrainingPromotionError("training_tokenizer_unavailable") from exc


def _stable_tokenizer_behavior_sha256(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> str:
    if before != after:
        _fail("training_tokenizer_generation_changed")
    digest = before.get("bundle_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        _fail("training_tokenizer_behavior_identity_invalid")
    return digest


def _canonical_binding_matches(document: Mapping[str, Any], binding: Mapping[str, Any]) -> bool:
    payload = canonical_json_bytes(document) + b"\n"
    return binding.get("sha256") == sha256_bytes(payload) and binding.get("size_bytes") == len(
        payload
    )


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _validate_safetensors(path: Path, *, role: str) -> dict[str, Any]:
    raw = read_stable_bytes(path, max_bytes=_MAX_FILE_BYTES)
    if len(raw) < 9:
        _fail(f"{role}_safetensors_truncated")
    header_size = int.from_bytes(raw[:8], "little", signed=False)
    if header_size < 2 or header_size > len(raw) - 8:
        _fail(f"{role}_safetensors_header_invalid")
    header_raw = raw[8 : 8 + header_size]
    try:
        header = json.loads(
            header_raw,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, OverflowError) as exc:
        raise TrainingPromotionError(f"{role}_safetensors_header_invalid") from exc
    if not isinstance(header, dict):
        _fail(f"{role}_safetensors_header_invalid")
    data_size = len(raw) - 8 - header_size
    intervals: list[tuple[int, int, str]] = []
    for name, record in header.items():
        if name == "__metadata__":
            if not isinstance(record, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in record.items()
            ):
                _fail(f"{role}_safetensors_metadata_invalid")
            continue
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(record, dict)
            or set(record) != {"dtype", "shape", "data_offsets"}
        ):
            _fail(f"{role}_safetensors_tensor_invalid")
        dtype = record.get("dtype")
        shape = record.get("shape")
        offsets = record.get("data_offsets")
        if (
            dtype not in _SAFETENSOR_DTYPES
            or not isinstance(shape, list)
            or any(type(size) is not int or size < 0 for size in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int or offset < 0 for offset in offsets)
        ):
            _fail(f"{role}_safetensors_tensor_invalid")
        start, end = offsets
        if (
            end < start
            or end > data_size
            or end - start != math.prod(shape) * _SAFETENSOR_DTYPES[dtype]
        ):
            _fail(f"{role}_safetensors_offsets_invalid")
        intervals.append((start, end, name))
    intervals.sort()
    if (
        not intervals
        or intervals[0][0] != 0
        or intervals[-1][1] != data_size
        or any(
            left[1] != right[0]
            for left, right in zip(intervals, intervals[1:], strict=False)
        )
    ):
        _fail(f"{role}_safetensors_layout_invalid")
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "tensor_count": len(intervals),
        "tensor_header_sha256": hashlib.sha256(header_raw).hexdigest(),
    }


def _authoritative_detached_evidence(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    inspection = detached._status(run_dir)
    receipt = inspection.get("receipt")
    if (
        inspection.get("terminal") is not True
        or inspection.get("completion_indeterminate") is not False
        or inspection.get("state") != "passed"
        or inspection.get("supervisor_alive") is not False
        or inspection.get("child_state") != "dead"
        or not isinstance(receipt, dict)
    ):
        _fail("detached_training_journal_not_terminal")
    plan_path = run_dir / detached.PLAN_FILE
    plan = read_canonical_json(plan_path, role="detached_training_plan")
    detached._verify_plan(plan, plan_path)
    persisted_receipt = detached._verified_receipt(run_dir / detached.RECEIPT_FILE)
    if (
        persisted_receipt != receipt
        or inspection.get("plan_sha256") != plan.get("plan_sha256")
        or inspection.get("restart_count") != 0
        or type(inspection.get("attempt_event_count")) is not int
        or inspection["attempt_event_count"] < 4
        or not isinstance(inspection.get("attempt_journal_head_sha256"), str)
        or len(inspection["attempt_journal_head_sha256"]) != 64
    ):
        _fail("detached_training_journal_binding_invalid")
    return plan, receipt, {
        "attempt_event_count": inspection["attempt_event_count"],
        "attempt_journal_head_sha256": inspection["attempt_journal_head_sha256"],
        "supervisor_attempt": inspection["supervisor_attempt"],
    }


def _validate_python_launcher(plan: Mapping[str, Any], launcher: Path) -> dict[str, Any]:
    binding = plan.get("executable_binding")
    venv_root = launcher.parent.parent
    pyvenv_path = venv_root / "pyvenv.cfg"
    resolved_value = Path(str(binding.get("resolved_path") or "")) if isinstance(binding, Mapping) else Path()
    if (
        not isinstance(binding, Mapping)
        or binding.get("schema") != "aura.detached_step.launcher_binding.v1"
        or binding.get("invocation_path") != str(launcher)
        or launcher.parent.name != "bin"
        or venv_root.name != ".venv"
        or not launcher.name.startswith("python")
        or not pyvenv_path.is_file()
        or not resolved_value.name.startswith("python")
        or resolved_value.resolve(strict=True) != launcher.resolve(strict=True)
    ):
        _fail("training_launcher_not_pinned_python")
    pyvenv = binding.get("pyvenv")
    observed_pyvenv = _file_binding(pyvenv_path, role="training_python_environment")
    if (
        not isinstance(pyvenv, Mapping)
        or pyvenv.get("path") != str(pyvenv_path)
        or pyvenv.get("sha256") != observed_pyvenv["sha256"]
        or pyvenv.get("size") != observed_pyvenv["size_bytes"]
    ):
        _fail("training_python_environment_mismatch")
    return {
        "binding_sha256": binding.get("binding_sha256"),
        "invocation_path": str(launcher),
        "resolved_path": str(binding["resolved_path"]),
        "resolved_sha256": str(binding["resolved_sha256"]),
        "pyvenv_sha256": str(pyvenv["sha256"]),
    }


def _artifact_json(
    root: Path,
    manifest: Mapping[str, Any],
    role: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = manifest.get(role)
    if not isinstance(binding, Mapping) or not isinstance(binding.get("path"), str):
        _fail(f"training_{role}_binding_invalid")
    path = root / str(binding["path"])
    try:
        if path.parent.resolve(strict=True) != root:
            _fail(f"training_{role}_path_invalid")
    except OSError as exc:
        raise TrainingPromotionError(f"training_{role}_path_invalid") from exc
    observed = _file_binding(path, role=f"training_{role}")
    if (
        binding.get("sha256") != observed["sha256"]
        or binding.get("size_bytes") != observed["size_bytes"]
    ):
        _fail(f"training_{role}_binding_mismatch")
    return read_canonical_json(path, role=f"training_{role}"), observed


def _validate_training_dataset(
    dataset: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    source_adapter: Path,
    training_worktree: Path,
    families: list[str],
    task_depths: list[int],
    per_cell: int,
    train_seed: int,
    tokenizer: Any,
) -> dict[str, Any]:
    source_records = manifest.get("sources")
    source = (
        source_records.get("task_generator")
        if isinstance(source_records, Mapping)
        else None
    )
    if not isinstance(source, Mapping) or set(source) != {
        "origin_path",
        "snapshot_path",
        "sha256",
        "size_bytes",
    }:
        _fail("training_curriculum_binding_invalid")
    if (
        source.get("origin_path") != "core/learning/recurrence_curriculum.py"
        or source.get("snapshot_path") != "source_snapshots/task_generator.py"
    ):
        _fail("training_curriculum_origin_invalid")
    worktree_source = training_worktree / str(source["origin_path"])
    snapshot = source_adapter / str(source["snapshot_path"])
    try:
        worktree_root = training_worktree.resolve(strict=True)
        adapter_root = source_adapter.resolve(strict=True)
        if (
            worktree_source.is_symlink()
            or snapshot.is_symlink()
            or not worktree_source.resolve(strict=True).is_relative_to(worktree_root)
            or not snapshot.resolve(strict=True).is_relative_to(adapter_root)
        ):
            _fail("training_curriculum_path_invalid")
    except OSError as exc:
        raise TrainingPromotionError("training_curriculum_path_invalid") from exc
    expected_source = _file_binding(_CURRICULUM_PATH, role="trusted_curriculum")
    for role, path in (
        ("training_curriculum", worktree_source),
        ("training_curriculum_snapshot", snapshot),
    ):
        observed = _file_binding(path, role=role)
        if (
            observed["sha256"] != source.get("sha256")
            or observed["size_bytes"] != source.get("size_bytes")
            or observed["sha256"] != expected_source["sha256"]
            or observed["size_bytes"] != expected_source["size_bytes"]
        ):
            _fail("training_curriculum_source_mismatch")
    expected_generator = {
        "path": source["origin_path"],
        "sha256": source["sha256"],
        "size_bytes": source["size_bytes"],
    }
    if (
        set(dataset)
        != {
            "schema",
            "generator",
            "train_seed",
            "families",
            "task_depths",
            "per_cell",
            "examples",
        }
        or dataset.get("schema") != TRAINING_DATASET_SCHEMA
        or dataset.get("generator") != expected_generator
        or dataset.get("train_seed") != train_seed
        or dataset.get("families") != families
        or dataset.get("task_depths") != task_depths
        or dataset.get("per_cell") != per_cell
        or any(family not in RECURRENCE_TRAINING_FAMILIES for family in families)
    ):
        _fail("training_dataset_contract_mismatch")
    examples = dataset.get("examples")
    expected_tasks = task_battery(families, task_depths, per_cell, seed=train_seed)
    if not isinstance(examples, list) or len(examples) != len(expected_tasks):
        _fail("training_dataset_example_count_mismatch")
    semantic_examples: list[dict[str, Any]] = []
    prompt_token_count = 0
    answer_token_count = 0
    for index, (example, task) in enumerate(
        zip(examples, expected_tasks, strict=True)
    ):
        if not isinstance(example, Mapping) or set(example) != {
            "family",
            "depth",
            "seed",
            "prompt",
            "answer",
            "prompt_tokens",
            "answer_tokens",
        }:
            _fail("training_dataset_example_schema_invalid")
        semantic = {
            "family": task.family,
            "depth": int(task.depth),
            "seed": int(task.seed),
            "prompt": str(task.prompt),
            "answer": str(task.answer),
        }
        if any(example.get(key) != value for key, value in semantic.items()):
            _fail(f"training_dataset_example_{index}_mismatch")
        try:
            expected_prompt_tokens = list(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": task.prompt}],
                    add_generation_prompt=True,
                    tokenize=True,
                )
            )
            try:
                expected_answer_tokens = list(
                    tokenizer.encode(str(task.answer), add_special_tokens=False)
                )
            except TypeError:
                expected_answer_tokens = list(tokenizer.encode(str(task.answer)))
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is not None:
                expected_answer_tokens.append(int(eos))
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            raise TrainingPromotionError(
                f"training_dataset_example_{index}_tokenization_failed"
            ) from exc
        for token_role in ("prompt_tokens", "answer_tokens"):
            tokens = example.get(token_role)
            if (
                not isinstance(tokens, list)
                or not tokens
                or any(
                    type(token) is not int or not 0 <= token < 2**31
                    for token in tokens
                )
            ):
                _fail(f"training_dataset_{token_role}_invalid")
        if (
            example["prompt_tokens"] != expected_prompt_tokens
            or example["answer_tokens"] != expected_answer_tokens
        ):
            _fail(f"training_dataset_example_{index}_tokenization_mismatch")
        prompt_token_count += len(example["prompt_tokens"])
        answer_token_count += len(example["answer_tokens"])
        semantic_examples.append(semantic)
    return {
        "example_count": len(examples),
        "cell_count": len(families) * len(task_depths),
        "prompt_token_count": prompt_token_count,
        "answer_token_count": answer_token_count,
        "semantic_examples_sha256": _sha256(semantic_examples),
        "trusted_curriculum_sha256": expected_source["sha256"],
    }


def _validate_acceptance_contract(
    contract: Mapping[str, Any],
    contract_binding: Mapping[str, Any],
    *,
    detached_plan: Mapping[str, Any],
    artifact_bindings: Mapping[str, Mapping[str, Any]],
    training_manifest: Mapping[str, Any],
    training_config: Mapping[str, Any],
    execution_spec: Mapping[str, Any],
    resource_envelope: Mapping[str, Any],
    adapter_id: str,
    workload: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        "accepted_artifacts",
        "accepted_plan",
        "adapter_id",
        "claim_scope",
        "contract_id",
        "contract_sha256",
        "evidence_limitations",
        "external_attestation_present",
        "gradient_execution",
        "model_identity",
        "optimizer",
        "producer_sources",
        "required_next_gates",
        "resource_envelope",
        "schema",
        "workload",
    }
    material = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }
    limitations = contract.get("evidence_limitations")
    required_next = contract.get("required_next_gates")
    if (
        set(contract) != expected_keys
        or contract.get("schema")
        != "aura.recurrence_training_acceptance_contract.v1"
        or contract.get("contract_sha256") != _sha256(material)
        or contract.get("claim_scope") != "internal_mechanics_acceptance_only"
        or contract.get("external_attestation_present") is not False
        or not isinstance(contract.get("contract_id"), str)
        or not contract["contract_id"]
        or not isinstance(limitations, list)
        or set(limitations)
        != {
            "training_started_before_contract_commit",
            "detached_training_terminal_does_not_bind_output_root",
            "external_attestation_not_present",
            "reasoning_and_frontier_gain_not_measured",
        }
        or not isinstance(required_next, list)
        or set(required_next)
        != {
            "terminal_artifact_generation_validation",
            "immutable_adapter_freeze",
            "resident_32b_frozen_adapter_mechanics_smoke",
            "fresh_hidden_task_pilot",
            "powered_external_frontier_campaign",
        }
        or contract.get("adapter_id") != adapter_id
    ):
        _fail("training_acceptance_contract_invalid")
    if not _canonical_binding_matches(contract, contract_binding):
        _fail("training_acceptance_contract_generation_changed")

    target_manifest = detached_plan.get("target_execution_manifest")
    observed_plan = {
        "broker_policy_sha256": detached_plan.get("broker_policy_sha256"),
        "command_sha256": detached_plan.get("command_sha256"),
        "executable_sha256": detached_plan.get("executable_sha256"),
        "execution_environment_sha256": detached_plan.get(
            "execution_environment_sha256"
        ),
        "fork_policy": detached_plan.get("fork_policy"),
        "name": detached_plan.get("name"),
        "plan_sha256": detached_plan.get("plan_sha256"),
        "restart_policy": detached_plan.get("restart_policy"),
        "resume_contract": detached_plan.get("resume_contract"),
        "target_execution_manifest_sha256": (
            target_manifest.get("manifest_sha256")
            if isinstance(target_manifest, Mapping)
            else None
        ),
        "timeout_s": detached_plan.get("timeout_s"),
    }
    if contract.get("accepted_plan") != observed_plan:
        _fail("training_acceptance_plan_mismatch")

    accepted_artifacts = contract.get("accepted_artifacts")
    if not isinstance(accepted_artifacts, Mapping) or set(accepted_artifacts) != {
        "training_config",
        "dataset_manifest",
        "execution_spec",
        "resource_envelope",
    }:
        _fail("training_acceptance_artifacts_invalid")
    artifact_roles = {
        "training_config": "config",
        "dataset_manifest": "dataset",
        "execution_spec": "execution_spec",
        "resource_envelope": "resource",
    }
    for contract_role, binding_role in artifact_roles.items():
        accepted = accepted_artifacts.get(contract_role)
        observed = artifact_bindings.get(binding_role)
        if (
            not isinstance(accepted, Mapping)
            or not isinstance(observed, Mapping)
            or accepted.get("sha256") != observed.get("sha256")
            or accepted.get("size_bytes") != observed.get("size_bytes")
        ):
            _fail("training_acceptance_artifact_mismatch")
    execution_acceptance = accepted_artifacts["execution_spec"]
    if execution_acceptance.get("semantic_sha256") != _sha256(execution_spec):
        _fail("training_acceptance_execution_spec_mismatch")

    base = training_config.get("base_checkpoint")
    behavior = training_config.get("model_behavior_bundle")
    runtime = training_config.get("training_runtime")
    observed_model = {
        "base_checkpoint_fingerprint": (
            base.get("fingerprint") if isinstance(base, Mapping) else None
        ),
        "base_checkpoint_files": base.get("files") if isinstance(base, Mapping) else None,
        "model_behavior_bundle_sha256": (
            behavior.get("bundle_sha256")
            if isinstance(behavior, Mapping)
            else None
        ),
        "training_runtime_identity_sha256": (
            runtime.get("identity_sha256") if isinstance(runtime, Mapping) else None
        ),
    }
    if contract.get("model_identity") != observed_model:
        _fail("training_acceptance_model_mismatch")
    if (
        contract.get("workload") != dict(workload)
        or contract.get("optimizer") != training_config.get("optimizer")
        or contract.get("gradient_execution")
        != training_config.get("gradient_execution")
    ):
        _fail("training_acceptance_workload_mismatch")

    source_records = training_manifest.get("sources")
    if not isinstance(source_records, Mapping):
        _fail("training_acceptance_sources_invalid")
    observed_sources = {
        f"{role}_sha256": record.get("sha256")
        for role, record in source_records.items()
        if isinstance(role, str) and isinstance(record, Mapping)
    }
    wrapper = artifact_bindings.get("wrapper")
    trainer = artifact_bindings.get("trainer")
    if isinstance(wrapper, Mapping):
        observed_sources["wrapper_sha256"] = wrapper.get("sha256")
    if isinstance(trainer, Mapping):
        observed_sources["trainer_sha256"] = trainer.get("sha256")
    if contract.get("producer_sources") != observed_sources:
        _fail("training_acceptance_sources_mismatch")

    device = resource_envelope.get("device")
    accepted_resource = contract.get("resource_envelope")
    if (
        not isinstance(device, Mapping)
        or not isinstance(accepted_resource, Mapping)
        or accepted_resource.get("memory_limit_bytes")
        != resource_envelope.get("memory_limit_bytes")
        or accepted_resource.get("cache_limit_bytes")
        != resource_envelope.get("cache_limit_bytes")
        or accepted_resource.get("wired_limit_bytes")
        != resource_envelope.get("wired_limit_bytes")
        or not isinstance(device.get("memory_size"), int)
        or device["memory_size"]
        < accepted_resource.get("minimum_device_memory_bytes", 2**63)
        or not isinstance(device.get("max_recommended_working_set_size"), int)
        or device["max_recommended_working_set_size"]
        < accepted_resource.get("minimum_recommended_working_set_bytes", 2**63)
    ):
        _fail("training_acceptance_resource_mismatch")
    return {
        "contract_id": contract["contract_id"],
        "contract_sha256": contract["contract_sha256"],
        "contract_file_sha256": contract_binding["sha256"],
        "claim_scope": contract["claim_scope"],
        "evidence_limitations": list(limitations),
        "required_next_gates": list(required_next),
    }


def _validate_loss_trail(
    loss_trail: Any,
    *,
    expected_steps: int,
    log_every: int,
) -> bool:
    if not isinstance(loss_trail, list) or not loss_trail:
        return False
    expected_logged_steps = list(range(log_every, expected_steps + 1, log_every))
    remainder = expected_steps % log_every
    if remainder:
        expected_logged_steps.append(expected_steps)
    if len(loss_trail) != len(expected_logged_steps):
        return False
    for record, step in zip(loss_trail, expected_logged_steps, strict=True):
        partial = remainder > 0 and step == expected_steps
        expected_record_keys = (
            {"step", "mean_loss", "window_steps", "partial_window"}
            if partial
            else {"step", "mean_loss"}
        )
        if (
            not isinstance(record, Mapping)
            or set(record) != expected_record_keys
            or record.get("step") != step
            or not isinstance(record.get("mean_loss"), (int, float))
            or isinstance(record.get("mean_loss"), bool)
            or not math.isfinite(float(record["mean_loss"]))
            or float(record["mean_loss"]) < 0.0
            or (
                partial
                and (
                    record.get("window_steps") != remainder
                    or record.get("partial_window") is not True
                )
            )
        ):
            return False
    return True


def _latest_checkpoint_evidence(
    source_adapter: Path,
    latest: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        set(latest) != {"schema", "checkpoint", "complete_sha256"}
        or latest.get("schema") != TRAINING_POINTER_SCHEMA
        or not isinstance(latest.get("checkpoint"), str)
    ):
        _fail("training_latest_invalid")
    relative = PurePosixPath(str(latest["checkpoint"]))
    if (
        len(relative.parts) != 2
        or relative.parts[0] != "checkpoints"
        or not relative.parts[1].startswith("step-")
        or str(relative) != latest["checkpoint"]
    ):
        _fail("training_latest_path_invalid")
    checkpoint = source_adapter.joinpath(*relative.parts)
    checkpoint_root = source_adapter / "checkpoints"
    try:
        if (
            checkpoint_root.is_symlink()
            or checkpoint_root.resolve(strict=True).parent != source_adapter
            or checkpoint.is_symlink()
            or checkpoint.resolve(strict=True).parent != checkpoint_root.resolve(strict=True)
            or not checkpoint.is_dir()
        ):
            _fail("training_latest_path_invalid")
    except OSError as exc:
        raise TrainingPromotionError("training_latest_path_invalid") from exc
    complete_path = checkpoint / "complete.json"
    complete_binding = _file_binding(complete_path, role="training_checkpoint_completion")
    if latest.get("complete_sha256") != complete_binding["sha256"]:
        _fail("training_latest_completion_mismatch")
    complete = read_canonical_json(complete_path, role="training_checkpoint_completion")
    if not _canonical_binding_matches(complete, complete_binding):
        _fail("training_checkpoint_completion_generation_changed")
    if (
        set(complete)
        != {
            "schema",
            "checkpoint_id",
            "step",
            "epoch",
            "cursor",
            "order",
            "config_sha256",
            "dataset_sha256",
            "execution_spec_sha256",
            "elapsed_training_s",
            "invocation_count",
            "loss_trail",
            "sampler",
            "stochastic_state",
            "adapter",
            "optimizer",
        }
        or complete.get("schema") != TRAINING_CHECKPOINT_SCHEMA
        or complete.get("sampler") != "sha256_stateless_epoch_permutation.v1"
        or complete.get("stochastic_state") != "none_all_keys_explicit"
    ):
        _fail("training_checkpoint_schema_invalid")
    bindings: dict[str, dict[str, Any]] = {}
    tensor_inventories: dict[str, dict[str, Any]] = {}
    for role in ("adapter", "optimizer"):
        declared = complete.get(role)
        if not isinstance(declared, Mapping) or set(declared) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            _fail(f"training_checkpoint_{role}_binding_invalid")
        artifact = checkpoint / str(declared["path"])
        if artifact.parent.resolve(strict=True) != checkpoint.resolve(strict=True):
            _fail(f"training_checkpoint_{role}_path_invalid")
        observed = _file_binding(artifact, role=f"training_checkpoint_{role}")
        if (
            declared.get("sha256") != observed["sha256"]
            or declared.get("size_bytes") != observed["size_bytes"]
        ):
            _fail(f"training_checkpoint_{role}_binding_mismatch")
        bindings[role] = observed
        tensor_inventories[role] = _validate_safetensors(
            artifact, role=f"training_checkpoint_{role}"
        )
    return {
        "checkpoint": str(latest["checkpoint"]),
        "complete": complete,
        "complete_binding": complete_binding,
        "artifacts": bindings,
        "tensor_inventories": tensor_inventories,
    }


def _validate_resource_envelope(
    resource: dict[str, Any],
    *,
    wrapper: Path,
    trainer: Path,
    wrapper_options: dict[str, str],
) -> dict[str, Any]:
    if (
        set(resource)
        != {
            "schema",
            "memory_limit_bytes",
            "cache_limit_bytes",
            "wired_limit_bytes",
            "cache_cleared_before_model_load",
            "device",
            "mlx_version",
            "wrapper_sha256",
            "trainer_sha256",
        }
        or resource.get("schema") != RESOURCE_SCHEMA
        or resource.get("cache_cleared_before_model_load") is not True
        or not isinstance(resource.get("mlx_version"), str)
        or not resource["mlx_version"]
    ):
        _fail("resource_envelope_schema_invalid")
    limits = {
        "memory_limit_bytes": resource.get("memory_limit_bytes"),
        "cache_limit_bytes": resource.get("cache_limit_bytes"),
        "wired_limit_bytes": resource.get("wired_limit_bytes"),
    }
    if any(type(value) is not int or value <= 0 for value in limits.values()):
        _fail("resource_envelope_limit_invalid")
    if not limits["cache_limit_bytes"] < limits["memory_limit_bytes"] < limits["wired_limit_bytes"]:
        _fail("resource_envelope_limit_order_invalid")
    device = resource.get("device")
    if (
        not isinstance(device, dict)
        or set(device) != {"architecture", "memory_size", "max_recommended_working_set_size"}
        or not isinstance(device.get("architecture"), str)
        or not device.get("architecture")
        or type(device.get("memory_size")) is not int
        or type(device.get("max_recommended_working_set_size")) is not int
        or device["memory_size"] <= 0
        or device["max_recommended_working_set_size"] <= 0
        or limits["wired_limit_bytes"] >= device["memory_size"]
        or limits["wired_limit_bytes"] > device["max_recommended_working_set_size"]
    ):
        _fail("resource_envelope_device_invalid")
    option_limits = {
        "memory_limit_bytes": "--memory-limit-gb",
        "cache_limit_bytes": "--cache-limit-gb",
        "wired_limit_bytes": "--wired-limit-gb",
    }
    for key, option in option_limits.items():
        try:
            expected = float(wrapper_options[option]) * _GIB
        except (KeyError, ValueError, OverflowError) as exc:
            raise TrainingPromotionError("resource_envelope_option_invalid") from exc
        if not expected.is_integer() or int(expected) != limits[key]:
            _fail("resource_envelope_option_mismatch")
    if (
        resource.get("wrapper_sha256") != _file_binding(wrapper, role="wrapper")["sha256"]
        or resource.get("trainer_sha256") != _file_binding(trainer, role="trainer")["sha256"]
    ):
        _fail("resource_envelope_source_mismatch")
    return {**limits, "device": device, "mlx_version": resource["mlx_version"]}


def _validate_documents(
    *,
    detached_plan: dict[str, Any],
    detached_receipt: dict[str, Any],
    detached_evidence: dict[str, Any],
    acceptance_contract: dict[str, Any],
    acceptance_contract_binding: dict[str, Any],
    resource_envelope: dict[str, Any],
    resource_path: Path,
    training_manifest: dict[str, Any],
    training_receipt: dict[str, Any],
    training_config: dict[str, Any],
    training_dataset: dict[str, Any],
    execution_spec: dict[str, Any],
    training_completion: dict[str, Any],
    latest: dict[str, Any],
    checkpoint_evidence: dict[str, Any],
    artifact_bindings: dict[str, dict[str, Any]],
    freeze_certificate: dict[str, Any],
    source_inventory: list[dict[str, Any]],
    source_adapter: Path,
    frozen_adapter: Path,
    model: Path,
    adapter_id: str,
    tokenizer: Any,
) -> dict[str, Any]:
    for role, document in (
        ("manifest", training_manifest),
        ("receipt", training_receipt),
        ("config", training_config),
        ("dataset", training_dataset),
        ("execution_spec", execution_spec),
        ("completion", training_completion),
        ("latest", latest),
        ("resource", resource_envelope),
    ):
        binding = artifact_bindings.get(role)
        if not isinstance(binding, Mapping) or not _canonical_binding_matches(document, binding):
            _fail(f"training_{role}_generation_changed")
    if (
        detached_plan.get("schema") != "aura.detached_step.plan.v2"
        or detached_receipt.get("schema") != "aura.detached_step.receipt.v1"
        or detached_receipt.get("plan_sha256") != detached_plan.get("plan_sha256")
        or detached_receipt.get("command_sha256") != detached_plan.get("command_sha256")
        or detached_receipt.get("command") != detached_plan.get("command")
        or detached_receipt.get("returncode") != 0
        or detached_receipt.get("passed") is not True
        or detached_receipt.get("containment_verified") is not True
        or detached_receipt.get("lineage_empty") is not True
        or detached_receipt.get("process_group_empty") is not True
        or detached_receipt.get("timed_out") is not False
        or detached_receipt.get("supervisor_error") is not None
        or detached_receipt.get("restart_count") != 0
        or detached_receipt.get("fork_policy") != "kernel_denied"
        or detached_plan.get("fork_policy") != "kernel_denied"
        or detached_plan.get("restart_policy") != "never"
        or detached_plan.get("resume_contract") != "none"
        or detached_plan.get("broker_policy") != []
        or type(detached_evidence.get("attempt_event_count")) is not int
        or detached_evidence["attempt_event_count"] < 4
        or not isinstance(
            detached_evidence.get("attempt_journal_head_sha256"), str
        )
        or len(detached_evidence["attempt_journal_head_sha256"]) != 64
        or detached_evidence.get("supervisor_attempt") != 1
    ):
        _fail("detached_training_terminal_invalid")
    command = detached_plan.get("command")
    if not isinstance(command, list) or len(command) < 8 or command.count("--") != 1:
        _fail("detached_training_command_invalid")
    separator = command.index("--")
    if separator < 4 or any(not isinstance(value, str) for value in command):
        _fail("detached_training_command_invalid")
    launcher = Path(command[0])
    if not launcher.is_absolute() or not launcher.is_file():
        _fail("training_launcher_path_invalid")
    launcher_identity = _validate_python_launcher(detached_plan, launcher)
    cwd = _canonical_path(
        str(detached_plan.get("cwd") or ""),
        role="training_worktree",
        kind="directory",
    )
    wrapper = _canonical_path(command[1], role="training_wrapper", kind="file")
    wrapper_options = _option_map(command[2:separator], role="training_wrapper")
    trainer_options = _option_map(command[separator + 1 :], role="recurrence_trainer")
    required_wrapper = {
        "--memory-limit-gb",
        "--cache-limit-gb",
        "--wired-limit-gb",
        "--envelope-out",
        "--trainer",
    }
    required_trainer = {
        "--model",
        "--out-dir",
        "--adapter-id",
        "--personality-adapter",
        "--train-seed",
        "--families",
        "--task-depths",
        "--per-cell",
        "--curriculum-depths",
        "--n-slots",
        "--branch-roles",
        "--exchange-interval",
        "--alpha",
        "--alpha-schedule",
        "--lora-rank",
        "--lora-targets",
        "--learning-rate",
        "--monotonicity-weight",
        "--max-minutes",
        "--max-steps",
        "--checkpoint-every",
        "--log-every",
    }
    if set(wrapper_options) != required_wrapper or set(trainer_options) != required_trainer:
        _fail("detached_training_option_set_invalid")
    trainer = _canonical_path(wrapper_options["--trainer"], role="recurrence_trainer", kind="file")
    if artifact_bindings.get("wrapper") != _file_binding(
        wrapper, role="training_wrapper"
    ) or artifact_bindings.get("trainer") != _file_binding(trainer, role="recurrence_trainer"):
        _fail("detached_training_source_generation_changed")
    if (
        wrapper != cwd / "tools/run_recurrence_training_envelope.py"
        or trainer != cwd / "tools/recurrence_native_train_v2.py"
        or _canonical_path(
            wrapper_options["--envelope-out"],
            role="resource_envelope",
            kind="file",
        )
        != resource_path
    ):
        _fail("detached_training_source_path_mismatch")
    if (
        _canonical_path(trainer_options["--model"], role="training_model", kind="directory")
        != model
        or _canonical_path(
            trainer_options["--out-dir"],
            role="training_output",
            kind="directory",
        )
        != source_adapter
        or trainer_options["--adapter-id"] != adapter_id
    ):
        _fail("detached_training_identity_mismatch")

    expected_steps = _integer_option(
        trainer_options,
        "--max-steps",
        role="detached_training_steps",
    )
    train_seed = _integer_option(
        trainer_options,
        "--train-seed",
        role="detached_training_seed",
    )
    families = _csv(trainer_options["--families"], role="detached_training_families")
    task_depths = _csv_integers(
        trainer_options["--task-depths"], role="detached_training_task_depths"
    )
    curriculum_depths = _csv_integers(
        trainer_options["--curriculum-depths"],
        role="detached_training_curriculum_depths",
    )
    branch_roles = _csv(trainer_options["--branch-roles"], role="detached_training_branch_roles")
    lora_targets = _csv(trainer_options["--lora-targets"], role="detached_training_lora_targets")
    per_cell = _integer_option(trainer_options, "--per-cell", role="detached_training_per_cell")
    n_slots = _integer_option(trainer_options, "--n-slots", role="detached_training_slots")
    exchange_interval = _integer_option(
        trainer_options,
        "--exchange-interval",
        role="detached_training_exchange_interval",
    )
    lora_rank = _integer_option(trainer_options, "--lora-rank", role="detached_training_lora_rank")
    learning_rate = _float_option(
        trainer_options,
        "--learning-rate",
        role="detached_training_learning_rate",
    )
    if learning_rate <= 0.0:
        _fail("detached_training_learning_rate_invalid")
    monotonicity_weight = _float_option(
        trainer_options,
        "--monotonicity-weight",
        role="detached_training_monotonicity_weight",
    )
    alpha = _float_option(trainer_options, "--alpha", role="detached_training_alpha")
    max_minutes = _float_option(
        trainer_options, "--max-minutes", role="detached_training_minutes"
    )
    if max_minutes <= 0.0:
        _fail("detached_training_minutes_invalid")
    checkpoint_every = _integer_option(
        trainer_options,
        "--checkpoint-every",
        role="detached_training_checkpoint_interval",
    )
    log_every = _integer_option(
        trainer_options,
        "--log-every",
        role="detached_training_log_interval",
    )
    workload = {
        "alpha": alpha,
        "alpha_schedule": trainer_options["--alpha-schedule"],
        "branch_roles": branch_roles,
        "checkpoint_every": checkpoint_every,
        "curriculum_depths": curriculum_depths,
        "exchange_interval": exchange_interval,
        "families": families,
        "log_every": log_every,
        "lora_rank": lora_rank,
        "lora_targets": lora_targets,
        "max_minutes": max_minutes,
        "max_steps": expected_steps,
        "monotonicity_weight": monotonicity_weight,
        "n_slots": n_slots,
        "per_cell": per_cell,
        "personality_adapter": trainer_options["--personality-adapter"],
        "task_depths": task_depths,
        "train_seed": train_seed,
    }
    contract_evidence = _validate_acceptance_contract(
        acceptance_contract,
        acceptance_contract_binding,
        detached_plan=detached_plan,
        artifact_bindings=artifact_bindings,
        training_manifest=training_manifest,
        training_config=training_config,
        execution_spec=execution_spec,
        resource_envelope=resource_envelope,
        adapter_id=adapter_id,
        workload=workload,
    )

    personality = training_manifest.get("personality_adapter")
    configured_personality = training_config.get("personality_adapter_path")
    requested_personality = trainer_options["--personality-adapter"]
    if not isinstance(personality, Mapping) or not isinstance(configured_personality, str):
        _fail("training_personality_binding_invalid")
    if requested_personality.lower() == "none":
        if configured_personality or personality.get("present") is not False:
            _fail("training_personality_binding_mismatch")
    elif requested_personality.lower() == "auto":
        if configured_personality:
            _canonical_path(
                configured_personality,
                role="training_personality_adapter",
                kind="directory",
            )
        if bool(configured_personality) is not bool(personality.get("present")):
            _fail("training_personality_binding_mismatch")
    else:
        requested_path = _canonical_path(
            requested_personality,
            role="training_personality_adapter",
            kind="directory",
        )
        if configured_personality != str(requested_path) or personality.get("present") is not True:
            _fail("training_personality_binding_mismatch")

    lora_config = training_config.get("lora")
    optimizer = training_config.get("optimizer")
    dataset_evidence = _validate_training_dataset(
        training_dataset,
        training_manifest,
        source_adapter=source_adapter,
        training_worktree=cwd,
        families=families,
        task_depths=task_depths,
        per_cell=per_cell,
        train_seed=train_seed,
        tokenizer=tokenizer,
    )
    if (
        training_manifest.get("schema") != "aura.recurrence_adapter_manifest.v2"
        or training_manifest.get("adapter_id") != adapter_id
        or training_receipt.get("schema") != TRAINING_RECEIPT_SCHEMA
        or training_receipt.get("complete") is not True
        or training_receipt.get("halt_reason") != "max_steps"
        or training_receipt.get("steps") != expected_steps
        or training_config.get("schema") != TRAINING_CONFIG_SCHEMA
        or training_config.get("model_path") != str(model)
        or training_config.get("train_seed") != train_seed
        or training_config.get("max_steps") != expected_steps
        or training_config.get("curriculum_depths") != curriculum_depths
        or training_config.get("monotonicity_weight") != monotonicity_weight
        or training_config.get("execution_spec") != execution_spec
        or not isinstance(lora_config, Mapping)
        or lora_config.get("rank") != lora_rank
        or lora_config.get("targets") != lora_targets
        or not isinstance(optimizer, Mapping)
        or optimizer.get("name") != "AdamW"
        or optimizer.get("learning_rate") != learning_rate
        or training_dataset.get("schema") != TRAINING_DATASET_SCHEMA
        or training_dataset.get("train_seed") != train_seed
        or training_dataset.get("families") != families
        or training_dataset.get("task_depths") != task_depths
        or training_dataset.get("per_cell") != per_cell
        or execution_spec.get("n_slots") != n_slots
        or execution_spec.get("branch_roles") != branch_roles
        or execution_spec.get("exchange_interval") != exchange_interval
        or execution_spec.get("alpha") != alpha
        or execution_spec.get("alpha_schedule") != trainer_options["--alpha-schedule"]
        or execution_spec.get("recurrent_steps") != max(curriculum_depths)
        or training_completion.get("schema") != TRAINING_COMPLETION_SCHEMA
        or training_completion.get("complete") is not True
        or training_completion.get("halt_reason") != "max_steps"
        or training_completion.get("step") != expected_steps
        or training_completion.get("manifest_sha256") != artifact_bindings["manifest"]["sha256"]
        or training_completion.get("adapter_sha256") != training_manifest["adapter"]["sha256"]
        or training_completion.get("receipt_sha256")
        != training_manifest["training_receipt"]["sha256"]
        or artifact_bindings["receipt"]["sha256"] != training_manifest["training_receipt"]["sha256"]
        or artifact_bindings["config"]["sha256"] != training_manifest["training_config"]["sha256"]
        or artifact_bindings["dataset"]["sha256"] != training_manifest["dataset_manifest"]["sha256"]
        or artifact_bindings["execution_spec"]["sha256"]
        != training_manifest["execution_spec"]["sha256"]
        or latest.get("checkpoint") != f"checkpoints/{training_receipt.get('final_checkpoint')}"
    ):
        _fail("training_completion_invalid")

    checkpoint = checkpoint_evidence.get("complete")
    checkpoint_artifacts = checkpoint_evidence.get("artifacts")
    checkpoint_adapter = (
        checkpoint_artifacts.get("adapter") if isinstance(checkpoint_artifacts, Mapping) else None
    )
    example_count = dataset_evidence["example_count"]
    checkpoint_epoch = checkpoint.get("epoch") if isinstance(checkpoint, Mapping) else None
    expected_epoch = (expected_steps - 1) // example_count
    expected_cursor = ((expected_steps - 1) % example_count) + 1
    expected_order = sorted(
        range(example_count),
        key=lambda index: hashlib.sha256(
            f"{train_seed}:{expected_epoch}:{index}".encode("ascii")
        ).digest(),
    )
    checkpoint_elapsed = (
        checkpoint.get("elapsed_training_s") if isinstance(checkpoint, Mapping) else None
    )
    receipt_elapsed = training_receipt.get("elapsed_training_s")
    checkpoint_loss = checkpoint.get("loss_trail") if isinstance(checkpoint, Mapping) else None
    loss_valid = _validate_loss_trail(
        checkpoint_loss,
        expected_steps=expected_steps,
        log_every=log_every,
    )
    detached_duration = detached_receipt.get("duration_s")
    detached_started = detached_receipt.get("started_at")
    detached_finished = detached_receipt.get("finished_at")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("schema") != TRAINING_CHECKPOINT_SCHEMA
        or checkpoint.get("checkpoint_id") != training_receipt.get("final_checkpoint")
        or checkpoint.get("step") != expected_steps
        or checkpoint_epoch != expected_epoch
        or checkpoint.get("cursor") != expected_cursor
        or checkpoint.get("order") != expected_order
        or checkpoint.get("invocation_count") != 1
        or checkpoint.get("sampler")
        != "sha256_stateless_epoch_permutation.v1"
        or checkpoint.get("stochastic_state") != "none_all_keys_explicit"
        or not isinstance(checkpoint_elapsed, (int, float))
        or isinstance(checkpoint_elapsed, bool)
        or not math.isfinite(float(checkpoint_elapsed))
        or checkpoint_elapsed <= 0.0
        or training_receipt.get("epoch") != checkpoint_epoch
        or training_receipt.get("cursor") != expected_cursor
        or training_receipt.get("invocation_count") != 1
        or training_receipt.get("loss_trail") != checkpoint_loss
        or not isinstance(receipt_elapsed, (int, float))
        or isinstance(receipt_elapsed, bool)
        or not math.isfinite(float(receipt_elapsed))
        or not 0.0 <= float(receipt_elapsed) - float(checkpoint_elapsed) <= 300.0
        or not loss_valid
        or not isinstance(detached_duration, (int, float))
        or isinstance(detached_duration, bool)
        or not math.isfinite(float(detached_duration))
        or float(detached_duration) <= 0.0
        or not isinstance(detached_started, (int, float))
        or isinstance(detached_started, bool)
        or not isinstance(detached_finished, (int, float))
        or isinstance(detached_finished, bool)
        or not math.isfinite(float(detached_started))
        or not math.isfinite(float(detached_finished))
        or abs(
            float(detached_finished)
            - float(detached_started)
            - float(detached_duration)
        )
        > 1.0
        or not 0.0
        <= float(detached_duration) - float(receipt_elapsed)
        <= min(7200.0, max_minutes * 60.0)
        or float(detached_duration) > float(detached_plan.get("timeout_s") or 0.0) + 2.0
        or checkpoint.get("config_sha256") != training_manifest.get("config_sha256")
        or checkpoint.get("dataset_sha256") != training_manifest.get("dataset_sha256")
        or checkpoint.get("execution_spec_sha256") != training_manifest.get("execution_spec_sha256")
        or not isinstance(checkpoint_adapter, Mapping)
        or checkpoint_adapter.get("sha256") != training_manifest["adapter"]["sha256"]
    ):
        _fail("training_final_checkpoint_mismatch")
    resource = _validate_resource_envelope(
        resource_envelope,
        wrapper=wrapper,
        trainer=trainer,
        wrapper_options=wrapper_options,
    )
    if read_canonical_json(resource_path, role="resource_envelope") != resource_envelope:
        _fail("resource_envelope_path_mismatch")
    identity_receipt = freeze_certificate.get("identity_receipt")
    frozen_model = freeze_certificate.get("model_identity")
    if (
        freeze_certificate.get("adapter_id") != adapter_id
        or freeze_certificate.get("artifacts") != source_inventory
        or freeze_certificate.get("content_root_sha256") != inventory_root_sha256(source_inventory)
        or not isinstance(identity_receipt, Mapping)
        or identity_receipt.get("adapter_sha256") != training_manifest["adapter"]["sha256"]
        or identity_receipt.get("training_receipt_sha256")
        != training_manifest["training_receipt"]["sha256"]
        or identity_receipt.get("training_config_sha256")
        != training_manifest["training_config"]["sha256"]
        or identity_receipt.get("training_completion_sha256")
        != artifact_bindings["completion"]["sha256"]
        or not isinstance(frozen_model, Mapping)
        or frozen_model.get("fingerprint") != training_manifest["base_checkpoint"]["fingerprint"]
    ):
        _fail("adapter_freeze_training_mismatch")
    return {
        "adapter_id": adapter_id,
        "acceptance_contract": contract_evidence,
        "adapter_sha256": training_manifest["adapter"]["sha256"],
        "base_checkpoint_sha256": training_manifest["base_checkpoint"]["fingerprint"],
        "content_root_sha256": freeze_certificate["content_root_sha256"],
        "detached_plan_sha256": detached_plan["plan_sha256"],
        "detached_receipt_sha256": detached_receipt["receipt_sha256"],
        "detached_attempt_event_count": detached_evidence["attempt_event_count"],
        "detached_attempt_journal_head_sha256": detached_evidence[
            "attempt_journal_head_sha256"
        ],
        "frozen_adapter": str(frozen_adapter),
        "launcher": launcher_identity,
        "latest_checkpoint": checkpoint_evidence["checkpoint"],
        "latest_checkpoint_complete_sha256": checkpoint_evidence["complete_binding"]["sha256"],
        "resource_limits": resource,
        "resource_envelope_sha256": artifact_bindings["resource"]["sha256"],
        "source_adapter": str(source_adapter),
        "steps": expected_steps,
        "training_dataset": dataset_evidence,
        "training_checkpoint_tensors": checkpoint_evidence["tensor_inventories"],
        "training_completion_sha256": artifact_bindings["completion"]["sha256"],
        "training_config_sha256": artifact_bindings["config"]["sha256"],
        "training_dataset_sha256": artifact_bindings["dataset"]["sha256"],
        "training_manifest_sha256": artifact_bindings["manifest"]["sha256"],
        "training_latest_sha256": artifact_bindings["latest"]["sha256"],
        "training_receipt_sha256": artifact_bindings["receipt"]["sha256"],
        "training_worktree": str(cwd),
        "training_wrapper_sha256": artifact_bindings["wrapper"]["sha256"],
        "training_trainer_sha256": artifact_bindings["trainer"]["sha256"],
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    detached_run = _canonical_path(
        args.detached_run, role="detached_training_run", kind="directory"
    )
    source_adapter = _canonical_path(
        args.source_adapter, role="training_source_adapter", kind="directory"
    )
    frozen_adapter = _canonical_path(
        args.frozen_adapter, role="training_frozen_adapter", kind="directory"
    )
    model = _canonical_path(args.model, role="training_model", kind="directory")
    contract_path = _canonical_path(
        args.acceptance_contract,
        role="training_acceptance_contract",
        kind="file",
    )
    acceptance_contract = read_canonical_json(
        contract_path, role="training_acceptance_contract"
    )
    acceptance_contract_binding = _file_binding(
        contract_path, role="training_acceptance_contract"
    )
    tokenizer_behavior_before = campaign_runner.model_behavior_bundle_identity(model)
    tokenizer = _load_bound_tokenizer(model)
    detached_plan, detached_receipt, detached_evidence = (
        _authoritative_detached_evidence(detached_run)
    )
    resource_path = _canonical_path(args.resource_envelope, role="resource_envelope", kind="file")
    resource = read_canonical_json(resource_path, role="resource_envelope")
    manifest_path = source_adapter / "recurrence_adapter_manifest.json"
    completion_path = source_adapter / "training_completion.json"
    latest_path = source_adapter / "latest.json"
    training_manifest = read_canonical_json(manifest_path, role="training_manifest")
    training_receipt, receipt_binding = _artifact_json(
        source_adapter, training_manifest, "training_receipt"
    )
    training_config, config_binding = _artifact_json(
        source_adapter, training_manifest, "training_config"
    )
    training_dataset, dataset_binding = _artifact_json(
        source_adapter, training_manifest, "dataset_manifest"
    )
    execution_spec, execution_binding = _artifact_json(
        source_adapter, training_manifest, "execution_spec"
    )
    training_completion = read_canonical_json(completion_path, role="training_completion")
    latest = read_canonical_json(latest_path, role="training_latest")
    checkpoint_evidence = _latest_checkpoint_evidence(source_adapter, latest)
    source_inventory = adapter_artifact_inventory(source_adapter, reject_unplanned=False)
    freeze_certificate = verify_adapter_freeze(frozen_adapter)
    expected_freeze_validators = {
        "campaign_runner_sha256": _file_binding(
            preparation.RUNNER_PATH, role="campaign_runner"
        )["sha256"],
        "freeze_contract_sha256": _file_binding(
            preparation.FREEZE_PATH, role="freeze_contract"
        )["sha256"],
        "identity_validator_sha256": _file_binding(
            preparation.IDENTITY_PATH, role="identity_validator"
        )["sha256"],
    }
    if freeze_certificate.get("validator_identity") != expected_freeze_validators:
        _fail("adapter_freeze_validator_identity_mismatch")
    model_identity, adapter_identity = campaign_runner._identity_material(
        SimpleNamespace(
            model=str(model),
            adapter=str(frozen_adapter),
            adapter_id=args.adapter_id,
            personality_adapter="trained",
        )
    )
    if (
        adapter_identity.get("identity_receipt")
        != freeze_certificate.get("identity_receipt")
        or preparation._selected_model_identity(model_identity)
        != freeze_certificate.get("model_identity")
        or model_identity.get("model_behavior_bundle") != tokenizer_behavior_before
    ):
        _fail("frozen_adapter_live_identity_mismatch")
    command = detached_plan["command"]
    separator = command.index("--")
    wrapper_options = _option_map(command[2:separator], role="training_wrapper")
    wrapper_path = Path(command[1])
    trainer_path = Path(wrapper_options["--trainer"])
    training = _validate_documents(
        detached_plan=detached_plan,
        detached_receipt=detached_receipt,
        detached_evidence=detached_evidence,
        acceptance_contract=acceptance_contract,
        acceptance_contract_binding=acceptance_contract_binding,
        resource_envelope=resource,
        resource_path=resource_path,
        training_manifest=training_manifest,
        training_receipt=training_receipt,
        training_config=training_config,
        training_dataset=training_dataset,
        execution_spec=execution_spec,
        training_completion=training_completion,
        latest=latest,
        checkpoint_evidence=checkpoint_evidence,
        artifact_bindings={
            "completion": _file_binding(completion_path, role="training_completion"),
            "config": config_binding,
            "dataset": dataset_binding,
            "execution_spec": execution_binding,
            "latest": _file_binding(latest_path, role="training_latest"),
            "manifest": _file_binding(manifest_path, role="training_manifest"),
            "receipt": receipt_binding,
            "resource": _file_binding(resource_path, role="resource_envelope"),
            "trainer": _file_binding(trainer_path, role="recurrence_trainer"),
            "wrapper": _file_binding(wrapper_path, role="training_wrapper"),
        },
        freeze_certificate=freeze_certificate,
        source_inventory=source_inventory,
        source_adapter=source_adapter,
        frozen_adapter=frozen_adapter,
        model=model,
        adapter_id=args.adapter_id,
        tokenizer=tokenizer,
    )
    tokenizer_behavior_after = campaign_runner.model_behavior_bundle_identity(model)
    training = {
        **training,
        "tokenizer_model_behavior_bundle_sha256": _stable_tokenizer_behavior_sha256(
            tokenizer_behavior_before,
            tokenizer_behavior_after,
        ),
    }
    material = {
        "schema": SCHEMA,
        "claim_scope": "terminal_training_and_immutable_adapter_identity_only",
        "training_complete": True,
        "immutable_freeze_verified": True,
        "ready_for_mechanics_smoke": True,
        "pilot_eligible": False,
        "reasoning_gain_proven": False,
        "frontier_gain_proven": False,
        "required_next_gate": "resident_32b_frozen_adapter_mechanics_smoke",
        "external_attestation_present": False,
        "external_trust_required_before_claim_campaign": True,
        "validator_identity": {
            "promotion_verifier_sha256": _file_binding(
                Path(__file__), role="promotion_verifier"
            )["sha256"],
            "detached_step_verifier_sha256": _file_binding(
                Path(detached.__file__), role="detached_step_verifier"
            )["sha256"],
            "freeze_contract_sha256": expected_freeze_validators[
                "freeze_contract_sha256"
            ],
            "identity_validator_sha256": expected_freeze_validators[
                "identity_validator_sha256"
            ],
            "training_acceptance_contract_file_sha256": acceptance_contract_binding[
                "sha256"
            ],
        },
        "training": training,
        "freeze_certificate_sha256": freeze_certificate["certificate_sha256"],
    }
    return {**material, "promotion_sha256": _sha256(material)}


def _write_create_or_verify(path: Path, document: dict[str, Any]) -> None:
    payload = canonical_json_bytes(document) + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_symlink() or read_stable_bytes(path, max_bytes=64 * 1024 * 1024) != payload:
            _fail("existing_promotion_receipt_differs")
        return
    preparation.write_canonical_exclusive(path, document)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detached-run", type=Path, required=True)
    parser.add_argument("--source-adapter", type=Path, required=True)
    parser.add_argument("--frozen-adapter", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter-id", required=True)
    parser.add_argument(
        "--acceptance-contract",
        type=Path,
        default=_DEFAULT_CONTRACT_PATH,
    )
    parser.add_argument("--resource-envelope", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        document = verify(args)
        _write_create_or_verify(args.output.expanduser().resolve(strict=False), document)
    except (
        TrainingPromotionError,
        detached.DetachedStepError,
        preparation.CampaignPreparationError,
        campaign_runner.CampaignProducerError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        print(
            f"verify_recurrence_training_promotion: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(canonical_json_bytes(document).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
