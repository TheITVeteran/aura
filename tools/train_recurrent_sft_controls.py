#!/usr/bin/env python3
"""Train equal-work negative controls for a completed recurrent-SFT candidate.

The process consumes only the historical candidate projection and terminal
training receipt. It has no evaluator argument and writes only quarantined
research adapters plus a hash-bound execution report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.execution_spec import (  # noqa: E402
    RLCExecutionSpec,
)
from core.learning.recurrent_sft_execution import (  # noqa: E402
    RecurrentSFTExecutionError,
    adapter_tensor_dict,
    assert_adapter_tensor_topology,
    wrap_recurrent_window,
)
from core.learning.recurrent_sft_falsification import (  # noqa: E402
    CONTROL_ARMS,
    RecurrentSFTFalsificationError,
    sha256_json,
    transform_control_rows,
)
from core.learning.structured_sft_research_authority import (  # noqa: E402
    RecurrentSFTTrainerConfig,
    StructuredSFTResearchAuthorityError,
    canonical_json_bytes,
    execution_spec_identity,
    small_model_identity,
    strict_json_bytes,
    validate_authority,
)
from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_bytes,
    ensure_private_directory,
)
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402

REPORT_SCHEMA = "aura.rlc.synthetic_recurrent_sft_control_training.v1"
PROJECTED_DATASET_SCHEMA = "aura.rlc.synthetic_recurrent_sft_projected_dataset.v1"
REFERENCE_COMPLETION_SCHEMA = "aura.rlc.synthetic_recurrent_sft_checkpoint.v1"
_MAX_DOCUMENT_BYTES = 256 * 1024 * 1024


class RecurrentSFTControlTrainingError(RuntimeError):
    """The equal-work control experiment could not proceed honestly."""


def _fail(code: str) -> Never:
    raise RecurrentSFTControlTrainingError(
        str(code or "recurrent_sft_control_training_failed")
    )


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        return strict_json_bytes(
            read_stable_bytes(
                path.expanduser().resolve(strict=True),
                max_bytes=_MAX_DOCUMENT_BYTES,
            ),
            role=f"control_training_{role}",
        )
    except (OSError, StructuredSFTResearchAuthorityError) as exc:
        raise RecurrentSFTControlTrainingError(
            f"control_training_{role}_unreadable"
        ) from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def control_source_paths() -> dict[str, Path]:
    """Return the complete Aura-owned source closure for this control process."""

    return {
        "authority": (
            REPO_ROOT / "core/learning/structured_sft_research_authority.py"
        ),
        "control_trainer": Path(__file__),
        "containment_launcher": (
            REPO_ROOT / "tools/launch_recurrent_sft_controls.py"
        ),
        "detached_supervisor": REPO_ROOT / "tools/run_detached_step.py",
        "execution_spec": (
            REPO_ROOT / "core/brain/llm/latent_cortex/execution_spec.py"
        ),
        "falsification": (
            REPO_ROOT / "core/learning/recurrent_sft_falsification.py"
        ),
        "file_read_gateway": REPO_ROOT / "core/runtime/file_read_gateway.py",
        "memory_guard": REPO_ROOT / "core/runtime/mlx_memory_guard.py",
        "model_lane": REPO_ROOT / "core/runtime/model_lane_control.py",
        "recurrence_adapter": (
            REPO_ROOT / "core/brain/llm/latent_cortex/recurrence_adapter.py"
        ),
        "recurrence_identity": (
            REPO_ROOT
            / "core/brain/llm/latent_cortex/recurrence_adapter_identity_v2.py"
        ),
        "recurrence_objective": (
            REPO_ROOT / "core/learning/recurrence_native_objective_v2.py"
        ),
        "recurrent_sft_execution": (
            REPO_ROOT / "core/learning/recurrent_sft_execution.py"
        ),
    }


def control_source_closure() -> dict[str, Any]:
    """Bind exactly the control process's declared Aura-owned source roles."""

    paths = control_source_paths()
    expected_roles = {
        "authority",
        "control_trainer",
        "containment_launcher",
        "detached_supervisor",
        "execution_spec",
        "falsification",
        "file_read_gateway",
        "memory_guard",
        "model_lane",
        "recurrence_adapter",
        "recurrence_identity",
        "recurrence_objective",
        "recurrent_sft_execution",
    }
    if set(paths) != expected_roles:
        _fail("control_training_source_roles_invalid")
    records: list[dict[str, Any]] = []
    for role in sorted(expected_roles):
        lexical = paths[role].expanduser()
        if lexical.is_symlink():
            _fail("control_training_source_symlink_rejected")
        try:
            path = lexical.resolve(strict=True)
        except OSError as exc:
            raise RecurrentSFTControlTrainingError(
                "control_training_source_unreadable"
            ) from exc
        if not path.is_file():
            _fail("control_training_source_file_invalid")
        payload = read_stable_bytes(path, max_bytes=_MAX_DOCUMENT_BYTES)
        records.append(
            {
                "role": role,
                "path": str(path),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    body = {
        "schema": "aura.rlc.synthetic_recurrent_sft_control_source_closure.v1",
        "files": records,
    }
    return {**body, "closure_sha256": sha256_json(body)}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _trainer_config(raw: Mapping[str, Any]) -> RecurrentSFTTrainerConfig:
    fixed_fields = {
        "schema",
        "training_mode",
        "sampler",
        "loss",
        "adapter_activation",
        "ordinary_lexical_activation",
        "validation_scope",
    }
    material = {key: value for key, value in raw.items() if key not in fixed_fields}
    targets = material.get("lora_targets")
    if not isinstance(targets, list):
        _fail("control_training_lora_targets_invalid")
    material["lora_targets"] = tuple(targets)
    try:
        config = RecurrentSFTTrainerConfig(**material)
    except (TypeError, StructuredSFTResearchAuthorityError) as exc:
        raise RecurrentSFTControlTrainingError(
            "control_training_trainer_config_invalid"
        ) from exc
    if config.to_dict() != dict(raw):
        _fail("control_training_trainer_config_drift")
    return config


def _projected_dataset(raw: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema",
        "candidate_identity_sha256",
        "train",
        "validation",
        "holdout",
        "verified_replay",
        "dataset_sha256",
    }
    if (
        set(raw) != fields
        or raw.get("schema") != PROJECTED_DATASET_SCHEMA
        or raw.get("holdout") is not None
        or raw.get("verified_replay") is not None
        or not isinstance(raw.get("train"), list)
        or not raw["train"]
        or not isinstance(raw.get("validation"), list)
        or not raw["validation"]
    ):
        _fail("control_training_projected_dataset_invalid")
    body = dict(raw)
    observed = body.pop("dataset_sha256", None)
    if observed != sha256_json(body):
        _fail("control_training_projected_dataset_commitment_mismatch")
    return json.loads(canonical_json_bytes(raw))


def _reference_checkpoint(
    raw: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    projected_dataset: Mapping[str, Any],
) -> dict[str, Any]:
    adapter = raw.get("adapter")
    optimizer = raw.get("optimizer")
    order = raw.get("order")
    if (
        raw.get("schema") != REFERENCE_COMPLETION_SCHEMA
        or raw.get("terminal") is not True
        or raw.get("last_step_committed") is not True
        or raw.get("authority_sha256") != authority["authority_sha256"]
        or raw.get("dataset_sha256") != projected_dataset["dataset_sha256"]
        or raw.get("model_identity_sha256") != authority["model"]["identity_sha256"]
        or raw.get("execution_spec_sha256")
        != authority["execution_spec"]["semantic_sha256"]
        or raw.get("trainer_config_sha256") != sha256_json(authority["trainer"])
        or type(raw.get("step")) is not int
        or raw["step"] < 1
        or raw.get("optimizer_updates") != raw["step"]
        or raw.get("epoch") != 0
        or raw.get("cursor") != raw["step"]
        or not isinstance(order, list)
        or len(order) != len(projected_dataset["train"])
        or sorted(order) != list(range(len(order)))
        or raw["step"] > len(order)
        or not isinstance(adapter, Mapping)
        or not isinstance(optimizer, Mapping)
        or set(adapter) != {"path", "sha256", "size_bytes"}
        or set(optimizer) != {"path", "sha256", "size_bytes"}
        or any(
            not isinstance(binding.get("path"), str)
            or Path(binding["path"]).name != binding["path"]
            or not _is_sha256(binding.get("sha256"))
            or type(binding.get("size_bytes")) is not int
            or binding["size_bytes"] < 1
            for binding in (adapter, optimizer)
        )
    ):
        _fail("control_training_reference_checkpoint_invalid")
    return json.loads(canonical_json_bytes(raw))


def _token_surfaces(tokenizer: Any, rows: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        _fail("control_training_tokenizer_decode_missing")
    token_ids = sorted(
        {
            int(token)
            for row in rows
            for token in row["answer_tokens"]
        }
    )
    surfaces: dict[int, str] = {}
    for token in token_ids:
        surface = decode([token], skip_special_tokens=False)
        if not isinstance(surface, str):
            _fail("control_training_token_surface_invalid")
        surfaces[token] = surface
    return surfaces


def _neutral_token_id(tokenizer: Any) -> int:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        _fail("control_training_tokenizer_encode_missing")
    tokens = encode("x", add_special_tokens=False)
    if (
        not isinstance(tokens, Sequence)
        or isinstance(tokens, (str, bytes, bytearray))
        or len(tokens) != 1
        or type(tokens[0]) is not int
        or tokens[0] < 0
    ):
        _fail("control_training_neutral_token_not_atomic")
    return int(tokens[0])


def _save_adapter(
    output: Path,
    tensors: Mapping[str, Any],
) -> dict[str, Any]:
    import mlx.core as mx

    scratch = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    if output.exists() or output.is_symlink():
        _fail("control_training_adapter_output_exists")
    try:
        mx.save_safetensors(str(scratch), dict(tensors))
        payload = scratch.read_bytes()
        atomic_write_bytes(output, payload, mode=0o600)
    finally:
        try:
            scratch.unlink(missing_ok=True)
        except OSError:
            pass
    return {
        "filename": output.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _tensor_fingerprint(tensors: Mapping[str, Any]) -> str:
    import numpy as np

    digest = hashlib.sha256()
    for name in sorted(tensors):
        array = np.asarray(tensors[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json_bytes(list(array.shape)))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_checkpoint_artifacts(
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
) -> None:
    generation = checkpoint_path.parent
    for role in ("adapter", "optimizer"):
        binding = checkpoint[role]
        path = (generation / binding["path"]).resolve(strict=True)
        if path.parent != generation:
            _fail(f"control_training_reference_{role}_path_escape")
        if (
            path.stat().st_size != binding["size_bytes"]
            or _sha256_file(path) != binding["sha256"]
        ):
            _fail(f"control_training_reference_{role}_commitment_mismatch")


def _run(arguments: argparse.Namespace) -> int:
    authority_raw = _read_json(arguments.reference_authority, role="authority")
    issued_at = authority_raw.get("issued_at_unix")
    if type(issued_at) is not int:
        _fail("control_training_authority_issue_time_invalid")
    authority = validate_authority(
        authority_raw,
        expected_authority_sha256=arguments.expected_authority_sha256,
        now_unix=issued_at,
    )
    config = _trainer_config(authority["trainer"])
    projected = _projected_dataset(
        _read_json(arguments.projected_dataset, role="projected_dataset")
    )
    reference_checkpoint_path = arguments.reference_checkpoint.expanduser().resolve(
        strict=True
    )
    if (
        _sha256_file(reference_checkpoint_path)
        != arguments.expected_reference_checkpoint_sha256
    ):
        _fail("control_training_reference_checkpoint_sha256_mismatch")
    reference = _reference_checkpoint(
        _read_json(reference_checkpoint_path, role="reference_checkpoint"),
        authority=authority,
        projected_dataset=projected,
    )
    _validate_checkpoint_artifacts(reference_checkpoint_path, reference)
    execution_raw = _read_json(arguments.execution_spec, role="execution_spec")
    if execution_spec_identity(execution_raw) != authority["execution_spec"]:
        _fail("control_training_execution_spec_drift")
    spec = RLCExecutionSpec.from_dict(execution_raw)
    model_identity = small_model_identity(arguments.model_dir)
    if model_identity != authority["model"]:
        _fail("control_training_model_identity_drift")
    sources = control_source_closure()
    if sources["closure_sha256"] != arguments.expected_source_closure_sha256:
        _fail("control_training_source_closure_drift")

    out_dir = ensure_private_directory(arguments.out_dir.expanduser())
    report_path = out_dir / "control_training_report.json"
    if report_path.exists() or report_path.is_symlink():
        _fail("control_training_report_exists")

    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm import load

    from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
        full_weight_checkpoint_identity,
    )
    from core.learning.recurrence_native_objective_v2 import (
        exact_adjoint_live_path_value_and_grad,
    )
    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane

    train_rows = list(projected["train"])
    sample_indices = list(reference["order"][: reference["step"]])
    sample_token_counts = [
        train_rows[index]["full_token_count"] for index in sample_indices
    ]
    started = time.monotonic()
    with (
        standalone_model_lane(
            owner_id=f"synthetic-recurrent-sft-controls:{out_dir.name}",
            model_path=str(arguments.model_dir),
            purpose="training",
            preemptible=False,
            metadata={
                "tool": "train_recurrent_sft_controls",
                "reference_authority_sha256": authority["authority_sha256"],
                "production_effect": False,
                "evaluator_access": False,
            },
        ) as lease,
        mlx_memory_envelope(fraction=config.memory_fraction) as envelope,
    ):
        if getattr(lease, "active", False) is not True:
            _fail("control_training_model_lane_not_active")
        base_weights_before = full_weight_checkpoint_identity(arguments.model_dir)
        mx.random.seed(config.seed)
        model, tokenizer = load(str(arguments.model_dir))
        wrapped = wrap_recurrent_window(
            model,
            spec=spec,
            lora_rank=config.lora_rank,
            lora_dropout=config.lora_dropout,
            lora_scale=config.lora_scale,
            lora_targets=config.lora_targets,
        )
        initial_adapter = adapter_tensor_dict(model)
        mx.eval(initial_adapter)
        initial_adapter_sha256 = _tensor_fingerprint(initial_adapter)
        surfaces = _token_surfaces(tokenizer, train_rows)
        neutral_token = _neutral_token_id(tokenizer)
        arm_reports: dict[str, Any] = {}
        for arm in CONTROL_ARMS:
            transformed = transform_control_rows(
                train_rows,
                arm=arm,
                seed=config.seed,
                token_surfaces=surfaces,
                neutral_token_id=neutral_token,
            )
            model.load_weights(list(initial_adapter.items()), strict=False)
            mx.eval(model.trainable_parameters())
            starting_adapter_sha256 = _tensor_fingerprint(
                adapter_tensor_dict(model)
            )
            if starting_adapter_sha256 != initial_adapter_sha256:
                _fail(f"control_training_{arm}_initialization_drift")
            optimizer = optim.AdamW(
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            optimizer.init(model.trainable_parameters())
            arm_started = time.monotonic()
            loss_trail: list[float] = []
            branch_cosine_trail: list[list[float]] = []
            observed_token_counts: list[int] = []
            for index in sample_indices:
                row = transformed.rows[index]
                loss, gradients, _base_loss, branch_cosines = (
                    exact_adjoint_live_path_value_and_grad(
                        model,
                        row["prompt_tokens"],
                        row["answer_tokens"],
                        spec=spec,
                    )
                )
                if not math.isfinite(loss) or loss < 0.0:
                    _fail(f"control_training_{arm}_loss_nonfinite")
                optimizer.update(model, gradients)
                mx.eval(model.trainable_parameters(), optimizer.state)
                loss_trail.append(round(float(loss), 12))
                branch_cosine_trail.append(
                    [round(float(value), 12) for value in branch_cosines]
                )
                observed_token_counts.append(row["full_token_count"])
                del gradients
                mx.clear_cache()
                envelope.reclaim(force=True)
            if observed_token_counts != sample_token_counts:
                _fail(f"control_training_{arm}_sample_workload_drift")
            adapter = adapter_tensor_dict(model)
            assert_adapter_tensor_topology(initial_adapter, adapter)
            adapter_binding = _save_adapter(
                out_dir / f"{arm}.safetensors",
                adapter,
            )
            arm_body = {
                "arm": arm,
                "transform": transformed.manifest,
                "adapter": adapter_binding,
                "starting_adapter_sha256": starting_adapter_sha256,
                "optimizer": "AdamW",
                "optimizer_updates": len(sample_indices),
                "sample_indices": sample_indices,
                "sample_token_counts": sample_token_counts,
                "sample_token_budget": sum(sample_token_counts),
                "loss_trail": loss_trail,
                "branch_cosine_trail": branch_cosine_trail,
                "duration_s": round(time.monotonic() - arm_started, 6),
            }
            arm_reports[arm] = {
                **arm_body,
                "arm_report_sha256": sha256_json(arm_body),
            }
            del optimizer
            mx.clear_cache()
            envelope.reclaim(force=True)
        base_weights_after = full_weight_checkpoint_identity(arguments.model_dir)
        if base_weights_after != base_weights_before:
            _fail("control_training_base_weights_changed")

    projected_path = arguments.projected_dataset.expanduser().resolve(strict=True)
    body = {
        "schema": REPORT_SCHEMA,
        "status": "completed_equal_work_negative_controls",
        "reference_authority_sha256": authority["authority_sha256"],
        "reference_checkpoint_sha256": _sha256_file(reference_checkpoint_path),
        "reference_adapter_sha256": reference["adapter"]["sha256"],
        "projected_dataset_sha256": projected["dataset_sha256"],
        "projected_dataset_file_sha256": _sha256_file(projected_path),
        "model_identity_sha256": model_identity["identity_sha256"],
        "execution_spec_sha256": authority["execution_spec"]["semantic_sha256"],
        "source_closure": sources,
        "trainer_config_sha256": sha256_json(authority["trainer"]),
        "wrapped_projections": wrapped,
        "initialization_seed": config.seed,
        "initial_adapter_sha256": initial_adapter_sha256,
        "identical_initial_adapter_for_all_controls": True,
        "reference_optimizer_updates": reference["optimizer_updates"],
        "control_optimizer_updates": {
            arm: arm_reports[arm]["optimizer_updates"] for arm in CONTROL_ARMS
        },
        "equal_sample_order": True,
        "equal_per_step_token_counts": True,
        "equal_optimizer_and_hyperparameters": True,
        "base_weights_unchanged": True,
        "evaluator_access": False,
        "production_effect": False,
        "promotion_allowed": False,
        "duration_s": round(time.monotonic() - started, 6),
        "arms": arm_reports,
        "claims_not_supported": [
            "heldout_transfer",
            "reasoning_gain",
            "frontier_performance",
            "resident_32b_result",
            "production_promotion",
            "wow_signal",
        ],
    }
    report = {**body, "report_sha256": sha256_json(body)}
    atomic_write_bytes(report_path, canonical_json_bytes(report), mode=0o600)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-authority", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-reference-checkpoint-sha256", required=True)
    parser.add_argument("--projected-dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--execution-spec", type=Path, required=True)
    parser.add_argument("--expected-source-closure-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if (
        not _is_sha256(arguments.expected_authority_sha256)
        or not _is_sha256(arguments.expected_reference_checkpoint_sha256)
        or not _is_sha256(arguments.expected_source_closure_sha256)
    ):
        _parser().error("expected authority SHA-256 is invalid")
    try:
        return _run(arguments)
    except (
        ImportError,
        MemoryError,
        OSError,
        RecurrentSFTControlTrainingError,
        RecurrentSFTExecutionError,
        RecurrentSFTFalsificationError,
        StructuredSFTResearchAuthorityError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {
                    "schema": f"{REPORT_SCHEMA}.error",
                    "ok": False,
                    "reason": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
