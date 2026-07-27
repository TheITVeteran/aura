#!/usr/bin/env python3
"""Evaluate recurrent-SFT transfer under independent holdout custody."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.execution_spec import (  # noqa: E402
    RLCExecutionSpec,
)
from core.learning.recurrent_sft_evaluation import (  # noqa: E402
    EVALUATION_SCHEMA,
    RecurrentSFTEvaluationError,
    build_regression_canary_rows,
    evaluator_holdout_rows,
    regression_canary_verdict,
    score_forward,
    sha256_bytes,
    strict_json_bytes,
    validate_control_report,
)
from core.learning.recurrent_sft_execution import (  # noqa: E402
    RecurrentSFTExecutionError,
    adapter_tensor_dict,
    assert_adapter_tensor_topology,
    project_chat_rows,
    wrap_recurrent_window,
)
from core.learning.recurrent_sft_falsification import (  # noqa: E402
    BASE_ARM,
    CONTROL_ARMS,
    TRAINED_ARM,
    RecurrentSFTFalsificationError,
    build_falsification_verdict,
    sha256_json,
)
from core.learning.structured_sft import (  # noqa: E402
    STRUCTURED_SFT_CANDIDATE_FILES,
    STRUCTURED_SFT_EVALUATOR_FILES,
)
from core.learning.structured_sft_research_authority import (  # noqa: E402
    RecurrentSFTTrainerConfig,
    StructuredSFTResearchAuthorityError,
    canonical_json_bytes,
    execution_spec_identity,
    small_model_identity,
    validate_authority,
)
from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_bytes,
    ensure_private_directory,
)
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402

REPORT_SCHEMA = f"{EVALUATION_SCHEMA}.report"
REFERENCE_COMPLETION_SCHEMA = "aura.rlc.synthetic_recurrent_sft_checkpoint.v1"
_MAX_FILE_BYTES = 512 * 1024 * 1024


class RecurrentSFTFalsificationEvaluationError(RuntimeError):
    """The independent recurrent-SFT evaluation could not proceed honestly."""


def _fail(code: str) -> Never:
    raise RecurrentSFTFalsificationEvaluationError(
        str(code or "recurrent_sft_falsification_evaluation_failed")
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_bytes(path: Path, *, role: str) -> bytes:
    try:
        return read_stable_bytes(
            path.expanduser().resolve(strict=True),
            max_bytes=_MAX_FILE_BYTES,
        )
    except OSError as exc:
        raise RecurrentSFTFalsificationEvaluationError(
            f"recurrent_sft_evaluation_{role}_unreadable"
        ) from exc


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    return strict_json_bytes(_read_bytes(path, role=role), role=role)


def _artifact_binding(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _trainer_config(raw: Mapping[str, Any]) -> RecurrentSFTTrainerConfig:
    fixed = {
        "schema",
        "training_mode",
        "sampler",
        "loss",
        "adapter_activation",
        "ordinary_lexical_activation",
        "validation_scope",
    }
    material = {key: value for key, value in raw.items() if key not in fixed}
    targets = material.get("lora_targets")
    if not isinstance(targets, list):
        _fail("recurrent_sft_evaluation_lora_targets_invalid")
    material["lora_targets"] = tuple(targets)
    try:
        config = RecurrentSFTTrainerConfig(**material)
    except (TypeError, StructuredSFTResearchAuthorityError) as exc:
        raise RecurrentSFTFalsificationEvaluationError(
            "recurrent_sft_evaluation_trainer_config_invalid"
        ) from exc
    if config.to_dict() != dict(raw):
        _fail("recurrent_sft_evaluation_trainer_config_drift")
    return config


def evaluation_source_paths() -> dict[str, Path]:
    return {
        "authority": (
            REPO_ROOT / "core/learning/structured_sft_research_authority.py"
        ),
        "evaluation_contract": (
            REPO_ROOT / "core/learning/recurrent_sft_evaluation.py"
        ),
        "evaluator": Path(__file__),
        "evaluator_launcher": (
            REPO_ROOT / "tools/launch_recurrent_sft_falsification.py"
        ),
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
        "structured_sft": REPO_ROOT / "core/learning/structured_sft.py",
    }


def evaluation_source_closure() -> dict[str, Any]:
    paths = evaluation_source_paths()
    expected = {
        "authority",
        "evaluation_contract",
        "evaluator",
        "evaluator_launcher",
        "execution_spec",
        "falsification",
        "file_read_gateway",
        "memory_guard",
        "model_lane",
        "recurrence_adapter",
        "recurrence_identity",
        "recurrence_objective",
        "recurrent_sft_execution",
        "structured_sft",
    }
    if set(paths) != expected:
        _fail("recurrent_sft_evaluation_source_roles_invalid")
    files: list[dict[str, Any]] = []
    for role in sorted(expected):
        lexical = paths[role].expanduser()
        if lexical.is_symlink():
            _fail("recurrent_sft_evaluation_source_symlink_rejected")
        path = lexical.resolve(strict=True)
        payload = _read_bytes(path, role=f"source_{role}")
        files.append(
            {
                "role": role,
                "path": str(path),
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )
    body = {
        "schema": f"{EVALUATION_SCHEMA}.source_closure",
        "files": files,
    }
    return {**body, "closure_sha256": sha256_json(body)}


def _candidate_and_evaluator_artifacts(
    candidate_dir: Path,
    evaluator_dir: Path,
) -> tuple[dict[str, bytes], dict[str, bytes], dict[str, Any]]:
    candidate = candidate_dir.expanduser().resolve(strict=True)
    evaluator = evaluator_dir.expanduser().resolve(strict=True)
    if (
        candidate.is_symlink()
        or evaluator.is_symlink()
        or not candidate.is_dir()
        or not evaluator.is_dir()
        or candidate == evaluator
    ):
        _fail("recurrent_sft_evaluation_custody_roots_invalid")
    candidate_artifacts = {
        name: _read_bytes(candidate / name, role=f"candidate_{name}")
        for name in STRUCTURED_SFT_CANDIDATE_FILES
    }
    evaluator_artifacts = {
        name: _read_bytes(evaluator / name, role=f"evaluator_{name}")
        for name in STRUCTURED_SFT_EVALUATOR_FILES
    }
    rows, custody = evaluator_holdout_rows(
        candidate_artifacts,
        evaluator_artifacts,
    )
    bindings = {
        "candidate": {
            name: _artifact_binding(candidate / name, payload)
            for name, payload in sorted(candidate_artifacts.items())
        },
        "evaluator": {
            name: _artifact_binding(evaluator / name, payload)
            for name, payload in sorted(evaluator_artifacts.items())
        },
        "custody": custody,
    }
    return candidate_artifacts, evaluator_artifacts, {
        "rows": rows,
        "bindings": bindings,
    }


def _reference_adapter(
    checkpoint_path: Path,
    *,
    expected_checkpoint_sha256: str,
    authority: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    payload = _read_bytes(checkpoint_path, role="reference_checkpoint")
    if sha256_bytes(payload) != expected_checkpoint_sha256:
        _fail("recurrent_sft_evaluation_reference_checkpoint_sha256_mismatch")
    checkpoint = strict_json_bytes(payload, role="reference_checkpoint")
    adapter = checkpoint.get("adapter")
    if (
        checkpoint.get("schema") != REFERENCE_COMPLETION_SCHEMA
        or checkpoint.get("terminal") is not True
        or checkpoint.get("last_step_committed") is not True
        or checkpoint.get("authority_sha256") != authority["authority_sha256"]
        or checkpoint.get("model_identity_sha256")
        != authority["model"]["identity_sha256"]
        or checkpoint.get("execution_spec_sha256")
        != authority["execution_spec"]["semantic_sha256"]
        or not isinstance(adapter, Mapping)
        or set(adapter) != {"path", "sha256", "size_bytes"}
        or not isinstance(adapter.get("path"), str)
        or Path(adapter["path"]).name != adapter["path"]
        or not _is_sha256(adapter.get("sha256"))
        or type(adapter.get("size_bytes")) is not int
        or adapter["size_bytes"] < 1
    ):
        _fail("recurrent_sft_evaluation_reference_checkpoint_invalid")
    path = checkpoint_path.parent / adapter["path"]
    adapter_payload = _read_bytes(path, role="trained_adapter")
    if (
        len(adapter_payload) != adapter["size_bytes"]
        or sha256_bytes(adapter_payload) != adapter["sha256"]
    ):
        _fail("recurrent_sft_evaluation_trained_adapter_mismatch")
    return path, {
        "checkpoint": _artifact_binding(checkpoint_path, payload),
        "adapter": _artifact_binding(path, adapter_payload),
    }


def _control_adapters(
    report_path: Path,
    *,
    expected_report_sha256: str,
    authority: Mapping[str, Any],
    expected_reference_checkpoint_sha256: str,
) -> tuple[dict[str, Path], dict[str, Any]]:
    payload = _read_bytes(report_path, role="control_report")
    report = strict_json_bytes(payload, role="control_report")
    bindings = validate_control_report(
        report,
        report_file_sha256=sha256_bytes(payload),
        expected_report_file_sha256=expected_report_sha256,
        expected_authority_sha256=authority["authority_sha256"],
        expected_reference_checkpoint_sha256=expected_reference_checkpoint_sha256,
        expected_model_identity_sha256=authority["model"]["identity_sha256"],
        expected_execution_spec_sha256=authority["execution_spec"]["semantic_sha256"],
    )
    paths: dict[str, Path] = {}
    adapter_bindings: dict[str, Any] = {}
    for arm in CONTROL_ARMS:
        binding = bindings[arm]
        path = report_path.parent / binding["filename"]
        payload = _read_bytes(path, role=f"control_adapter_{arm}")
        if (
            len(payload) != binding["size_bytes"]
            or sha256_bytes(payload) != binding["sha256"]
        ):
            _fail(f"recurrent_sft_evaluation_control_adapter_{arm}_mismatch")
        paths[arm] = path
        adapter_bindings[arm] = _artifact_binding(path, payload)
    return paths, {
        "report": _artifact_binding(report_path, _read_bytes(report_path, role="control_report")),
        "adapters": adapter_bindings,
        "report_sha256": report["report_sha256"],
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
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_adapter(
    model: Any,
    path: Path,
    *,
    expected_topology: Mapping[str, Any],
) -> str:
    import mlx.core as mx

    tensors = mx.load(str(path))
    if not isinstance(tensors, Mapping):
        _fail("recurrent_sft_evaluation_adapter_tensor_mapping_invalid")
    assert_adapter_tensor_topology(expected_topology, tensors)
    model.load_weights(list(tensors.items()), strict=False)
    mx.eval(model.trainable_parameters())
    observed = adapter_tensor_dict(model)
    assert_adapter_tensor_topology(tensors, observed)
    expected_fingerprint = _tensor_fingerprint(tensors)
    if _tensor_fingerprint(observed) != expected_fingerprint:
        _fail("recurrent_sft_evaluation_adapter_load_mismatch")
    return expected_fingerprint


def _score_rows(
    model: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    spec: RLCExecutionSpec,
    disable_adapter: bool,
    envelope: Any,
) -> list[dict[str, Any]]:
    import mlx.core as mx

    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_disabled,
    )
    from core.learning.recurrence_native_objective_v2 import (
        live_path_forward,
    )

    observations: list[dict[str, Any]] = []
    context = recurrence_adapter_disabled() if disable_adapter else nullcontext()
    with context:
        for row in rows:
            forward = live_path_forward(
                model,
                row["prompt_tokens"],
                row["answer_tokens"],
                spec=spec,
            )
            loss, top1 = score_forward(forward, row["answer_tokens"])
            observations.append(
                {
                    "example_id": row["example_id"],
                    "family": row["family"],
                    "loss": round(loss, 12),
                    "target_top1": top1,
                    "generated_correct": None,
                }
            )
            del forward
            mx.clear_cache()
            envelope.reclaim(force=True)
    return observations


def _ordinary_lexical_hash(model: Any, tokenizer: Any) -> str:
    import mlx.core as mx
    import numpy as np

    from core.brain.llm.latent_cortex.recurrence_adapter import (
        current_recurrence_adapter_scope,
    )

    if current_recurrence_adapter_scope() is not None:
        _fail("recurrent_sft_evaluation_lexical_scope_leak")
    tokens = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "You are Aura. Answer directly and truthfully.",
            },
            {"role": "user", "content": "State whether two plus two equals four."},
        ],
        add_generation_prompt=True,
        return_dict=False,
    )
    logits = model(mx.array([tokens]))
    mx.eval(logits)
    array = np.asarray(logits)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _run(arguments: argparse.Namespace) -> int:
    if not all(
        _is_sha256(value)
        for value in (
            arguments.expected_authority_sha256,
            arguments.expected_reference_checkpoint_sha256,
            arguments.expected_control_report_sha256,
            arguments.expected_source_closure_sha256,
        )
    ):
        _fail("recurrent_sft_evaluation_expected_sha256_invalid")
    authority_raw = _read_json(arguments.reference_authority, role="authority")
    issued_at = authority_raw.get("issued_at_unix")
    if type(issued_at) is not int:
        _fail("recurrent_sft_evaluation_authority_issue_time_invalid")
    authority = validate_authority(
        authority_raw,
        expected_authority_sha256=arguments.expected_authority_sha256,
        now_unix=issued_at,
    )
    config = _trainer_config(authority["trainer"])
    spec_raw = _read_json(arguments.execution_spec, role="execution_spec")
    if execution_spec_identity(spec_raw) != authority["execution_spec"]:
        _fail("recurrent_sft_evaluation_execution_spec_drift")
    spec = RLCExecutionSpec.from_dict(spec_raw)
    model_identity = small_model_identity(arguments.model_dir)
    if model_identity != authority["model"]:
        _fail("recurrent_sft_evaluation_model_identity_drift")
    sources = evaluation_source_closure()
    if sources["closure_sha256"] != arguments.expected_source_closure_sha256:
        _fail("recurrent_sft_evaluation_source_closure_drift")
    _candidate, _evaluator, custody_material = _candidate_and_evaluator_artifacts(
        arguments.candidate_dir,
        arguments.evaluator_dir,
    )
    holdout_rows = custody_material["rows"]
    trained_adapter_path, trained_binding = _reference_adapter(
        arguments.reference_checkpoint.expanduser().resolve(strict=True),
        expected_checkpoint_sha256=arguments.expected_reference_checkpoint_sha256,
        authority=authority,
    )
    control_paths, control_bindings = _control_adapters(
        arguments.control_report.expanduser().resolve(strict=True),
        expected_report_sha256=arguments.expected_control_report_sha256,
        authority=authority,
        expected_reference_checkpoint_sha256=(
            arguments.expected_reference_checkpoint_sha256
        ),
    )
    out_dir = ensure_private_directory(arguments.out_dir.expanduser())
    report_path = out_dir / "falsification_evaluation_report.json"
    if report_path.exists() or report_path.is_symlink():
        _fail("recurrent_sft_evaluation_report_exists")

    import mlx.core as mx
    from mlx_lm import load

    from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
        full_weight_checkpoint_identity,
    )
    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane

    started = time.monotonic()
    with (
        standalone_model_lane(
            owner_id=f"synthetic-recurrent-sft-evaluator:{out_dir.name}",
            model_path=str(arguments.model_dir),
            purpose="evaluation",
            preemptible=False,
            metadata={
                "tool": "evaluate_recurrent_sft_falsification",
                "evaluator_custody": True,
                "production_effect": False,
            },
        ) as lease,
        mlx_memory_envelope(fraction=config.memory_fraction) as envelope,
    ):
        if getattr(lease, "active", False) is not True:
            _fail("recurrent_sft_evaluation_model_lane_not_active")
        base_before = full_weight_checkpoint_identity(arguments.model_dir)
        model, tokenizer = load(str(arguments.model_dir))
        wrapped = wrap_recurrent_window(
            model,
            spec=spec,
            lora_rank=config.lora_rank,
            lora_dropout=config.lora_dropout,
            lora_scale=config.lora_scale,
            lora_targets=config.lora_targets,
        )
        topology = adapter_tensor_dict(model)
        mx.eval(topology)
        projected_holdout = project_chat_rows(
            holdout_rows,
            tokenizer=tokenizer,
            max_seq_length=config.max_seq_length,
        )
        projected_canaries = project_chat_rows(
            build_regression_canary_rows(),
            tokenizer=tokenizer,
            max_seq_length=config.max_seq_length,
        )
        observations: dict[str, list[dict[str, Any]]] = {}
        lexical_hashes: dict[str, str] = {}
        adapter_fingerprints: dict[str, str | None] = {BASE_ARM: None}
        observations[BASE_ARM] = _score_rows(
            model,
            projected_holdout,
            spec=spec,
            disable_adapter=True,
            envelope=envelope,
        )
        base_canaries = _score_rows(
            model,
            projected_canaries,
            spec=spec,
            disable_adapter=True,
            envelope=envelope,
        )
        lexical_hashes[BASE_ARM] = _ordinary_lexical_hash(model, tokenizer)

        arm_paths = {
            TRAINED_ARM: trained_adapter_path,
            **control_paths,
        }
        trained_canaries: list[dict[str, Any]] | None = None
        for arm in (TRAINED_ARM, *CONTROL_ARMS):
            adapter_fingerprints[arm] = _load_adapter(
                model,
                arm_paths[arm],
                expected_topology=topology,
            )
            observations[arm] = _score_rows(
                model,
                projected_holdout,
                spec=spec,
                disable_adapter=False,
                envelope=envelope,
            )
            lexical_hashes[arm] = _ordinary_lexical_hash(model, tokenizer)
            if arm == TRAINED_ARM:
                trained_canaries = _score_rows(
                    model,
                    projected_canaries,
                    spec=spec,
                    disable_adapter=False,
                    envelope=envelope,
                )
        if trained_canaries is None:
            _fail("recurrent_sft_evaluation_trained_canaries_missing")
        base_after = full_weight_checkpoint_identity(arguments.model_dir)
        if base_after != base_before:
            _fail("recurrent_sft_evaluation_base_weights_changed")

    falsification = build_falsification_verdict(observations)
    canary_verdict = regression_canary_verdict(
        base_canaries,
        trained_canaries,
    )
    lexical_invariance = len(set(lexical_hashes.values())) == 1
    all_gates_passed = (
        falsification["heldout_transfer_proven"]
        and canary_verdict["passed"]
        and lexical_invariance
    )
    body = {
        "schema": REPORT_SCHEMA,
        "status": (
            "small_checkpoint_transfer_with_regression_gates_passed"
            if all_gates_passed
            else "small_checkpoint_transfer_not_proven"
        ),
        "authority_sha256": authority["authority_sha256"],
        "model_identity_sha256": model_identity["identity_sha256"],
        "execution_spec_sha256": authority["execution_spec"]["semantic_sha256"],
        "source_closure": sources,
        "custody": custody_material["bindings"],
        "trained_candidate": trained_binding,
        "controls": control_bindings,
        "wrapped_projections": wrapped,
        "adapter_fingerprints": adapter_fingerprints,
        "holdout_example_count": len(projected_holdout),
        "canary_example_count": len(projected_canaries),
        "observations": observations,
        "falsification": falsification,
        "regression_canary_observations": {
            BASE_ARM: base_canaries,
            TRAINED_ARM: trained_canaries,
        },
        "regression_canary_verdict": canary_verdict,
        "ordinary_lexical_hashes": lexical_hashes,
        "ordinary_lexical_invariance_proven": lexical_invariance,
        "base_weights_unchanged": True,
        "all_small_checkpoint_gates_passed": all_gates_passed,
        "evaluator_custody_opened_only_in_this_process": True,
        "production_effect": False,
        "promotion_allowed": False,
        "duration_s": round(time.monotonic() - started, 6),
        "claims_not_supported": [
            "broad_reasoning_gain",
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
    parser.add_argument("--control-report", type=Path, required=True)
    parser.add_argument("--expected-control-report-sha256", required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--evaluator-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--execution-spec", type=Path, required=True)
    parser.add_argument("--expected-source-closure-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        return _run(arguments)
    except (
        ImportError,
        MemoryError,
        OSError,
        RecurrentSFTEvaluationError,
        RecurrentSFTExecutionError,
        RecurrentSFTFalsificationError,
        RecurrentSFTFalsificationEvaluationError,
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
