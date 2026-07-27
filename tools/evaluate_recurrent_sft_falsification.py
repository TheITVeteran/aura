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
from dataclasses import replace
from pathlib import Path
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.execution_spec import (  # noqa: E402
    RLCExecutionSpec,
)
from core.learning.recurrent_sft_behavior_canaries import (  # noqa: E402
    build_generated_behavior_canaries,
    build_generated_behavior_generation_contract,
    generated_behavior_verdict,
    grade_generated_behavior_text,
)
from core.learning.recurrent_sft_evaluation import (  # noqa: E402
    EVALUATION_SCHEMA,
    EVALUATION_SOURCE_ROLES,
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
        "atomic_writer": REPO_ROOT / "core/runtime/atomic_writer.py",
        "authority": (REPO_ROOT / "core/learning/structured_sft_research_authority.py"),
        "action_state_capture": (
            REPO_ROOT / "core/brain/llm/latent_cortex/action_state_capture.py"
        ),
        "branch_exchange": (REPO_ROOT / "core/brain/llm/latent_cortex/branch_exchange.py"),
        "branch_roles": REPO_ROOT / "core/brain/llm/latent_cortex/branches.py",
        "capability_canaries": (REPO_ROOT / "core/brain/llm/latent_cortex/capability_canaries.py"),
        "code_repl": REPO_ROOT / "core/skills/code_repl.py",
        "cognitive_operators": (REPO_ROOT / "core/brain/llm/latent_cortex/cognitive_operators.py"),
        "control_containment": (REPO_ROOT / "tools/launch_recurrent_sft_controls.py"),
        "depth_conditioned_lora": (REPO_ROOT / "core/learning/depth_conditioned_lora.py"),
        "detached_supervisor": REPO_ROOT / "tools/run_detached_step.py",
        "evaluation_contract": (REPO_ROOT / "core/learning/recurrent_sft_evaluation.py"),
        "epistemic_state": (REPO_ROOT / "core/brain/llm/latent_cortex/epistemic_state.py"),
        "escape": REPO_ROOT / "core/brain/llm/latent_cortex/escape.py",
        "evaluator": Path(__file__),
        "evaluator_launcher": (REPO_ROOT / "tools/launch_recurrent_sft_falsification.py"),
        "independent_verifier": (REPO_ROOT / "tools/verify_recurrent_sft_falsification.py"),
        "execution_spec": (REPO_ROOT / "core/brain/llm/latent_cortex/execution_spec.py"),
        "fast_weights": (REPO_ROOT / "core/brain/llm/latent_cortex/fast_weights.py"),
        "fast_weight_learning": (
            REPO_ROOT / "core/brain/llm/latent_cortex/fast_weight_learning.py"
        ),
        "falsification": (REPO_ROOT / "core/learning/recurrent_sft_falsification.py"),
        "file_read_gateway": REPO_ROOT / "core/runtime/file_read_gateway.py",
        "generated_behavior_canaries": (
            REPO_ROOT / "core/learning/recurrent_sft_behavior_canaries.py"
        ),
        "latent_cortex_engine": (REPO_ROOT / "core/brain/llm/latent_cortex/engine.py"),
        "latent_cortex_governance": (REPO_ROOT / "core/brain/llm/latent_cortex/governance.py"),
        "latent_optimizer": (REPO_ROOT / "core/brain/llm/latent_cortex/latent_opt.py"),
        "memory_guard": REPO_ROOT / "core/runtime/mlx_memory_guard.py",
        "model_lane": REPO_ROOT / "core/runtime/model_lane_control.py",
        "model_lane_admission": REPO_ROOT / "core/brain/lane_admission.py",
        "natural_deduction": REPO_ROOT / "core/reasoning/natural_deduction.py",
        "proof_kernel": REPO_ROOT / "core/reasoning/proof_kernel.py",
        "probe_cache": (REPO_ROOT / "core/brain/llm/latent_cortex/probe_cache.py"),
        "recurrence_adapter": (REPO_ROOT / "core/brain/llm/latent_cortex/recurrence_adapter.py"),
        "recurrence_identity": (
            REPO_ROOT / "core/brain/llm/latent_cortex/recurrence_adapter_identity_v2.py"
        ),
        "recurrent_loop_core": (REPO_ROOT / "core/brain/llm/latent_cortex/loop_core.py"),
        "recurrence_runner": (REPO_ROOT / "core/brain/llm/latent_cortex/recurrence.py"),
        "recurrence_objective": (REPO_ROOT / "core/learning/recurrence_native_objective_v2.py"),
        "resource_accounting": (REPO_ROOT / "core/brain/llm/latent_cortex/resource_accounting.py"),
        "runtime_errors": REPO_ROOT / "core/runtime/errors.py",
        "recurrent_sft_execution": (REPO_ROOT / "core/learning/recurrent_sft_execution.py"),
        "sandbox_profile_builder": (REPO_ROOT / "tools/launch_structured_sft_research.py"),
        "sandbox_runner": REPO_ROOT / "core/sandbox/runner.py",
        "layer_schedules": (REPO_ROOT / "core/brain/llm/latent_cortex/schedules.py"),
        "structured_sft": REPO_ROOT / "core/learning/structured_sft.py",
        "subprocess_gateway": REPO_ROOT / "core/runtime/subprocess_gateway.py",
        "latent_telemetry": (REPO_ROOT / "core/brain/llm/latent_cortex/telemetry.py"),
        "test_time_training": (REPO_ROOT / "core/brain/llm/latent_cortex/test_time_training.py"),
        "types": REPO_ROOT / "core/brain/llm/latent_cortex/types.py",
        "value_of_computation": (
            REPO_ROOT / "core/brain/llm/latent_cortex/value_of_computation.py"
        ),
        "verified_best": (REPO_ROOT / "core/brain/llm/latent_cortex/verified_best.py"),
        "workspace": REPO_ROOT / "core/brain/llm/latent_cortex/workspace.py",
    }


def evaluation_source_closure() -> dict[str, Any]:
    paths = evaluation_source_paths()
    expected = set(EVALUATION_SOURCE_ROLES)
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
    *,
    authority: Mapping[str, Any],
    expected_custody_binding_sha256: str,
    expected_custody: Mapping[str, Any],
) -> tuple[dict[str, bytes], dict[str, bytes], dict[str, Any]]:
    candidate_lexical = candidate_dir.expanduser()
    evaluator_lexical = evaluator_dir.expanduser()
    if candidate_lexical.is_symlink() or evaluator_lexical.is_symlink():
        _fail("recurrent_sft_evaluation_custody_root_symlink_rejected")
    candidate = candidate_lexical.resolve(strict=True)
    evaluator = evaluator_lexical.resolve(strict=True)
    if (
        candidate.is_symlink()
        or evaluator.is_symlink()
        or not candidate.is_dir()
        or not evaluator.is_dir()
        or candidate == evaluator
    ):
        _fail("recurrent_sft_evaluation_custody_roots_invalid")
    candidate_artifacts: dict[str, bytes] = {}
    evaluator_artifacts: dict[str, bytes] = {}
    for name in STRUCTURED_SFT_CANDIDATE_FILES:
        lexical = candidate / name
        if lexical.is_symlink():
            _fail("recurrent_sft_evaluation_candidate_symlink_rejected")
        path = lexical.resolve(strict=True)
        if path.parent != candidate:
            _fail("recurrent_sft_evaluation_candidate_path_escape")
        candidate_artifacts[name] = _read_bytes(path, role=f"candidate_{name}")
    for name in STRUCTURED_SFT_EVALUATOR_FILES:
        lexical = evaluator / name
        if lexical.is_symlink():
            _fail("recurrent_sft_evaluation_evaluator_symlink_rejected")
        path = lexical.resolve(strict=True)
        if path.parent != evaluator:
            _fail("recurrent_sft_evaluation_evaluator_path_escape")
        evaluator_artifacts[name] = _read_bytes(path, role=f"evaluator_{name}")
    rows, custody = evaluator_holdout_rows(
        candidate_artifacts,
        evaluator_artifacts,
        replay_semantics=False,
        expected_custody=expected_custody,
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
    custody_binding_sha256 = sha256_json(bindings)
    candidate_authority = authority.get("candidate")
    candidate_files = (
        candidate_authority.get("files") if isinstance(candidate_authority, Mapping) else None
    )
    observed_candidate_files = [
        {
            "name": name,
            "sha256": bindings["candidate"][name]["sha256"],
            "size_bytes": bindings["candidate"][name]["size_bytes"],
        }
        for name in STRUCTURED_SFT_CANDIDATE_FILES
    ]
    if (
        custody_binding_sha256 != expected_custody_binding_sha256
        or not isinstance(candidate_authority, Mapping)
        or observed_candidate_files != candidate_files
        or custody.get("candidate_package_sha256")
        != candidate_authority.get("candidate_package_sha256")
        or custody.get("evaluator_package_sha256")
        != candidate_authority.get("evaluator_package_sha256")
        or custody.get("custody_root_sha256") != candidate_authority.get("custody_root_sha256")
    ):
        _fail("recurrent_sft_evaluation_authority_custody_drift")
    return (
        candidate_artifacts,
        evaluator_artifacts,
        {
            "rows": rows,
            "bindings": bindings,
            "custody_binding_sha256": custody_binding_sha256,
        },
    )


def _reference_adapter(
    checkpoint_path: Path,
    *,
    expected_checkpoint_sha256: str,
    authority: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    if checkpoint_path.is_symlink():
        _fail("recurrent_sft_evaluation_reference_checkpoint_symlink_rejected")
    checkpoint_path = checkpoint_path.resolve(strict=True)
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
        or checkpoint.get("model_identity_sha256") != authority["model"]["identity_sha256"]
        or checkpoint.get("execution_spec_sha256") != authority["execution_spec"]["semantic_sha256"]
        or not isinstance(adapter, Mapping)
        or set(adapter) != {"path", "sha256", "size_bytes"}
        or not isinstance(adapter.get("path"), str)
        or Path(adapter["path"]).name != adapter["path"]
        or not _is_sha256(adapter.get("sha256"))
        or type(adapter.get("size_bytes")) is not int
        or adapter["size_bytes"] < 1
    ):
        _fail("recurrent_sft_evaluation_reference_checkpoint_invalid")
    lexical = checkpoint_path.parent / adapter["path"]
    if lexical.is_symlink():
        _fail("recurrent_sft_evaluation_trained_adapter_symlink_rejected")
    path = lexical.resolve(strict=True)
    if path.parent != checkpoint_path.parent.resolve(strict=True):
        _fail("recurrent_sft_evaluation_trained_adapter_path_escape")
    adapter_payload = _read_bytes(path, role="trained_adapter")
    if (
        len(adapter_payload) != adapter["size_bytes"]
        or sha256_bytes(adapter_payload) != adapter["sha256"]
    ):
        _fail("recurrent_sft_evaluation_trained_adapter_mismatch")
    return path, {
        "checkpoint": _artifact_binding(checkpoint_path, payload),
        "adapter": _artifact_binding(path, adapter_payload),
        "optimizer_updates": checkpoint.get("optimizer_updates"),
        "step": checkpoint.get("step"),
        "trainer_config_sha256": checkpoint.get("trainer_config_sha256"),
    }


def _control_adapters(
    report_path: Path,
    *,
    expected_report_sha256: str,
    authority: Mapping[str, Any],
    expected_reference_checkpoint_sha256: str,
    expected_reference_optimizer_updates: int,
    expected_trainer_config_sha256: str,
) -> tuple[dict[str, Path], dict[str, Any]]:
    if report_path.is_symlink():
        _fail("recurrent_sft_evaluation_control_report_symlink_rejected")
    report_path = report_path.resolve(strict=True)
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
        expected_reference_optimizer_updates=expected_reference_optimizer_updates,
        expected_trainer_config_sha256=expected_trainer_config_sha256,
    )
    paths: dict[str, Path] = {}
    adapter_bindings: dict[str, Any] = {}
    for arm in CONTROL_ARMS:
        binding = bindings[arm]
        lexical = report_path.parent / binding["filename"]
        if lexical.is_symlink():
            _fail(f"recurrent_sft_evaluation_control_adapter_{arm}_symlink")
        path = lexical.resolve(strict=True)
        if path.parent != report_path.parent:
            _fail(f"recurrent_sft_evaluation_control_adapter_{arm}_escape")
        payload = _read_bytes(path, role=f"control_adapter_{arm}")
        if len(payload) != binding["size_bytes"] or sha256_bytes(payload) != binding["sha256"]:
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


def _tokens_sha256(tokens: Sequence[int]) -> str:
    return sha256_bytes(
        json.dumps(
            list(tokens),
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )


def _generate_behavior_canaries(
    model: Any,
    tokenizer: Any,
    *,
    model_path: Path,
    spec: RLCExecutionSpec,
    arm: str,
    adapter_fingerprint: str | None,
    generation_contract: Mapping[str, Any],
    envelope: Any,
) -> list[dict[str, Any]]:
    import mlx.core as mx

    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_disabled,
    )
    from core.learning.recurrent_grpo import cortex_config_from_execution_spec

    if arm not in {BASE_ARM, TRAINED_ARM}:
        _fail("recurrent_sft_evaluation_behavior_arm_invalid")
    decode = generation_contract.get("decode")
    if not isinstance(decode, Mapping):
        _fail("recurrent_sft_evaluation_behavior_contract_invalid")
    config = replace(
        cortex_config_from_execution_spec(spec),
        decode_max_tokens=int(decode["max_tokens"]),
        decode_temperature=float(decode["temperature"]),
        decode_top_p=float(decode["top_p"]),
        decode_bridge_policy=str(decode["bridge_policy"]),
        allow_vanilla_fallback=bool(decode["allow_vanilla_fallback"]),
        generative_verifier_enabled=False,
        counterfactual_verifier_enabled=False,
        prefix_stability_enabled=False,
        local_repair_enabled=False,
        answer_replacement_enabled=False,
        telemetry_enabled=False,
        probe_cache_enabled=False,
    )
    if config.validate():
        _fail("recurrent_sft_evaluation_behavior_config_invalid")
    engine = LatentCortexEngine(
        model,
        tokenizer=tokenizer,
        config=config,
        model_path=str(model_path),
        schedule_library=None,
    )
    context = recurrence_adapter_disabled() if arm == BASE_ARM else nullcontext()
    observations: list[dict[str, Any]] = []
    with context:
        for case in build_generated_behavior_canaries():
            result = engine.reason(
                messages=[
                    {"role": "system", "content": case["system"]},
                    {"role": "user", "content": case["prompt"]},
                ],
                domain="evaluation",
                decode_max_tokens=int(decode["max_tokens"]),
                decode_sentence_grace_tokens=int(decode["sentence_grace_tokens"]),
                nonparametric_memory_enabled=bool(decode["nonparametric_memory_enabled"]),
            )
            receipt = result.receipt
            integrity = receipt.weight_integrity.to_dict()
            params_before = str(integrity.get("params_before") or "")
            params_after = str(integrity.get("params_after") or "")
            fallback_used = any(
                str(flag).startswith(("fallback", "latent_and_fallback"))
                for flag in receipt.honest_flags
            ) or "fallback" in str(result.reason)
            text = str(result.text or "")
            observations.append(
                {
                    "case_id": case["case_id"],
                    "family": case["family"],
                    "arm": arm,
                    "prompt_sha256": sha256_bytes(case["prompt"].encode("utf-8")),
                    "generation_contract_sha256": generation_contract["contract_sha256"],
                    "engine_ok": bool(result.ok),
                    "engine_reason": str(result.reason or ""),
                    "text": text,
                    "text_sha256": sha256_bytes(text.encode("utf-8")),
                    "tokens": list(result.tokens),
                    "token_count": len(result.tokens),
                    "tokens_sha256": _tokens_sha256(result.tokens),
                    "decode_termination": str(receipt.decode_termination or ""),
                    "fallback_used": fallback_used,
                    "adapter_active": arm == TRAINED_ARM,
                    "adapter_fingerprint": (adapter_fingerprint if arm == TRAINED_ARM else None),
                    "params_before": params_before,
                    "params_after": params_after,
                    "params_unchanged": bool(
                        receipt.params_unchanged and params_before and params_before == params_after
                    ),
                    "grade": grade_generated_behavior_text(case, text),
                }
            )
            mx.clear_cache()
            envelope.reclaim(force=True)
    return observations


def _validated_containment_contract(
    path: Path,
    *,
    arguments: argparse.Namespace,
    source_closure: Mapping[str, Any],
) -> dict[str, Any]:
    if path.is_symlink():
        _fail("recurrent_sft_evaluation_contract_symlink_rejected")
    contract = _read_json(path, role="containment_contract")
    body = dict(contract)
    observed = body.pop("contract_sha256", None)
    if (
        observed != sha256_json(body)
        or contract.get("authority_sha256") != arguments.expected_authority_sha256
        or contract.get("reference_checkpoint_sha256")
        != arguments.expected_reference_checkpoint_sha256
        or contract.get("control_report_file_sha256") != arguments.expected_control_report_sha256
        or contract.get("custody_binding_sha256") != arguments.expected_custody_binding_sha256
        or contract.get("source_closure") != source_closure
        or contract.get("network") != "kernel_denied"
        or contract.get("process_fork") != "kernel_denied"
        or contract.get("evaluator_access") is not True
        or contract.get("training_write_access") is not False
        or contract.get("resident_checkpoint_access") is not False
        or contract.get("production_write_access") is not False
        or contract.get("resume_contract") != "none"
    ):
        _fail("recurrent_sft_evaluation_containment_contract_invalid")
    return contract


def _run(arguments: argparse.Namespace) -> int:
    if not all(
        _is_sha256(value)
        for value in (
            arguments.expected_authority_sha256,
            arguments.expected_reference_checkpoint_sha256,
            arguments.expected_control_report_sha256,
            arguments.expected_custody_binding_sha256,
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
    behavior_generation_contract = build_generated_behavior_generation_contract(
        execution_spec_sha256=authority["execution_spec"]["semantic_sha256"],
    )
    model_identity = small_model_identity(arguments.model_dir)
    if model_identity != authority["model"]:
        _fail("recurrent_sft_evaluation_model_identity_drift")
    sources = evaluation_source_closure()
    if sources["closure_sha256"] != arguments.expected_source_closure_sha256:
        _fail("recurrent_sft_evaluation_source_closure_drift")
    trained_adapter_path, trained_binding = _reference_adapter(
        arguments.reference_checkpoint.expanduser(),
        expected_checkpoint_sha256=arguments.expected_reference_checkpoint_sha256,
        authority=authority,
    )
    expected_trainer_config_sha256 = sha256_json(authority["trainer"])
    if (
        type(trained_binding.get("optimizer_updates")) is not int
        or trained_binding["optimizer_updates"] < 1
        or trained_binding.get("step") != trained_binding["optimizer_updates"]
        or trained_binding.get("trainer_config_sha256") != expected_trainer_config_sha256
    ):
        _fail("recurrent_sft_evaluation_reference_workload_invalid")
    control_paths, control_bindings = _control_adapters(
        arguments.control_report.expanduser(),
        expected_report_sha256=arguments.expected_control_report_sha256,
        authority=authority,
        expected_reference_checkpoint_sha256=(arguments.expected_reference_checkpoint_sha256),
        expected_reference_optimizer_updates=trained_binding["optimizer_updates"],
        expected_trainer_config_sha256=expected_trainer_config_sha256,
    )
    containment_contract = _validated_containment_contract(
        arguments.containment_contract.expanduser().resolve(strict=True),
        arguments=arguments,
        source_closure=sources,
    )
    contract_custody = containment_contract.get("custody_bindings")
    if not isinstance(contract_custody, Mapping) or not isinstance(
        contract_custody.get("custody"), Mapping
    ):
        _fail("recurrent_sft_evaluation_contract_custody_invalid")
    _candidate, _evaluator, custody_material = _candidate_and_evaluator_artifacts(
        arguments.candidate_dir,
        arguments.evaluator_dir,
        authority=authority,
        expected_custody_binding_sha256=(arguments.expected_custody_binding_sha256),
        expected_custody=contract_custody["custody"],
    )
    holdout_rows = custody_material["rows"]
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
        base_behavior_canaries = _generate_behavior_canaries(
            model,
            tokenizer,
            model_path=arguments.model_dir,
            spec=spec,
            arm=BASE_ARM,
            adapter_fingerprint=None,
            generation_contract=behavior_generation_contract,
            envelope=envelope,
        )
        lexical_hashes[BASE_ARM] = _ordinary_lexical_hash(model, tokenizer)

        arm_paths = {
            TRAINED_ARM: trained_adapter_path,
            **control_paths,
        }
        trained_canaries: list[dict[str, Any]] | None = None
        trained_behavior_canaries: list[dict[str, Any]] | None = None
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
                trained_behavior_canaries = _generate_behavior_canaries(
                    model,
                    tokenizer,
                    model_path=arguments.model_dir,
                    spec=spec,
                    arm=TRAINED_ARM,
                    adapter_fingerprint=adapter_fingerprints[TRAINED_ARM],
                    generation_contract=behavior_generation_contract,
                    envelope=envelope,
                )
        if trained_canaries is None or trained_behavior_canaries is None:
            _fail("recurrent_sft_evaluation_trained_canaries_missing")
        base_after = full_weight_checkpoint_identity(arguments.model_dir)
        if base_after != base_before:
            _fail("recurrent_sft_evaluation_base_weights_changed")

    falsification = build_falsification_verdict(observations)
    likelihood_canary_verdict = regression_canary_verdict(
        base_canaries,
        trained_canaries,
    )
    behavior_canary_verdict = generated_behavior_verdict(
        base_behavior_canaries,
        trained_behavior_canaries,
        expected_generation_contract_sha256=(behavior_generation_contract["contract_sha256"]),
        expected_trained_adapter_fingerprint=(adapter_fingerprints[TRAINED_ARM]),
    )
    lexical_invariance = len(set(lexical_hashes.values())) == 1
    all_gates_passed = (
        falsification["heldout_transfer_proven"]
        and likelihood_canary_verdict["passed"]
        and behavior_canary_verdict["passed"]
        and lexical_invariance
    )
    body = {
        "schema": REPORT_SCHEMA,
        "status": (
            "small_checkpoint_transfer_with_all_regression_gates_passed"
            if all_gates_passed
            else "small_checkpoint_transfer_not_proven"
        ),
        "authority_sha256": authority["authority_sha256"],
        "model_identity_sha256": model_identity["identity_sha256"],
        "execution_spec_sha256": authority["execution_spec"]["semantic_sha256"],
        "containment_contract_sha256": containment_contract["contract_sha256"],
        "source_closure": sources,
        "custody": custody_material["bindings"],
        "custody_binding_sha256": custody_material["custody_binding_sha256"],
        "trained_candidate": trained_binding,
        "controls": control_bindings,
        "wrapped_projections": wrapped,
        "adapter_fingerprints": adapter_fingerprints,
        "holdout_example_count": len(projected_holdout),
        "canary_example_count": len(projected_canaries),
        "observations": observations,
        "falsification": falsification,
        "regression_likelihood_canary_observations": {
            BASE_ARM: base_canaries,
            TRAINED_ARM: trained_canaries,
        },
        "regression_likelihood_canary_verdict": likelihood_canary_verdict,
        "generated_behavior_canary_count": len(build_generated_behavior_canaries()),
        "generated_behavior_generation_contract": (behavior_generation_contract),
        "generated_behavior_generation_contract_sha256": (
            behavior_generation_contract["contract_sha256"]
        ),
        "generated_behavior_canary_observations": {
            BASE_ARM: base_behavior_canaries,
            TRAINED_ARM: trained_behavior_canaries,
        },
        "generated_behavior_canary_verdict": behavior_canary_verdict,
        "generated_behavior_regression_tested": True,
        "ordinary_lexical_hashes": lexical_hashes,
        "ordinary_lexical_invariance_proven": lexical_invariance,
        "base_weights_unchanged": True,
        "all_small_checkpoint_gates_passed": all_gates_passed,
        "custody_execution": {
            "launcher_semantic_replay_bound": True,
            "evaluator_exact_byte_rehash": True,
            "evaluator_projection_validation": True,
            "evaluator_semantic_replay": False,
            "independent_verifier_semantic_replay_required": True,
            "reason": "kernel_process_fork_denied",
        },
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
    parser.add_argument("--expected-custody-binding-sha256", required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--evaluator-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--execution-spec", type=Path, required=True)
    parser.add_argument("--expected-source-closure-sha256", required=True)
    parser.add_argument("--containment-contract", type=Path, required=True)
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
