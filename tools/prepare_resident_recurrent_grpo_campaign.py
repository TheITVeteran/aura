#!/usr/bin/env python3
"""Freeze, verify, and execute the resident recurrent-GRPO preregistration.

The contract is deliberately created before the long resident run. It binds
the model, recurrent graph, training-only task corpus, executable sources,
resource limits, causal factorial, powered confirmatory design, and claims
that remain unavailable until independent evidence accepts them.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _repository_roots(repo_root: Path | None = None) -> tuple[Path, ...]:
    """Every root that is this repository, including its main checkout.

    A git worktree is the same repository under a different path, and large
    artifacts — model weights especially — live once in the main checkout and
    are reached from a worktree through a link. Resolving such a path lands
    outside the worktree root and looked exactly like a traversal escape, so a
    preregistration run from a worktree failed with model_path_invalid on a
    perfectly legitimate model.

    Confinement is still confinement: the declared path stays lexically
    relative and free of "..", and the resolved path must land inside one of
    the roots that *are* this repository. Nothing else is admitted.
    """
    # Resolved per call, not frozen at import: REPO_ROOT is monkeypatched by
    # tests that build a whole contract tree in a temporary directory, and a
    # cached root list silently ignores them.
    base = Path(repo_root) if repo_root is not None else REPO_ROOT
    roots = [base]
    marker = base / ".git"
    try:
        if marker.is_file():
            text = marker.read_text(encoding="utf-8").strip()
            if text.startswith("gitdir:"):
                gitdir = Path(text.split(":", 1)[1].strip())
                # .../<main>/.git/worktrees/<name> -> <main>
                for parent in gitdir.resolve().parents:
                    if parent.name == ".git":
                        roots.append(parent.parent)
                        break
    except OSError:
        pass
    return tuple(dict.fromkeys(root.resolve() for root in roots))


from core.brain.llm.latent_cortex.execution_spec import (  # noqa: E402
    RLCExecutionSpec,
)
from core.brain.llm.latent_cortex.frontier_tasks import (  # noqa: E402
    CURRENT_EXCLUDED_TRAINING_FAMILIES,
    CURRENT_REGISTRY_VERSION,
    FRONTIER_DOMAINS,
)
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (  # noqa: E402
    full_weight_checkpoint_identity,
    model_behavior_bundle_identity,
)
from core.learning.grpo_training_state import canonical_json_bytes  # noqa: E402
from core.learning.recurrence_curriculum import (  # noqa: E402
    RECURRENCE_TRAINING_FAMILIES,
)
from core.learning.recurrent_grpo import (  # noqa: E402
    VerifiedTrajectoryGroupConfig,
)
from core.learning.recurrent_grpo_artifact_schema import (  # noqa: E402
    recurrent_training_adequacy_policy,
)
from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_bytes_if_absent,
    ensure_private_directory,
)
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402
from tools import run_detached_step  # noqa: E402
from tools.train_grpo import _build_task_split, _dataset_payload  # noqa: E402

CONTRACT_SCHEMA = "aura.resident_recurrent_grpo_preregistration.v1"
FULL_TRAINING_PROFILE = "full_training"
UPDATE_CANARY_PROFILE = "update_canary"
CAMPAIGN_PROFILES = frozenset({FULL_TRAINING_PROFILE, UPDATE_CANARY_PROFILE})
DEFAULT_CAMPAIGN_ID = "resident-32b-recurrent-grpo-cp259"
CAMPAIGN_ID = DEFAULT_CAMPAIGN_ID
DEFAULT_MODEL = "training/fused-model/Aura-32B-crsm-closeout-jul1-20260701-215118"
DEFAULT_SPEC = "config/latent_cortex/resident_32b_recurrent_grpo_execution_spec.json"
DEFAULT_CONTRACT = "config/latent_cortex/resident_32b_recurrent_grpo_preregistration.json"
DEFAULT_ROOT = "artifacts/closeout/latent_cortex/cp259_resident_32b_recurrent_grpo"
NOT_BEFORE = "2026-07-21T17:00:00-07:00"
TRAINING_SEED = 2026072102
CONFIRMATORY_OBSERVATIONS_PER_DOMAIN = 411
TRAINING_WATCHDOG_POLICY: Mapping[str, Any] = {
    "schema": "aura.resident_recurrent_grpo.training_watchdog.v1",
    "max_attempts": 8,
    "retry_backoff_s": 30.0,
    "max_consecutive_no_progress_failures": 2,
    "restart_scope": "exact_source_bound_checkpoint_resume_only",
}
UPDATE_CANARY_WATCHDOG_POLICY: Mapping[str, Any] = {
    **TRAINING_WATCHDOG_POLICY,
    "max_attempts": 3,
}
TRAINING_PARAMETERS: Mapping[str, Any] = {
    "task_source": "recurrence_curriculum",
    "domains": list(RECURRENCE_TRAINING_FAMILIES),
    "depths": [2, 4, 8],
    "train_per_cell": 8,
    "holdout_per_cell": 1,
    "group_size": 2,
    "temperature": 1.0,
    "max_tokens": 320,
    "kl_coefficient": 0.02,
    "format_credit": 0.0,
    "trajectory_credit": False,
    "trajectory_shaping_weight": 0.25,
    "lora_rank": 8,
    "lora_targets": "o_proj,v_proj,q_proj",
    "lora_layers": 8,
    "learning_rate": 0.000005,
    "max_steps": 288,
    "eval_every": 96,
    "checkpoint_every": 1,
    "checkpoint_keep": 3,
    "min_signal_groups": 8,
    "calibrate": True,
    "calibrate_samples": 1,
    "calibrate_group": 4,
    "calibrate_tokens": 320,
    "calibrate_minutes": 60.0,
    "cot": True,
    "max_minutes": 2160.0,
    "memory_fraction": 0.42,
    "fixed_update_canary": False,
    "seed": TRAINING_SEED,
    "verified_trajectory_config": (
        "config/latent_cortex/resident_32b_verified_intervention_group_config.json"
    ),
}
UPDATE_CANARY_PARAMETERS: Mapping[str, Any] = {
    **TRAINING_PARAMETERS,
    "domains": list(RECURRENCE_TRAINING_FAMILIES),
    "depths": [4],
    "train_per_cell": 1,
    "holdout_per_cell": 1,
    "max_steps": len(RECURRENCE_TRAINING_FAMILIES),
    "eval_every": len(RECURRENCE_TRAINING_FAMILIES),
    "checkpoint_keep": len(RECURRENCE_TRAINING_FAMILIES),
    "calibrate_minutes": 60.0,
    "calibrate": False,
    "fixed_update_canary": True,
    "max_minutes": 240.0,
    "seed": TRAINING_SEED + 1,
}
FULL_RESOURCE_ENVELOPE: Mapping[str, Any] = {
    "host_memory_bytes": 68_719_476_736,
    "mlx_memory_fraction": 0.42,
    "exclusive_resident_model_owner": True,
    "detached_timeout_s": 259_200,
    "multi_hour_soak": False,
}
UPDATE_CANARY_RESOURCE_ENVELOPE: Mapping[str, Any] = {
    **FULL_RESOURCE_ENVELOPE,
    "detached_timeout_s": 21_600,
}
UPDATE_CANARY_VERDICT_SCHEMA = "aura.resident_recurrent_grpo.update_canary_verdict.v1"
SOURCE_ROLES: Mapping[str, str] = {
    "campaign_contract": "tools/prepare_resident_recurrent_grpo_campaign.py",
    "trainer": "tools/train_grpo.py",
    "transition_provider_factory": ("core/learning/verified_transition_production_factory.py"),
    "transition_launch_bundle": ("core/learning/verified_transition_launch_bundle.py"),
    "transition_launch_runner": "tools/run_verified_recurrent_grpo_training.py",
    "transition_launch_materializer": ("tools/materialize_verified_recurrent_grpo_launch.py"),
    "transition_recurrent_evidence": ("core/learning/verified_recurrent_transition_evidence.py"),
    "transition_recurrent_repository": (
        "core/learning/verified_recurrent_transition_repository.py"
    ),
    "transition_policy_probe": ("core/learning/verified_transition_policy_probe.py"),
    "transition_measurement_chain": ("core/learning/verified_transition_measurement_chain.py"),
    "transition_policy_state_replay": ("core/learning/verified_transition_policy_state_replay.py"),
    "transition_policy_state_replay_worker": ("tools/replay_verified_recurrent_policy_states.py"),
    "transition_policy_state_replay_resume": ("tools/resume_durable_external_verifier_job.py"),
    "durable_external_verifier_job": ("core/learning/durable_external_verifier_job.py"),
    "recurrent_training_prompt": ("core/learning/recurrent_training_prompt.py"),
    "atomic_writer": "core/runtime/atomic_writer.py",
    "file_read_gateway": "core/runtime/file_read_gateway.py",
    "file_write_gateway": "core/runtime/file_write_gateway.py",
    "transition_rejection_transaction": (
        "core/learning/verified_transition_rejection_transaction.py"
    ),
    "grpo": "core/learning/grpo.py",
    "curriculum": "core/learning/adaptive_curriculum.py",
    "tasks": "core/learning/recurrence_curriculum.py",
    "checkpoint": "core/learning/grpo_training_state.py",
    "artifact_schema": "core/learning/recurrent_grpo_artifact_schema.py",
    "recurrent_grpo": "core/learning/recurrent_grpo.py",
    "recurrent_objective": "core/learning/recurrence_native_objective_v2.py",
    "verified_trainer": "core/learning/verified_transition_trainer.py",
    "transition_campaign": "core/learning/verified_transition_campaign.py",
    "transition_episode": "core/learning/verified_transition_episode.py",
    "transition_reward": "core/learning/verified_transition_reward.py",
    "scope_reachability": "core/learning/scope_reachability.py",
    "transition_admission": ("core/learning/verified_transition_group_admission.py"),
    "transition_update": "core/learning/verified_transition_update.py",
    "transition_training_evidence": ("core/learning/verified_transition_training_evidence.py"),
    "campaign_trust": ("core/brain/llm/latent_cortex/campaign_trust.py"),
    "transition_provider": "core/learning/verified_transition_provider.py",
    "transition_transaction": ("core/learning/verified_transition_transaction.py"),
    "transition_causal_campaign": ("core/learning/verified_transition_causal_campaign.py"),
    "verified_training_task": "core/learning/verified_training_task.py",
    "verified_token_trace": "core/learning/verified_token_trace.py",
    "execution_spec": "core/brain/llm/latent_cortex/execution_spec.py",
    "latent_engine": "core/brain/llm/latent_cortex/engine.py",
    "latent_schedules": "core/brain/llm/latent_cortex/schedules.py",
    "recurrence": "core/brain/llm/latent_cortex/recurrence.py",
    "adapter": "core/brain/llm/latent_cortex/recurrence_adapter.py",
    "adapter_identity": ("core/brain/llm/latent_cortex/recurrent_grpo_adapter_identity.py"),
    "campaign_runner": "tools/run_latent_cortex_paired_campaign.py",
    "campaign_freezer": "tools/prepare_latent_cortex_campaign.py",
    "campaign_verifier": "tools/verify_paired_campaign_evidence.py",
    "independent_scorer": "tools/independent_paired_campaign_scoring.py",
    "contamination_auditor": "tools/produce_contamination_audit.py",
    "detached_supervisor": "tools/run_detached_step.py",
}


class PreregistrationError(ValueError):
    """Stable fail-closed contract error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise PreregistrationError(code)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _document_sha(document: Mapping[str, Any]) -> str:
    return _sha256(canonical_json_bytes(document))


def _bare_canonical(document: Mapping[str, Any]) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _strict_json(path: Path) -> dict[str, Any]:
    supplied = path.expanduser()
    if not supplied.is_absolute():
        supplied = REPO_ROOT / supplied
    if supplied.is_symlink():
        _fail("document_path_invalid")
    try:
        raw = supplied.resolve(strict=True).read_bytes()
    except OSError as exc:
        raise PreregistrationError("document_unreadable") from exc

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                _fail("document_duplicate_key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: _fail("document_nonfinite"),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PreregistrationError("document_invalid") from exc
    if not isinstance(value, dict):
        _fail("document_schema_invalid")
    return value


def _repo_path(value: str, *, role: str, must_exist: bool = True) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        _fail(f"{role}_path_invalid")
    pure = PurePosixPath(value)
    if pure.is_absolute() or str(pure) != value or ".." in pure.parts:
        _fail(f"{role}_path_invalid")
    for root in _repository_roots(REPO_ROOT):
        candidate = root / pure
        try:
            resolved = candidate.resolve(strict=must_exist)
        except OSError:
            continue
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        return resolved
    _fail(f"{role}_path_invalid")


def _binding(relative: str) -> dict[str, Any]:
    path = _repo_path(relative, role="binding")
    if not path.is_file() or path.is_symlink():
        _fail("binding_path_invalid")
    payload = path.read_bytes()
    return {
        "path": relative,
        "sha256": _sha256(payload),
        "size_bytes": len(payload),
    }


def _load_spec(relative: str) -> tuple[RLCExecutionSpec, dict[str, Any]]:
    binding = _binding(relative)
    raw = _repo_path(relative, role="execution_spec").read_bytes()
    try:
        parsed = json.loads(raw.decode("ascii"))
        spec = RLCExecutionSpec.from_dict(parsed)
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise PreregistrationError("execution_spec_invalid") from exc
    return spec, {**binding, "semantic_sha256": spec.sha256}


def _verified_trajectory_config_commitment(
    spec: RLCExecutionSpec,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    params = parameters or TRAINING_PARAMETERS
    declared = params.get("verified_trajectory_config")
    if declared is None:
        return None
    if not isinstance(declared, str) or not declared:
        _fail("verified_trajectory_config_path_invalid")
    lexical_source = REPO_ROOT / PurePosixPath(declared)
    if lexical_source.is_symlink():
        _fail("verified_trajectory_config_file_invalid")
    source = _repo_path(declared, role="verified_trajectory_config")
    if not source.is_file():
        _fail("verified_trajectory_config_file_invalid")
    try:
        raw = read_stable_bytes(source, max_bytes=65_536)
        parsed = json.loads(raw.decode("ascii"))
        config = VerifiedTrajectoryGroupConfig.from_dict(parsed)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise PreregistrationError("verified_trajectory_config_invalid") from exc
    canonical = canonical_json_bytes(config.to_dict())
    if raw != canonical:
        _fail("verified_trajectory_config_noncanonical")
    if int(params["group_size"]) != len(spec.branch_roles):
        _fail("verified_trajectory_group_branch_count_mismatch")
    try:
        if config.trajectory_config is not None:
            config.trajectory_config.validate_depth(spec.recurrent_steps)
        if config.intervention_config is not None:
            config.intervention_config.validate_depth(spec.recurrent_steps)
    except ValueError as exc:
        raise PreregistrationError("verified_trajectory_config_depth_invalid") from exc
    return {
        **_binding(declared),
        "config": config.to_dict(),
        "semantic_sha256": _sha256(canonical),
    }


def _training_parameters_for_profile(profile: str) -> Mapping[str, Any]:
    if profile == FULL_TRAINING_PROFILE:
        return TRAINING_PARAMETERS
    if profile == UPDATE_CANARY_PROFILE:
        return UPDATE_CANARY_PARAMETERS
    _fail("campaign_profile_invalid")


def _watchdog_policy_for_profile(profile: str) -> Mapping[str, Any]:
    if profile == FULL_TRAINING_PROFILE:
        return TRAINING_WATCHDOG_POLICY
    if profile == UPDATE_CANARY_PROFILE:
        return UPDATE_CANARY_WATCHDOG_POLICY
    _fail("campaign_profile_invalid")


def _resource_envelope_for_profile(profile: str) -> Mapping[str, Any]:
    if profile == FULL_TRAINING_PROFILE:
        return FULL_RESOURCE_ENVELOPE
    if profile == UPDATE_CANARY_PROFILE:
        return UPDATE_CANARY_RESOURCE_ENVELOPE
    _fail("campaign_profile_invalid")


def _training_argv(
    *,
    campaign_id: str,
    model: str,
    output: str,
    execution_spec: str,
    parameters: Mapping[str, Any] | None = None,
) -> list[str]:
    params = parameters or TRAINING_PARAMETERS
    argv = [
        "tools/train_grpo.py",
        "--model",
        model,
        "--out-dir",
        output,
        "--adapter-id",
        campaign_id,
        "--execution-mode",
        "recurrent",
        "--execution-spec",
        execution_spec,
        "--task-source",
        str(params["task_source"]),
        "--domains",
        ",".join(params["domains"]),
        "--depths",
        ",".join(str(value) for value in params["depths"]),
    ]
    flags = (
        ("train_per_cell", "--train-per-cell"),
        ("holdout_per_cell", "--holdout-per-cell"),
        ("group_size", "--group-size"),
        ("temperature", "--temperature"),
        ("max_tokens", "--max-tokens"),
        ("kl_coefficient", "--kl-coefficient"),
        ("format_credit", "--format-credit"),
        ("trajectory_shaping_weight", "--trajectory-shaping-weight"),
        ("lora_rank", "--lora-rank"),
        ("lora_targets", "--lora-targets"),
        ("lora_layers", "--lora-layers"),
        ("learning_rate", "--learning-rate"),
        ("max_steps", "--max-steps"),
        ("eval_every", "--eval-every"),
        ("checkpoint_every", "--checkpoint-every"),
        ("checkpoint_keep", "--checkpoint-keep"),
        ("min_signal_groups", "--min-signal-groups"),
        ("calibrate_samples", "--calibrate-samples"),
        ("calibrate_group", "--calibrate-group"),
        ("calibrate_tokens", "--calibrate-tokens"),
        ("calibrate_minutes", "--calibrate-minutes"),
        ("max_minutes", "--max-minutes"),
        ("memory_fraction", "--memory-fraction"),
        ("seed", "--seed"),
    )
    for key, flag in flags:
        argv.extend((flag, str(params[key])))
    if params["calibrate"]:
        argv.append("--calibrate")
    if params["trajectory_credit"]:
        argv.append("--trajectory-credit")
    if params.get("verified_trajectory_config") is not None:
        argv.extend(
            (
                "--verified-trajectory-config",
                str(params.get("verified_trajectory_config")),
            )
        )
    if params["cot"]:
        argv.append("--cot")
    if params.get("fixed_update_canary"):
        argv.append("--fixed-update-canary")
    return argv


def _dataset_commitment(
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    params = parameters or TRAINING_PARAMETERS
    train, holdout, source = _build_task_split(
        task_source=str(params["task_source"]),
        domains=list(params["domains"]),
        depths=list(params["depths"]),
        train_per_cell=int(params["train_per_cell"]),
        holdout_per_cell=int(params["holdout_per_cell"]),
        seed=int(params["seed"]),
    )
    payload = _dataset_payload(train, holdout, seed=int(params["seed"]))
    raw = canonical_json_bytes(payload)
    return {
        "schema": payload["schema"],
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
        "train_tasks": len(train),
        "holdout_tasks": len(holdout),
        "task_source": str(source.relative_to(REPO_ROOT)),
        "families": list(params["domains"]),
        "depths": list(params["depths"]),
        "train_holdout_id_overlap": 0,
        "train_holdout_prompt_overlap": 0,
        "excluded_evaluation_registry": CURRENT_REGISTRY_VERSION,
        "excluded_evaluation_families": list(CURRENT_EXCLUDED_TRAINING_FAMILIES),
    }


def build_contract(
    *,
    campaign_id: str = DEFAULT_CAMPAIGN_ID,
    campaign_profile: str = FULL_TRAINING_PROFILE,
    model: str = DEFAULT_MODEL,
    execution_spec: str = DEFAULT_SPEC,
    artifact_root: str = DEFAULT_ROOT,
    committed_at: str,
    model_identity: Mapping[str, Any] | None = None,
    behavior_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not campaign_id.startswith("resident-32b-recurrent-grpo-cp"):
        _fail("campaign_id_invalid")
    if campaign_profile not in CAMPAIGN_PROFILES:
        _fail("campaign_profile_invalid")
    params = _training_parameters_for_profile(campaign_profile)
    watchdog_policy = _watchdog_policy_for_profile(campaign_profile)
    resource_envelope = _resource_envelope_for_profile(campaign_profile)
    model_path = _repo_path(model, role="model")
    if not model_path.is_dir():
        _fail("model_path_invalid")
    spec, spec_binding = _load_spec(execution_spec)
    if spec.n_slots != 16 or len(spec.branch_roles) != 2 or spec.recurrent_steps != 4:
        _fail("execution_spec_campaign_profile_invalid")
    datetime.fromisoformat(committed_at)
    not_before = datetime.fromisoformat(NOT_BEFORE)
    paths = {
        "artifact_root": artifact_root,
        "training_output": f"{artifact_root}/training",
        "initial_policy_probe": f"{artifact_root}/policy-probe",
        "verified_launch_bundle": (f"{artifact_root}/verified-launch/launch-bundle.json"),
        "verified_launch_bundle_sha256": (f"{artifact_root}/verified-launch/launch-bundle.sha256"),
        "detached_training": f"{artifact_root}/detached-training",
        "frozen_adapter": f"{artifact_root}/frozen-adapter",
        "directional_campaign": f"{artifact_root}/directional-campaign",
        "confirmatory_campaign": f"{artifact_root}/confirmatory-campaign",
        "external_comparison": f"{artifact_root}/external-frontier",
    }
    for role, value in paths.items():
        _repo_path(value, role=role, must_exist=False)
    resolved_model_identity = dict(model_identity or full_weight_checkpoint_identity(model_path))
    resolved_behavior_identity = dict(
        behavior_identity or model_behavior_bundle_identity(model_path)
    )
    sources = {role: _binding(path) for role, path in SOURCE_ROLES.items()}
    training_argv = _training_argv(
        campaign_id=campaign_id,
        model=model,
        output=paths["training_output"],
        execution_spec=execution_spec,
        parameters=params,
    )
    trajectory_config_commitment = _verified_trajectory_config_commitment(
        spec,
        parameters=params,
    )
    arms = [
        "base_vanilla",
        "base_rlc",
        "adapter_vanilla",
        "adapter_rlc",
        "base_equal_compute",
        "adapter_equal_compute",
    ]
    mechanism_attribution = {
        "required": True,
        "claim_eligible": False,
        "purpose": (
            "separate the permanent recurrent-adapter gain from runtime "
            "latent optimization, episodic fast weights, branch exchange, "
            "and equal-compute sampling"
        ),
        "candidate_profiles": [
            "recurrent_trained_fixed_depth",
            "resident_full_stack",
            "resident_full_stack_no_latent_opt",
            "resident_full_stack_no_fast_weights",
            "resident_full_stack_no_branch_exchange",
        ],
        "required_comparisons": [
            "resident_full_stack > recurrent_trained_fixed_depth",
            ("resident_full_stack > resident_full_stack_no_latent_opt"),
            ("resident_full_stack > resident_full_stack_no_fast_weights"),
            ("resident_full_stack > resident_full_stack_no_branch_exchange"),
            "resident_full_stack > adapter_equal_compute",
        ],
        "acceptance_rules": [
            "same_model_checkpoint",
            "same_adapter_checkpoint",
            "same_task_manifest",
            "same_answer_reveal",
            "same_decode_contract",
            "same_or_lower_layer_app_budget_for_controls",
            "fast_weight_erase_and_canary_receipts_required",
            "latent_opt_acceptance_and_rejection_receipts_required",
            "no_claim_from_profile_without_required_receipts",
        ],
    }
    confirmatory_tasks = len(FRONTIER_DOMAINS) * CONFIRMATORY_OBSERVATIONS_PER_DOMAIN
    material = {
        "schema": CONTRACT_SCHEMA,
        "campaign_id": campaign_id,
        "campaign_profile": campaign_profile,
        "committed_at": committed_at,
        "launch_not_before": NOT_BEFORE,
        "launch_not_before_unix": int(not_before.timestamp()),
        "model": {
            "path": model,
            "base_checkpoint": resolved_model_identity,
            "behavior_bundle": resolved_behavior_identity,
            "personality_adapter": "none",
        },
        "execution_spec": spec_binding,
        "sources": sources,
        "paths": paths,
        "training": {
            "campaign_profile": campaign_profile,
            "execution_mode": "recurrent",
            "parameters": dict(params),
            "argv": training_argv,
            "dataset": _dataset_commitment(params),
            **(
                {"verified_trajectory_config_artifact": (trajectory_config_commitment)}
                if trajectory_config_commitment is not None
                else {}
            ),
            "resume_contract": "exact_identity_bound_checkpoint",
            "watchdog_policy": dict(watchdog_policy),
            "completion_required": {
                "schema": "aura.recurrent_grpo_training_completion.v1",
                "complete": True,
                "halt_reason": "max_steps",
                "causal_gain_proven": False,
                "training_adequacy": recurrent_training_adequacy_policy(),
            },
            "resource_envelope": dict(resource_envelope),
        },
        "evaluation": {
            "campaign_profile": campaign_profile,
            "claim_eligible": False,
            "registry_version": CURRENT_REGISTRY_VERSION,
            "domains": list(FRONTIER_DOMAINS),
            "difficulty": 3,
            "arms": arms,
            "directional_pilot": {
                "observations_per_domain": 8,
                "task_count": len(FRONTIER_DOMAINS) * 8,
                "cell_count": len(FRONTIER_DOMAINS) * 8 * len(arms),
                "claim_eligible": False,
            },
            "powered_confirmatory": {
                "observations_per_domain": CONFIRMATORY_OBSERVATIONS_PER_DOMAIN,
                "task_count": confirmatory_tasks,
                "cell_count": confirmatory_tasks * len(arms),
                "alpha": 0.05,
                "multiplicity": "holm_familywise",
                "fresh_externally_issued_tasks": True,
            },
            "equal_information": True,
            "equal_tools": True,
            "operation_level_compute_reconstruction": True,
            "separate_latency_budget_report": True,
            "broad_regression_required": True,
            "max_material_domain_drop": 0.02,
            "external_frontier": {
                "minimum_named_contemporaneous_providers": 2,
                "provider_versions_frozen_before_task_reveal": True,
                "same_information_and_tools": True,
                "separate_compute_and_latency_reporting": True,
            },
            "mechanism_attribution": mechanism_attribution,
            "engineering_canary": (
                {
                    "purpose": (
                        "prove real signed resident-32B optimizer admission, "
                        "policy mutation, durable checkpointing, base-checkpoint "
                        "immutability, and observed step latency"
                    ),
                    "training_tasks": len(RECURRENCE_TRAINING_FAMILIES),
                    "minimum_optimizer_updates": math.ceil(
                        len(RECURRENCE_TRAINING_FAMILIES) * 0.25
                    ),
                    "reasoning_gain_claim_eligible": False,
                    "frontier_claim_eligible": False,
                }
                if campaign_profile == UPDATE_CANARY_PROFILE
                else None
            ),
        },
        "hypotheses": {
            "positive_interaction": ("(adapter_rlc-adapter_vanilla) > (base_rlc-base_vanilla)"),
            "recurrent_execution_dividend": (
                "adapter_rlc > adapter_vanilla and adapter_equal_compute"
            ),
            "frontier_level": (
                "adapter_rlc meets or exceeds preregistered named frontier "
                "baselines without material domain regression"
            ),
        },
        "independent_custody": {
            "required_roles": [
                "task_issuer",
                "campaign_runner",
                "answer_revealer",
                "final_run_attestor",
                "independent_verifier",
                "contamination_auditor",
            ],
            "distinct_keys_and_organizations_required": True,
            "producer_private_key_access_disqualifies_claim": True,
            "external_trust_present": False,
        },
        "claim_state": {
            "mechanics_proven": True,
            "resident_training_complete": False,
            "positive_interaction_proven": False,
            "broad_reasoning_gain_proven": False,
            "frontier_level_proven": False,
            "release_eligible": False,
        },
        "required_stage_order": (
            [
                "verify_preregistration",
                "run_exact_signed_update_canary",
                "validate_recurrent_grpo_identity",
                "verify_base_checkpoint_immutability",
                "publish_canary_latency_and_admission_verdict",
                "retire_canary_adapter",
            ]
            if campaign_profile == UPDATE_CANARY_PROFILE
            else [
                "verify_preregistration",
                "train_or_exactly_resume_to_max_steps",
                "validate_recurrent_grpo_identity",
                "freeze_adapter",
                "externally_sign_contamination_audit",
                "directional_six_arm_factorial",
                "powered_confirmatory_six_arm_factorial",
                "broad_regression_battery",
                "named_external_frontier_comparison",
                "independent_replay_and_release_certificate",
            ]
        ),
    }
    return {**material, "contract_sha256": _document_sha(material)}


def validate_contract(contract: Mapping[str, Any], *, verify_model: bool = True) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "campaign_id",
        "campaign_profile",
        "committed_at",
        "launch_not_before",
        "launch_not_before_unix",
        "model",
        "execution_spec",
        "sources",
        "paths",
        "training",
        "evaluation",
        "hypotheses",
        "independent_custody",
        "claim_state",
        "required_stage_order",
        "contract_sha256",
    }
    if set(contract) != expected_keys or contract.get("schema") != CONTRACT_SCHEMA:
        _fail("contract_schema_invalid")
    material = dict(contract)
    claimed_sha = material.pop("contract_sha256")
    if claimed_sha != _document_sha(material):
        _fail("contract_digest_mismatch")
    campaign_id = contract.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id.startswith(
        "resident-32b-recurrent-grpo-cp"
    ):
        _fail("campaign_identity_mismatch")
    campaign_profile = contract.get("campaign_profile")
    if campaign_profile not in CAMPAIGN_PROFILES:
        _fail("campaign_profile_invalid")
    expected_parameters = _training_parameters_for_profile(str(campaign_profile))
    expected_watchdog = _watchdog_policy_for_profile(str(campaign_profile))
    expected_resource_envelope = _resource_envelope_for_profile(str(campaign_profile))
    try:
        committed = datetime.fromisoformat(str(contract["committed_at"]))
        not_before = datetime.fromisoformat(str(contract["launch_not_before"]))
    except ValueError as exc:
        raise PreregistrationError("contract_time_invalid") from exc
    if (
        committed.tzinfo is None
        or not_before.tzinfo is None
        or contract.get("launch_not_before") != NOT_BEFORE
        or contract.get("launch_not_before_unix") != int(not_before.timestamp())
    ):
        _fail("contract_time_invalid")
    sources = contract.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != set(SOURCE_ROLES):
        _fail("source_inventory_invalid")
    for role, relative in SOURCE_ROLES.items():
        if sources[role] != _binding(relative):
            _fail(f"source_{role}_mismatch")
    spec_record = contract.get("execution_spec")
    if not isinstance(spec_record, Mapping):
        _fail("execution_spec_binding_invalid")
    _spec, expected_spec = _load_spec(DEFAULT_SPEC)
    if dict(spec_record) != expected_spec:
        _fail("execution_spec_binding_mismatch")
    paths = contract.get("paths")
    training = contract.get("training")
    if not isinstance(paths, Mapping) or not isinstance(training, Mapping):
        _fail("training_contract_invalid")
    expected_argv = _training_argv(
        campaign_id=campaign_id,
        model=DEFAULT_MODEL,
        output=str(paths.get("training_output")),
        execution_spec=DEFAULT_SPEC,
        parameters=expected_parameters,
    )
    expected_trajectory_config = _verified_trajectory_config_commitment(
        _spec,
        parameters=expected_parameters,
    )
    expected_completion = {
        "schema": "aura.recurrent_grpo_training_completion.v1",
        "complete": True,
        "halt_reason": "max_steps",
        "causal_gain_proven": False,
        "training_adequacy": recurrent_training_adequacy_policy(),
    }
    if (
        training.get("campaign_profile") != campaign_profile
        or training.get("execution_mode") != "recurrent"
        or training.get("parameters") != expected_parameters
        or training.get("argv") != expected_argv
        or training.get("dataset") != _dataset_commitment(expected_parameters)
        or (
            expected_trajectory_config is None and "verified_trajectory_config_artifact" in training
        )
        or (
            expected_trajectory_config is not None
            and training.get("verified_trajectory_config_artifact") != expected_trajectory_config
        )
        or training.get("resume_contract") != "exact_identity_bound_checkpoint"
        or training.get("watchdog_policy") != expected_watchdog
        or training.get("completion_required") != expected_completion
        or training.get("resource_envelope") != expected_resource_envelope
    ):
        _fail("training_contract_mismatch")
    dataset = training["dataset"]
    if (
        dataset.get("families") != list(CURRENT_EXCLUDED_TRAINING_FAMILIES)
        or dataset.get("excluded_evaluation_families") != list(CURRENT_EXCLUDED_TRAINING_FAMILIES)
        or dataset.get("train_holdout_id_overlap") != 0
        or dataset.get("train_holdout_prompt_overlap") != 0
    ):
        _fail("training_evaluation_separation_invalid")
    evaluation = contract.get("evaluation")
    if not isinstance(evaluation, Mapping):
        _fail("evaluation_contract_invalid")
    confirmatory = evaluation.get("powered_confirmatory")
    expected_canary = (
        {
            "purpose": (
                "prove real signed resident-32B optimizer admission, "
                "policy mutation, durable checkpointing, base-checkpoint "
                "immutability, and observed step latency"
            ),
            "training_tasks": len(RECURRENCE_TRAINING_FAMILIES),
            "minimum_optimizer_updates": math.ceil(
                len(RECURRENCE_TRAINING_FAMILIES) * 0.25
            ),
            "reasoning_gain_claim_eligible": False,
            "frontier_claim_eligible": False,
        }
        if campaign_profile == UPDATE_CANARY_PROFILE
        else None
    )
    if (
        evaluation.get("campaign_profile") != campaign_profile
        or evaluation.get("claim_eligible") is not False
        or evaluation.get("registry_version") != CURRENT_REGISTRY_VERSION
        or evaluation.get("domains") != list(FRONTIER_DOMAINS)
        or not isinstance(confirmatory, Mapping)
        or confirmatory.get("observations_per_domain") != CONFIRMATORY_OBSERVATIONS_PER_DOMAIN
        or confirmatory.get("task_count")
        != len(FRONTIER_DOMAINS) * CONFIRMATORY_OBSERVATIONS_PER_DOMAIN
        or confirmatory.get("cell_count")
        != len(FRONTIER_DOMAINS) * CONFIRMATORY_OBSERVATIONS_PER_DOMAIN * 6
        or evaluation.get("engineering_canary") != expected_canary
    ):
        _fail("evaluation_power_invalid")
    claim_state = contract.get("claim_state")
    custody = contract.get("independent_custody")
    if (
        not isinstance(claim_state, Mapping)
        or claim_state.get("positive_interaction_proven") is not False
        or claim_state.get("frontier_level_proven") is not False
        or claim_state.get("release_eligible") is not False
        or not isinstance(custody, Mapping)
        or custody.get("external_trust_present") is not False
    ):
        _fail("prelaunch_claim_state_invalid")
    expected_stages = (
        [
            "verify_preregistration",
            "run_exact_signed_update_canary",
            "validate_recurrent_grpo_identity",
            "verify_base_checkpoint_immutability",
            "publish_canary_latency_and_admission_verdict",
            "retire_canary_adapter",
        ]
        if campaign_profile == UPDATE_CANARY_PROFILE
        else [
            "verify_preregistration",
            "train_or_exactly_resume_to_max_steps",
            "validate_recurrent_grpo_identity",
            "freeze_adapter",
            "externally_sign_contamination_audit",
            "directional_six_arm_factorial",
            "powered_confirmatory_six_arm_factorial",
            "broad_regression_battery",
            "named_external_frontier_comparison",
            "independent_replay_and_release_certificate",
        ]
    )
    if contract.get("required_stage_order") != expected_stages:
        _fail("required_stage_order_invalid")
    model = contract.get("model")
    if not isinstance(model, Mapping) or model.get("path") != DEFAULT_MODEL:
        _fail("model_contract_invalid")
    if verify_model:
        model_path = _repo_path(DEFAULT_MODEL, role="model")
        if model.get("base_checkpoint") != full_weight_checkpoint_identity(model_path):
            _fail("model_checkpoint_identity_mismatch")
        if model.get("behavior_bundle") != model_behavior_bundle_identity(model_path):
            _fail("model_behavior_identity_mismatch")
    return {
        "schema": "aura.resident_recurrent_grpo_preregistration_receipt.v1",
        "campaign_id": campaign_id,
        "campaign_profile": campaign_profile,
        "contract_sha256": claimed_sha,
        "model_verified": bool(verify_model),
        "training_tasks": dataset["train_tasks"],
        "holdout_tasks": dataset["holdout_tasks"],
        "confirmatory_tasks": confirmatory["task_count"],
        "confirmatory_cells": confirmatory["cell_count"],
        "claim_eligible": False,
    }


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values or not 0.0 < percentile <= 1.0:
        _fail("update_canary_latency_input_invalid")
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return ordered[index]


def _wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if (
        type(successes) is not int
        or type(total) is not int
        or total <= 0
        or not 0 <= successes <= total
        or not math.isfinite(z)
        or z <= 0.0
    ):
        _fail("update_canary_admission_interval_input_invalid")
    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def build_update_canary_verdict(
    contract: Mapping[str, Any],
    *,
    verify_model: bool = True,
) -> dict[str, Any]:
    """Recompute the bounded resident update canary from durable artifacts."""

    validation = validate_contract(contract, verify_model=verify_model)
    if contract.get("campaign_profile") != UPDATE_CANARY_PROFILE:
        _fail("update_canary_profile_required")
    parameters = contract["training"]["parameters"]
    max_steps = int(parameters["max_steps"])
    training_root = _repo_path(
        str(contract["paths"]["training_output"]),
        role="training_output",
    )
    from core.brain.llm.latent_cortex.recurrent_grpo_adapter_identity import (
        MANIFEST_FILE,
    )

    completion = _strict_json(training_root / "training_completion.json")
    receipt = _strict_json(training_root / "grpo_receipt.json")
    protocol = _strict_json(training_root / "training_protocol.json")
    manifest = _strict_json(training_root / MANIFEST_FILE)
    if (
        completion.get("schema") != "aura.recurrent_grpo_training_completion.v1"
        or completion.get("complete") is not True
        or completion.get("halt_reason") != "max_steps"
        or completion.get("step") != max_steps
        or receipt.get("adapter_id") != contract["campaign_id"]
        or receipt.get("steps") != max_steps
        or receipt.get("termination")
        != {
            "reason": "max_steps",
            "completed_budget": True,
            "signal": None,
        }
    ):
        _fail("update_canary_training_incomplete")
    adequacy = receipt.get("training_adequacy")
    if (
        not isinstance(adequacy, Mapping)
        or adequacy.get("policy") != recurrent_training_adequacy_policy()
        or adequacy.get("admitted") is not True
    ):
        _fail("update_canary_training_adequacy_failed")
    steps = receipt.get("step_receipts")
    if not isinstance(steps, list) or len(steps) != max_steps:
        _fail("update_canary_step_receipts_incomplete")
    update_steps = [
        step for step in steps if step.get("step_kind") == "verified_optimizer_update"
    ]
    minimum_updates = math.ceil(max_steps * 0.25)
    if (
        receipt.get("optimizer_updates") != len(update_steps)
        or len(update_steps) < minimum_updates
    ):
        _fail("update_canary_optimizer_admission_insufficient")
    prior_policy: str | None = None
    for index, step in enumerate(steps, start=1):
        before = step.get("policy_before_sha256")
        after = step.get("policy_after_sha256")
        kind = step.get("step_kind")
        if (
            step.get("step") != index
            or not isinstance(before, str)
            or not isinstance(after, str)
            or (prior_policy is not None and before != prior_policy)
            or (kind == "verified_optimizer_update" and before == after)
            or (kind == "verified_rejected_group" and before != after)
            or kind not in {"verified_optimizer_update", "verified_rejected_group"}
        ):
            _fail("update_canary_policy_lineage_invalid")
        prior_policy = after

    from tools.train_grpo import _validate_published_recurrent_bundle

    identity = _validate_published_recurrent_bundle(
        training_root,
        adapter_id=str(contract["campaign_id"]),
        base_identity=contract["model"]["base_checkpoint"],
        behavior_identity=contract["model"]["behavior_bundle"],
        personality_identity=protocol["personality_adapter"],
        runtime_identity=protocol["runtime"],
    )
    if identity.get("adapter_sha256") != completion.get("adapter_sha256"):
        _fail("update_canary_adapter_identity_mismatch")
    non_promotable = _strict_json(training_root / "NON_PROMOTABLE_CANARY.json")
    if (
        non_promotable.get("schema") != "aura.recurrent_grpo.non_promotable_canary.v1"
        or non_promotable.get("adapter_id") != contract["campaign_id"]
        or non_promotable.get("adapter_sha256") != completion["adapter_sha256"]
        or non_promotable.get("runtime_promotion_allowed") is not False
        or non_promotable.get("reasoning_gain_proven") is not False
        or non_promotable.get("frontier_level_proven") is not False
    ):
        _fail("update_canary_non_promotion_marker_invalid")

    artifact_root = _repo_path(
        str(contract["paths"]["artifact_root"]),
        role="artifact_root",
    )
    initial_adapter = artifact_root / "verified-launch" / "initial_adapter.safetensors"
    final_adapter = training_root / str(manifest["adapter"]["path"])
    if (
        initial_adapter.is_symlink()
        or final_adapter.is_symlink()
        or not initial_adapter.is_file()
        or not final_adapter.is_file()
        or _sha256(initial_adapter.read_bytes()) == _sha256(final_adapter.read_bytes())
    ):
        _fail("update_canary_adapter_did_not_change")

    pointer = _strict_json(training_root / "latest.json")
    checkpoint_relative = pointer.get("checkpoint")
    if (
        not isinstance(checkpoint_relative, str)
        or PurePosixPath(checkpoint_relative).parent != PurePosixPath("checkpoints")
    ):
        _fail("update_canary_checkpoint_pointer_invalid")
    checkpoint = _strict_json(training_root / checkpoint_relative / "complete.json")
    if checkpoint.get("step") != max_steps:
        _fail("update_canary_terminal_checkpoint_missing")

    launch_bundle = _strict_json(
        _repo_path(
            str(contract["paths"]["verified_launch_bundle"]),
            role="verified_launch_bundle",
        )
    )
    campaign_root = _repo_path(
        str(launch_bundle.get("campaign_ledger_root", "")),
        role="campaign_ledger_root",
    )
    if not (campaign_root / "campaign.closed.json").is_file():
        _fail("update_canary_campaign_not_closed")
    campaign_durations: list[float] = []
    checkpoint_durations: list[float] = []
    statuses: list[str] = []
    for sequence in range(max_steps):
        started = _strict_json(campaign_root / f"group-{sequence:08d}.started.json")
        terminal = _strict_json(campaign_root / f"group-{sequence:08d}.terminal.json")
        admitted_ns = started.get("admitted_at_unix_ns")
        finished_ns = terminal.get("finished_at_unix_ns")
        if (
            started.get("sequence") != sequence
            or terminal.get("sequence") != sequence
            or type(admitted_ns) is not int
            or type(finished_ns) is not int
            or finished_ns < admitted_ns
            or terminal.get("policy_before_sha256")
            != steps[sequence].get("policy_before_sha256")
            or terminal.get("policy_after_sha256")
            != steps[sequence].get("policy_after_sha256")
        ):
            _fail("update_canary_campaign_timing_invalid")
        statuses.append(str(terminal.get("status")))
        campaign_durations.append((finished_ns - admitted_ns) / 1_000_000_000.0)
        timing_path = (
            training_root
            / "update-canary-step-timings"
            / f"step-{sequence + 1:08d}.json"
        )
        timing_raw = read_stable_bytes(timing_path, max_bytes=64 * 1024)
        timing = json.loads(timing_raw)
        timing_material = dict(timing)
        timing_claim = timing_material.pop("receipt_sha256", None)
        checkpoint_relative = timing.get("checkpoint")
        if not isinstance(checkpoint_relative, str):
            _fail("update_canary_checkpoint_timing_invalid")
        checkpoint_complete_path = training_root / checkpoint_relative / "complete.json"
        elapsed_ns = timing.get("elapsed_monotonic_ns")
        if (
            timing.get("schema")
            != "aura.recurrent_grpo.update_canary_step_timing.v1"
            or timing.get("adapter_id") != contract["campaign_id"]
            or timing.get("protocol_sha256") != _sha256(
                (training_root / "training_protocol.json").read_bytes()
            )
            or timing.get("step") != sequence + 1
            or timing.get("task_id") != steps[sequence].get("task_id")
            or timing.get("step_receipt_sha256")
            != steps[sequence].get("receipt_sha256")
            or timing.get("includes_durable_checkpoint_publication") is not True
            or type(elapsed_ns) is not int
            or elapsed_ns <= 0
            or timing_claim != _sha256(canonical_json_bytes(timing_material))
            or checkpoint_complete_path.is_symlink()
            or not checkpoint_complete_path.is_file()
            or timing.get("checkpoint_complete_sha256")
            != _sha256(checkpoint_complete_path.read_bytes())
        ):
            _fail("update_canary_checkpoint_timing_invalid")
        checkpoint_durations.append(elapsed_ns / 1_000_000_000.0)

    base_checkpoint_unchanged = True
    if verify_model:
        model_path = _repo_path(str(contract["model"]["path"]), role="model")
        base_checkpoint_unchanged = (
            full_weight_checkpoint_identity(model_path)
            == contract["model"]["base_checkpoint"]
        )
    if not base_checkpoint_unchanged:
        _fail("update_canary_base_checkpoint_changed")
    containment = _strict_json(training_root / "CANARY_CONTAINMENT.json")
    containment_material = dict(containment)
    containment_claim = containment_material.pop("receipt_sha256", None)
    if (
        containment.get("schema")
        != "aura.resident_recurrent_grpo.canary_containment.v1"
        or containment.get("campaign_id") != contract["campaign_id"]
        or containment.get("campaign_contract_sha256") != contract["contract_sha256"]
        or containment.get("runtime_model_state_released") is not True
        or containment.get("runtime_promotion_allowed") is not False
        or containment.get("base_checkpoint_sha256")
        != contract["model"]["base_checkpoint"]["fingerprint"]
        or containment_claim != _sha256(canonical_json_bytes(containment_material))
    ):
        _fail("update_canary_containment_invalid")

    interval_low, interval_high = _wilson_interval(len(update_steps), max_steps)
    body = {
        "schema": UPDATE_CANARY_VERDICT_SCHEMA,
        "campaign_id": contract["campaign_id"],
        "campaign_contract_sha256": contract["contract_sha256"],
        "campaign_profile": UPDATE_CANARY_PROFILE,
        "resident_model_verified": bool(verify_model),
        "base_checkpoint_unchanged": base_checkpoint_unchanged,
        "steps": max_steps,
        "optimizer_updates": len(update_steps),
        "optimizer_update_fraction": round(len(update_steps) / max_steps, 6),
        "optimizer_update_fraction_wilson_95": {
            "low": round(interval_low, 6),
            "high": round(interval_high, 6),
        },
        "minimum_optimizer_updates": minimum_updates,
        "updated_groups": statuses.count("updated"),
        "rejected_groups": statuses.count("rejected"),
        "policy_initial_sha256": steps[0]["policy_before_sha256"],
        "policy_final_sha256": steps[-1]["policy_after_sha256"],
        "adapter_sha256": completion["adapter_sha256"],
        "adapter_identity_sha256": identity["composite_identity_sha256"],
        "process_containment_rollback": True,
        "latency_s": {
            "scope": "task_admission_through_durable_checkpoint_publication",
            "count": len(checkpoint_durations),
            "p50": round(_nearest_rank(checkpoint_durations, 0.5), 6),
            "p90": round(_nearest_rank(checkpoint_durations, 0.9), 6),
            "max": round(max(checkpoint_durations), 6),
            "total": round(sum(checkpoint_durations), 6),
        },
        "campaign_execution_latency_s": {
            "scope": "signed_group_admission_through_campaign_terminal",
            "count": len(campaign_durations),
            "p50": round(_nearest_rank(campaign_durations, 0.5), 6),
            "p90": round(_nearest_rank(campaign_durations, 0.9), 6),
            "max": round(max(campaign_durations), 6),
            "total": round(sum(campaign_durations), 6),
        },
        "verdict": "pass",
        "full_training_launch_ready": True,
        "reasoning_gain_proven": False,
        "frontier_level_proven": False,
        "claim_boundary": (
            "engineering_update_and_latency_canary_only; "
            "no capability or generalization claim"
        ),
        "preregistration_validation": validation,
    }
    return {**body, "verdict_sha256": _document_sha(body)}


def _checkpoint_binding(path: Path, expected: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"resume_{role}_path_invalid")
    raw = path.read_bytes()
    if (
        set(expected) != {"path", "sha256", "size_bytes"}
        or expected.get("path") != path.name
        or expected.get("sha256") != _sha256(raw)
        or expected.get("size_bytes") != len(raw)
    ):
        _fail(f"resume_{role}_binding_mismatch")
    return {
        "path": path.name,
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
    }


def build_resume_verdict(
    contract: Mapping[str, Any],
    *,
    environment: Mapping[str, str],
    verify_model: bool = True,
) -> dict[str, Any]:
    """Prove one committed trainer generation is safe to resume."""
    validate_contract(contract, verify_model=verify_model)
    required_environment = {
        "AURA_DETACHED_PLAN_SHA256",
        "AURA_DETACHED_COMMAND_SHA256",
        "AURA_DETACHED_PRIOR_ATTEMPT",
        "AURA_DETACHED_PRIOR_JOURNAL_HEAD_SHA256",
        "AURA_DETACHED_RESUME_EVIDENCE_PATH",
    }
    if any(not environment.get(key) for key in required_environment):
        _fail("resume_environment_incomplete")
    try:
        prior_attempt = int(environment["AURA_DETACHED_PRIOR_ATTEMPT"])
    except ValueError as exc:
        raise PreregistrationError("resume_attempt_invalid") from exc
    if prior_attempt < 1:
        _fail("resume_attempt_invalid")
    training_root = _repo_path(
        str(contract["paths"]["training_output"]),
        role="training_output",
    )
    if (training_root / "training_completion.json").exists():
        _fail("resume_training_already_completed")
    protocol_raw = (training_root / "training_protocol.json").read_bytes()
    dataset_raw = (training_root / "dataset_manifest.json").read_bytes()
    if _sha256(dataset_raw) != contract["training"]["dataset"]["sha256"]:
        _fail("resume_dataset_mismatch")
    pointer = _strict_json(training_root / "latest.json")
    if set(pointer) != {"schema", "checkpoint", "complete_sha256"}:
        _fail("resume_pointer_schema_invalid")
    relative = pointer.get("checkpoint")
    if (
        pointer.get("schema") != "aura.grpo_checkpoint_pointer.v1"
        or not isinstance(relative, str)
        or PurePosixPath(relative).parent != PurePosixPath("checkpoints")
    ):
        _fail("resume_pointer_invalid")
    checkpoint = (training_root / relative).resolve(strict=True)
    if checkpoint.parent != (training_root / "checkpoints").resolve(strict=True):
        _fail("resume_checkpoint_path_invalid")
    complete_path = checkpoint / "complete.json"
    complete_raw = complete_path.read_bytes()
    if pointer.get("complete_sha256") != _sha256(complete_raw):
        _fail("resume_checkpoint_completion_mismatch")
    complete = _strict_json(complete_path)
    step = complete.get("step")
    if (
        complete.get("schema") != "aura.grpo_checkpoint.v2"
        or complete.get("checkpoint_id") != checkpoint.name
        or type(step) is not int
        or step < 0
        or complete.get("last_step_committed") is not True
        or complete.get("protocol_sha256") != _sha256(protocol_raw)
        or complete.get("dataset_sha256") != _sha256(dataset_raw)
        or complete.get("execution_mode") != "recurrent"
        or complete.get("execution_spec_sha256") != contract["execution_spec"]["semantic_sha256"]
    ):
        _fail("resume_checkpoint_state_invalid")
    adapter = _checkpoint_binding(
        checkpoint / str(complete.get("adapter", {}).get("path", "")),
        complete.get("adapter", {}),
        role="adapter",
    )
    optimizer = _checkpoint_binding(
        checkpoint / str(complete.get("optimizer", {}).get("path", "")),
        complete.get("optimizer", {}),
        role="optimizer",
    )
    plan_sha = environment["AURA_DETACHED_PLAN_SHA256"]
    command_sha = environment["AURA_DETACHED_COMMAND_SHA256"]
    journal_head = environment["AURA_DETACHED_PRIOR_JOURNAL_HEAD_SHA256"]
    if any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in (plan_sha, command_sha, journal_head)
    ):
        _fail("resume_supervisor_binding_invalid")
    evidence_path = Path(environment["AURA_DETACHED_RESUME_EVIDENCE_PATH"])
    if not evidence_path.is_absolute() or evidence_path.is_symlink():
        _fail("resume_evidence_path_invalid")
    evidence = {
        "schema": "aura.detached_step.resume_evidence.v1",
        "plan_sha256": plan_sha,
        "command_sha256": command_sha,
        "prior_attempt": prior_attempt,
        "prior_journal_head_sha256": journal_head,
        "checkpoint_sequence": step,
        "campaign_contract_sha256": contract["contract_sha256"],
        "training_protocol_sha256": _sha256(protocol_raw),
        "dataset_sha256": _sha256(dataset_raw),
        "checkpoint_complete_sha256": _sha256(complete_raw),
        "adapter": adapter,
        "optimizer": optimizer,
    }
    evidence_raw = canonical_json_bytes(evidence)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    if not atomic_write_bytes_if_absent(evidence_path, evidence_raw, mode=0o600):
        if evidence_path.read_bytes() != evidence_raw:
            _fail("resume_evidence_publication_raced")
    evidence_sha = _sha256(evidence_raw)
    checkpoint_identity = _sha256(
        _bare_canonical(
            {
                "prior_attempt": prior_attempt,
                "prior_journal_head_sha256": journal_head,
                "checkpoint_sequence": step,
                "evidence_sha256": evidence_sha,
            }
        )
    )
    return {
        "schema": "aura.detached_step.resume_verdict.v2",
        "plan_sha256": plan_sha,
        "command_sha256": command_sha,
        "prior_attempt": prior_attempt,
        "prior_journal_head_sha256": journal_head,
        "checkpoint_sequence": step,
        "checkpoint_identity": checkpoint_identity,
        "verdict": "safe_to_resume",
        "evidence_path": str(evidence_path),
        "evidence_sha256": evidence_sha,
        "evidence": evidence,
    }


def _write_once(path: Path, document: Mapping[str, Any]) -> None:
    destination = path.expanduser()
    if not destination.is_absolute():
        destination = REPO_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(document)
    if destination.exists():
        if destination.is_symlink() or destination.read_bytes() != payload:
            _fail("contract_exists_different")
        return
    if not atomic_write_bytes_if_absent(destination, payload, mode=0o600):
        _fail("contract_publication_raced")


def _training_progress_snapshot(training_root: Path) -> dict[str, Any]:
    """Bind watchdog decisions to durable trainer progress, not elapsed time."""

    files: dict[str, str] = {}
    for relative in (
        "latest.json",
        "baseline-progress.json",
        "calibration-progress.json",
        "training_completion.json",
        "grpo_receipt.json",
    ):
        path = training_root / relative
        if path.is_file() and not path.is_symlink():
            files[relative] = _sha256(read_stable_bytes(path, max_bytes=32 * 1024 * 1024))
    checkpoint_step: int | None = None
    latest = training_root / "latest.json"
    if latest.is_file() and not latest.is_symlink():
        pointer = _strict_json(latest)
        relative = pointer.get("checkpoint")
        if isinstance(relative, str):
            complete_path = training_root / relative / "complete.json"
            if complete_path.is_file() and not complete_path.is_symlink():
                complete = _strict_json(complete_path)
                if type(complete.get("step")) is int:
                    checkpoint_step = int(complete["step"])
                files[f"{relative}/complete.json"] = _sha256(
                    read_stable_bytes(complete_path, max_bytes=1024 * 1024)
                )
    document = {
        "schema": "aura.resident_recurrent_grpo.training_progress.v1",
        "checkpoint_step": checkpoint_step,
        "files": dict(sorted(files.items())),
    }
    return {**document, "sha256": _sha256(canonical_json_bytes(document))}


def _successful_training_disposition(
    contract: Mapping[str, Any],
    *,
    training_root: Path,
    progress: Mapping[str, Any],
) -> str:
    """Refuse to equate a zero process exit with the complete training dose."""

    max_steps = int(contract["training"]["parameters"]["max_steps"])
    completion_path = training_root / "training_completion.json"
    if completion_path.is_file() and not completion_path.is_symlink():
        completion = _strict_json(completion_path)
        required = contract["training"]["completion_required"]
        if (
            completion.get("schema") != required["schema"]
            or completion.get("complete") is not True
            or completion.get("halt_reason") != "max_steps"
            or completion.get("step") != max_steps
        ):
            _fail("training_completion_not_admissible")
        return "complete"

    receipt_path = training_root / "grpo_receipt.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        _fail("successful_training_exit_without_receipt")
    receipt = _strict_json(receipt_path)
    termination = receipt.get("termination")
    steps = receipt.get("steps")
    reason = termination.get("reason") if isinstance(termination, Mapping) else None
    if (
        not isinstance(termination, Mapping)
        or reason not in {"wall_clock_budget", "operator_pause"}
        or termination.get("completed_budget") is not False
        or type(steps) is not int
        or not 0 <= steps < max_steps
        or progress.get("checkpoint_step") != steps
    ):
        _fail("successful_training_exit_without_complete_dose")
    return "paused" if reason == "operator_pause" else "resume"


def _watchdog_documents(
    contract: Mapping[str, Any],
) -> tuple[Path, Path, Path, list[dict[str, Any]]]:
    root = ensure_private_directory(
        _repo_path(
            str(PurePosixPath(str(contract["paths"]["artifact_root"])) / "training-watchdog"),
            role="training_watchdog",
            must_exist=False,
        )
    )
    journal_path = root / "attempts.json"
    status_path = root / "status.json"
    if not journal_path.exists():
        return root, journal_path, status_path, []
    raw = read_stable_bytes(journal_path, max_bytes=4 * 1024 * 1024)
    try:
        document = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise PreregistrationError("training_watchdog_journal_invalid") from exc
    if (
        not isinstance(document, dict)
        or set(document) != {"schema", "campaign_id", "contract_sha256", "records"}
        or document.get("schema") != "aura.resident_recurrent_grpo.training_watchdog_journal.v1"
        or document.get("campaign_id") != contract["campaign_id"]
        or document.get("contract_sha256") != contract["contract_sha256"]
        or not isinstance(document.get("records"), list)
        or any(not isinstance(record, dict) for record in document["records"])
        or canonical_json_bytes(document) != raw
    ):
        _fail("training_watchdog_journal_invalid")
    return root, journal_path, status_path, [dict(record) for record in document["records"]]


def _write_watchdog_documents(
    contract: Mapping[str, Any],
    *,
    journal_path: Path,
    status_path: Path,
    records: Sequence[Mapping[str, Any]],
    status: Mapping[str, Any],
) -> None:
    journal = {
        "schema": "aura.resident_recurrent_grpo.training_watchdog_journal.v1",
        "campaign_id": contract["campaign_id"],
        "contract_sha256": contract["contract_sha256"],
        "records": [dict(record) for record in records],
    }
    atomic_write_bytes(journal_path, canonical_json_bytes(journal), mode=0o600)
    atomic_write_bytes(status_path, canonical_json_bytes(dict(status)), mode=0o600)


def _request_training_pause(contract: Mapping[str, Any]) -> dict[str, Any]:
    validate_contract(contract, verify_model=True)
    training_root = _repo_path(
        str(contract["paths"]["training_output"]),
        role="training_output",
    )
    protocol_path = training_root / "training_protocol.json"
    protocol_raw = read_stable_bytes(protocol_path, max_bytes=16 * 1024 * 1024)
    protocol = json.loads(protocol_raw)
    if (
        not isinstance(protocol, dict)
        or protocol.get("adapter_id") != contract["campaign_id"]
    ):
        _fail("training_pause_protocol_mismatch")
    body = {
        "schema": "aura.grpo.operator_pause_request.v1",
        "adapter_id": contract["campaign_id"],
        "protocol_sha256": _sha256(protocol_raw),
        "requested_at_unix_ns": time.time_ns(),
    }
    request = {**body, "request_sha256": _sha256(canonical_json_bytes(body))}
    path = training_root / "pause.request.json"
    payload = canonical_json_bytes(request)
    if path.exists():
        existing = _strict_json(path)
        if (
            existing.get("schema") != request["schema"]
            or existing.get("adapter_id") != request["adapter_id"]
            or existing.get("protocol_sha256") != request["protocol_sha256"]
        ):
            _fail("training_pause_request_conflict")
        return existing
    if not atomic_write_bytes_if_absent(path, payload, mode=0o600):
        _fail("training_pause_request_publication_raced")
    return request


def _training_resume_request_path(contract: Mapping[str, Any]) -> Path:
    return _repo_path(
        str(
            PurePosixPath(str(contract["paths"]["artifact_root"]))
            / "training-watchdog"
            / "resume.request.json"
        ),
        role="training_resume_request",
        must_exist=False,
    )


def _request_training_resume(contract: Mapping[str, Any]) -> dict[str, Any]:
    validate_contract(contract, verify_model=True)
    _root, _journal_path, status_path, _records = _watchdog_documents(contract)
    status = _strict_json(status_path)
    if status.get("state") != "paused":
        _fail("training_is_not_paused")
    body = {
        "schema": "aura.resident_recurrent_grpo.resume_request.v1",
        "campaign_id": contract["campaign_id"],
        "contract_sha256": contract["contract_sha256"],
        "paused_attempt": status.get("attempt"),
        "requested_at_unix_ns": time.time_ns(),
    }
    request = {**body, "request_sha256": _sha256(canonical_json_bytes(body))}
    path = _training_resume_request_path(contract)
    if not atomic_write_bytes_if_absent(
        path,
        canonical_json_bytes(request),
        mode=0o600,
    ):
        existing = _strict_json(path)
        if (
            existing.get("campaign_id") != request["campaign_id"]
            or existing.get("contract_sha256") != request["contract_sha256"]
        ):
            _fail("training_resume_request_conflict")
        return existing
    return request


def _wait_for_training_resume(contract: Mapping[str, Any]) -> dict[str, Any]:
    request_path = _training_resume_request_path(contract)
    while True:
        if request_path.exists():
            request = _strict_json(request_path)
            material = dict(request)
            claimed = material.pop("request_sha256", None)
            if (
                set(material)
                != {
                    "schema",
                    "campaign_id",
                    "contract_sha256",
                    "paused_attempt",
                    "requested_at_unix_ns",
                }
                or material.get("schema")
                != "aura.resident_recurrent_grpo.resume_request.v1"
                or material.get("campaign_id") != contract["campaign_id"]
                or material.get("contract_sha256") != contract["contract_sha256"]
                or claimed != _sha256(canonical_json_bytes(material))
            ):
                _fail("training_resume_request_invalid")
            receipt_root = ensure_private_directory(request_path.parent / "resume-receipts")
            receipt_path = receipt_root / f"{claimed}.json"
            _write_once(receipt_path, request)
            request_path.unlink()
            return request
        time.sleep(1.0)


def _release_failed_training_runtime() -> None:
    """Release resident MLX state before an exact checkpoint resume attempt."""

    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except (ImportError, RuntimeError):
        pass
    gc.collect()


def _publish_canary_containment_receipt(
    contract: Mapping[str, Any],
    *,
    training_root: Path,
) -> dict[str, Any]:
    if contract.get("campaign_profile") != UPDATE_CANARY_PROFILE:
        _fail("canary_containment_profile_invalid")
    completion_path = training_root / "training_completion.json"
    marker_path = training_root / "NON_PROMOTABLE_CANARY.json"
    body = {
        "schema": "aura.resident_recurrent_grpo.canary_containment.v1",
        "campaign_id": contract["campaign_id"],
        "campaign_contract_sha256": contract["contract_sha256"],
        "training_completion_sha256": _sha256(
            read_stable_bytes(completion_path, max_bytes=1024 * 1024)
        ),
        "non_promotable_marker_sha256": _sha256(
            read_stable_bytes(marker_path, max_bytes=1024 * 1024)
        ),
        "base_checkpoint_sha256": contract["model"]["base_checkpoint"][
            "fingerprint"
        ],
        "runtime_model_state_released": True,
        "runtime_promotion_allowed": False,
        "released_at_unix_ns": time.time_ns(),
    }
    receipt = {**body, "receipt_sha256": _sha256(canonical_json_bytes(body))}
    path = training_root / "CANARY_CONTAINMENT.json"
    payload = canonical_json_bytes(receipt)
    if path.exists():
        existing = _strict_json(path)
        if (
            existing.get("campaign_id") != receipt["campaign_id"]
            or existing.get("campaign_contract_sha256")
            != receipt["campaign_contract_sha256"]
            or existing.get("training_completion_sha256")
            != receipt["training_completion_sha256"]
            or existing.get("non_promotable_marker_sha256")
            != receipt["non_promotable_marker_sha256"]
            or existing.get("runtime_model_state_released") is not True
            or existing.get("runtime_promotion_allowed") is not False
        ):
            _fail("canary_containment_receipt_conflict")
        return existing
    if not atomic_write_bytes_if_absent(path, payload, mode=0o600):
        _fail("canary_containment_receipt_publication_raced")
    return receipt


def _run_training(
    contract: Mapping[str, Any],
    *,
    expected_launch_bundle_sha256: str,
) -> int:
    validate_contract(contract, verify_model=True)
    not_before = int(contract["launch_not_before_unix"])
    if time.time() < not_before:
        _fail("presentation_window_still_active")
    from tools import run_verified_recurrent_grpo_training

    argv = list(contract["training"]["argv"])
    bundle_path = _repo_path(
        str(contract["paths"]["verified_launch_bundle"]),
        role="verified_launch_bundle",
    )
    bundle_digest = str(expected_launch_bundle_sha256)
    if len(bundle_digest) != 64 or any(
        character not in "0123456789abcdef" for character in bundle_digest
    ):
        _fail("verified_launch_bundle_digest_invalid")
    informational_digest_path = _repo_path(
        str(contract["paths"]["verified_launch_bundle_sha256"]),
        role="verified_launch_bundle_sha256",
    )
    informational_digest = informational_digest_path.read_text(encoding="ascii").strip()
    if informational_digest != bundle_digest:
        _fail("verified_launch_bundle_external_digest_mismatch")
    training_root = _repo_path(
        str(contract["paths"]["training_output"]),
        role="training_output",
        must_exist=False,
    )
    _root, journal_path, status_path, records = _watchdog_documents(contract)
    policy = contract["training"]["watchdog_policy"]
    max_attempts = int(policy["max_attempts"])
    no_progress_limit = int(policy["max_consecutive_no_progress_failures"])
    consecutive_no_progress = 0
    if records:
        consecutive_no_progress = int(records[-1].get("consecutive_no_progress_failures") or 0)
    if len(records) >= max_attempts:
        _fail("training_watchdog_attempt_budget_exhausted")
    invocation = [
        "--verified-launch-bundle",
        str(bundle_path),
        "--expected-launch-bundle-sha256",
        bundle_digest,
        "--expected-preregistration-sha256",
        str(contract["contract_sha256"]),
        *argv[1:],
    ]
    while len(records) < max_attempts:
        attempt = len(records) + 1
        before = _training_progress_snapshot(training_root)
        started_at = time.time()
        status = {
            "schema": "aura.resident_recurrent_grpo.training_watchdog_status.v1",
            "campaign_id": contract["campaign_id"],
            "contract_sha256": contract["contract_sha256"],
            "state": "running",
            "attempt": attempt,
            "max_attempts": max_attempts,
            "started_at": started_at,
            "progress": before,
        }
        _write_watchdog_documents(
            contract,
            journal_path=journal_path,
            status_path=status_path,
            records=records,
            status=status,
        )
        error: Exception | None = None
        result = 1
        try:
            result = int(run_verified_recurrent_grpo_training.main(invocation))
        except Exception as exc:
            error = exc
        after = _training_progress_snapshot(training_root)
        disposition: str | None = None
        if error is None and result == 0:
            try:
                disposition = _successful_training_disposition(
                    contract,
                    training_root=training_root,
                    progress=after,
                )
            except Exception as exc:
                error = exc
        progressed = after["sha256"] != before["sha256"]
        consecutive_no_progress = 0 if progressed else consecutive_no_progress + 1
        record = {
            "attempt": attempt,
            "started_at": started_at,
            "finished_at": time.time(),
            "returncode": result if error is None else 1,
            "error_type": type(error).__name__ if error is not None else None,
            "error": str(error)[:1000] if error is not None else None,
            "progress_before": before,
            "progress_after": after,
            "durable_progress": progressed,
            "disposition": disposition,
            "consecutive_no_progress_failures": consecutive_no_progress,
        }
        records.append(record)
        terminal_state = (
            "complete"
            if error is None and result == 0 and disposition == "complete"
            else "paused"
            if error is None and result == 0 and disposition == "paused"
            else "exhausted"
            if len(records) >= max_attempts
            or consecutive_no_progress >= no_progress_limit
            else "retry_wait"
        )
        _write_watchdog_documents(
            contract,
            journal_path=journal_path,
            status_path=status_path,
            records=records,
            status={
                **status,
                "state": terminal_state,
                "finished_at": record["finished_at"],
                "progress": after,
                "last_error_type": record["error_type"],
                "last_error": record["error"],
                "consecutive_no_progress_failures": consecutive_no_progress,
            },
        )
        if terminal_state == "complete":
            if contract.get("campaign_profile") == UPDATE_CANARY_PROFILE:
                _release_failed_training_runtime()
                _publish_canary_containment_receipt(
                    contract,
                    training_root=training_root,
                )
            break
        if terminal_state == "paused":
            _release_failed_training_runtime()
            _wait_for_training_resume(contract)
            consecutive_no_progress = 0
            continue
        if terminal_state == "exhausted":
            if error is not None:
                raise error
            if disposition == "resume":
                _fail("training_watchdog_attempt_budget_exhausted")
            return int(result)
        _release_failed_training_runtime()
        time.sleep(float(policy["retry_backoff_s"]))

    produced_dataset = (training_root / "dataset_manifest.json").read_bytes()
    if _sha256(produced_dataset) != contract["training"]["dataset"]["sha256"]:
        _fail("produced_dataset_commitment_mismatch")
    return 0


def _policy_probe_argv(contract: Mapping[str, Any]) -> list[str]:
    argv = list(contract["training"]["argv"])
    output_index = argv.index("--out-dir") + 1
    argv[output_index] = str(contract["paths"]["initial_policy_probe"])
    if "--verified-trajectory-config" in argv:
        config_index = argv.index("--verified-trajectory-config")
        if config_index + 1 >= len(argv):
            _fail("verified_trajectory_config_argv_invalid")
        del argv[config_index : config_index + 2]
    if "--fixed-update-canary" in argv:
        argv.remove("--fixed-update-canary")
    argv.append("--initial-policy-probe")
    return argv


def _run_initial_policy_probe(contract: Mapping[str, Any]) -> int:
    validate_contract(contract, verify_model=True)
    from tools import train_grpo

    argv = _policy_probe_argv(contract)
    previous = list(sys.argv)
    try:
        sys.argv = [argv[0], *argv[1:]]
        return int(train_grpo.main())
    finally:
        sys.argv = previous


def _answer_channel_preflight_argv(contract: Mapping[str, Any]) -> list[str]:
    """Small resident gate for verifier-entry before long recurrent GRPO."""

    root = PurePosixPath(str(contract["paths"]["artifact_root"]))
    output = str(root / "answer-channel-preflight")
    params = contract["training"]["parameters"]
    max_tokens = min(160, int(params["max_tokens"]))
    return [
        "tools/train_grpo.py",
        "--model",
        str(contract["model"]["path"]),
        "--out-dir",
        output,
        "--adapter-id",
        f"{contract['campaign_id']}-answer-channel-preflight",
        "--execution-mode",
        "recurrent",
        "--execution-spec",
        str(contract["execution_spec"]["path"]),
        "--task-source",
        "answer_channel_curriculum",
        "--domains",
        "json_copy,typed_boolean,key_selection",
        "--depths",
        "1,2",
        "--train-per-cell",
        "2",
        "--holdout-per-cell",
        "1",
        "--group-size",
        str(params["group_size"]),
        "--temperature",
        "1.0",
        "--max-tokens",
        str(max_tokens),
        "--kl-coefficient",
        str(params["kl_coefficient"]),
        "--format-credit",
        "0.0",
        "--lora-rank",
        str(params["lora_rank"]),
        "--lora-targets",
        str(params["lora_targets"]),
        "--lora-layers",
        str(params["lora_layers"]),
        "--learning-rate",
        str(params["learning_rate"]),
        "--max-steps",
        "1",
        "--eval-every",
        "1",
        "--checkpoint-every",
        "1",
        "--checkpoint-keep",
        "1",
        "--calibrate-samples",
        "1",
        "--calibrate-group",
        str(params["calibrate_group"]),
        "--calibrate-tokens",
        str(max_tokens),
        "--calibrate-minutes",
        "10.0",
        "--max-minutes",
        "45.0",
        "--memory-fraction",
        str(params["memory_fraction"]),
        "--seed",
        str(int(params["seed"]) + 311),
        "--calibrate",
        "--cot",
        "--read-only-answer-channel-preflight",
    ]


def _run_answer_channel_preflight(contract: Mapping[str, Any]) -> int:
    validate_contract(contract, verify_model=True)
    from tools import train_grpo

    argv = _answer_channel_preflight_argv(contract)
    previous = list(sys.argv)
    try:
        sys.argv = [argv[0], *argv[1:]]
        return int(train_grpo.main())
    finally:
        sys.argv = previous


def _launch_answer_channel_preflight(contract_path: Path) -> int:
    contract = _strict_json(contract_path)
    validate_contract(contract, verify_model=True)
    python = str(Path(sys.executable))
    if not Path(python).exists():
        _fail("python_launcher_missing")
    tool = str(Path(__file__).resolve(strict=True))
    supplied = contract_path.expanduser()
    if not supplied.is_absolute():
        supplied = REPO_ROOT / supplied
    contract_absolute = str(supplied.resolve(strict=True))
    root = _repo_path(
        str(contract["paths"]["artifact_root"]),
        role="artifact_root",
        must_exist=False,
    )
    run_dir = str(root / "detached-answer-channel-preflight")
    argv = [
        "launch",
        "--run-dir",
        run_dir,
        "--name",
        f"{contract['campaign_id']}-answer-channel-preflight",
        "--cwd",
        str(REPO_ROOT),
        "--timeout",
        "5400",
        "--resume-contract",
        "none",
        python,
        tool,
        "run-answer-channel-preflight",
        "--contract",
        contract_absolute,
    ]
    return run_detached_step.main(argv)


def _launch_initial_policy_probe(contract_path: Path) -> int:
    contract = _strict_json(contract_path)
    validate_contract(contract, verify_model=True)
    python = str(Path(sys.executable))
    if not Path(python).exists():
        _fail("python_launcher_missing")
    tool = str(Path(__file__).resolve(strict=True))
    supplied = contract_path.expanduser()
    if not supplied.is_absolute():
        supplied = REPO_ROOT / supplied
    contract_absolute = str(supplied.resolve(strict=True))
    root = _repo_path(
        str(contract["paths"]["artifact_root"]),
        role="artifact_root",
        must_exist=False,
    )
    argv = [
        "launch",
        "--run-dir",
        str(root / "detached-initial-policy-probe"),
        "--name",
        f"{contract['campaign_id']}-initial-policy-probe",
        "--cwd",
        str(REPO_ROOT),
        "--timeout",
        "7200",
        "--resume-contract",
        "none",
        python,
        tool,
        "run-initial-policy-probe",
        "--contract",
        contract_absolute,
    ]
    return run_detached_step.main(argv)


def _launch_training(
    contract_path: Path,
    *,
    resume: bool,
    expected_launch_bundle_sha256: str,
) -> int:
    contract = _strict_json(contract_path)
    validate_contract(contract, verify_model=True)
    if time.time() < int(contract["launch_not_before_unix"]):
        _fail("presentation_window_still_active")
    _sha256_value = str(expected_launch_bundle_sha256)
    if len(_sha256_value) != 64 or any(
        character not in "0123456789abcdef" for character in _sha256_value
    ):
        _fail("verified_launch_bundle_digest_invalid")
    python = str(Path(sys.executable))
    if not Path(python).exists():
        _fail("python_launcher_missing")
    tool = str(Path(__file__).resolve(strict=True))
    supplied = contract_path.expanduser()
    if not supplied.is_absolute():
        supplied = REPO_ROOT / supplied
    contract_absolute = str(supplied.resolve(strict=True))
    run_dir = str(
        _repo_path(
            str(contract["paths"]["detached_training"]),
            role="detached_training",
            must_exist=False,
        )
    )
    verifier = json.dumps(
        [python, tool, "verify-resume", "--contract", contract_absolute],
        separators=(",", ":"),
    )
    bundle_path = _repo_path(
        str(contract["paths"]["verified_launch_bundle"]),
        role="verified_launch_bundle",
    )
    bundle = _strict_json(bundle_path)
    observed_bundle_file_sha256 = _sha256(
        read_stable_bytes(bundle_path, max_bytes=512 * 1024 * 1024)
    )
    if observed_bundle_file_sha256 != _sha256_value:
        _fail("verified_launch_bundle_digest_invalid")
    from core.learning.verified_transition_production_factory import (
        detached_signer_broker_paths,
    )

    broker_policy: list[dict[str, Any]] = []
    max_steps = int(contract["training"]["parameters"]["max_steps"])
    max_attempts = int(contract["training"]["watchdog_policy"]["max_attempts"])
    for role, invocation_limit in (
        ("task_issuer", max_steps * 2 + max_attempts * 2 + 8),
        ("evidence_verifier", max_steps + max_attempts + 8),
    ):
        signer = bundle.get("signers", {}).get(role)
        if not isinstance(signer, Mapping):
            _fail("verified_launch_signer_binding_invalid")
        request_path, response_path = detached_signer_broker_paths(
            str(signer["identity"]),
            str(signer["release_manifest"]),
        )
        broker_policy.append(
            {
                "command": [
                    str(signer["executable"]),
                    *[str(value) for value in signer["arguments"]],
                    "--request-file",
                    str(request_path),
                ],
                "cwd": str(REPO_ROOT),
                "stdout_path": str(response_path),
                "timeout_s_max": float(signer["timeout_millis"]) / 1000.0,
                "max_invocations": invocation_limit,
            }
        )
    argv = [
        "launch",
        "--run-dir",
        run_dir,
        "--name",
        str(contract["campaign_id"]),
        "--cwd",
        str(REPO_ROOT),
        "--timeout",
        str(contract["training"]["resource_envelope"]["detached_timeout_s"]),
        "--resume-contract",
        "target_checkpoint",
        "--resume-verifier-json",
        verifier,
        "--broker-policy-json",
        json.dumps(broker_policy, separators=(",", ":")),
    ]
    if resume:
        argv.append("--resume")
    argv.extend(
        [
            python,
            tool,
            "run-training",
            "--contract",
            contract_absolute,
            "--expected-launch-bundle-sha256",
            _sha256_value,
        ]
    )
    return run_detached_step.main(argv)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--contract", default=DEFAULT_CONTRACT)
    prepare.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    prepare.add_argument(
        "--campaign-profile",
        choices=sorted(CAMPAIGN_PROFILES),
        default=FULL_TRAINING_PROFILE,
    )
    prepare.add_argument("--model", default=DEFAULT_MODEL)
    prepare.add_argument("--execution-spec", default=DEFAULT_SPEC)
    prepare.add_argument("--artifact-root", default=DEFAULT_ROOT)
    prepare.add_argument("--committed-at", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--contract", default=DEFAULT_CONTRACT)
    verify.add_argument("--skip-model", action="store_true")
    run = subparsers.add_parser("run-training")
    run.add_argument("--contract", default=DEFAULT_CONTRACT)
    run.add_argument("--expected-launch-bundle-sha256", required=True)
    probe = subparsers.add_parser("run-initial-policy-probe")
    probe.add_argument("--contract", default=DEFAULT_CONTRACT)
    probe_launch = subparsers.add_parser("launch-initial-policy-probe")
    probe_launch.add_argument("--contract", default=DEFAULT_CONTRACT)
    preflight = subparsers.add_parser("run-answer-channel-preflight")
    preflight.add_argument("--contract", default=DEFAULT_CONTRACT)
    preflight_launch = subparsers.add_parser("launch-answer-channel-preflight")
    preflight_launch.add_argument("--contract", default=DEFAULT_CONTRACT)
    launch = subparsers.add_parser("launch-training")
    launch.add_argument("--contract", default=DEFAULT_CONTRACT)
    launch.add_argument("--resume", action="store_true")
    launch.add_argument("--expected-launch-bundle-sha256", required=True)
    resume = subparsers.add_parser("verify-resume")
    resume.add_argument("--contract", default=DEFAULT_CONTRACT)
    canary = subparsers.add_parser("verify-update-canary")
    canary.add_argument("--contract", required=True)
    canary.add_argument("--output", required=True)
    pause = subparsers.add_parser("request-training-pause")
    pause.add_argument("--contract", required=True)
    continue_training = subparsers.add_parser("request-training-resume")
    continue_training.add_argument("--contract", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "prepare":
            contract = build_contract(
                campaign_id=args.campaign_id,
                campaign_profile=args.campaign_profile,
                model=args.model,
                execution_spec=args.execution_spec,
                artifact_root=args.artifact_root,
                committed_at=args.committed_at,
            )
            _write_once(Path(args.contract), contract)
            receipt = validate_contract(contract, verify_model=True)
        else:
            contract = _strict_json(Path(args.contract))
            if args.action == "run-training":
                return _run_training(
                    contract,
                    expected_launch_bundle_sha256=(args.expected_launch_bundle_sha256),
                )
            if args.action == "run-initial-policy-probe":
                return _run_initial_policy_probe(contract)
            if args.action == "launch-initial-policy-probe":
                return _launch_initial_policy_probe(Path(args.contract))
            if args.action == "run-answer-channel-preflight":
                return _run_answer_channel_preflight(contract)
            if args.action == "launch-answer-channel-preflight":
                return _launch_answer_channel_preflight(Path(args.contract))
            if args.action == "launch-training":
                return _launch_training(
                    Path(args.contract),
                    resume=args.resume,
                    expected_launch_bundle_sha256=(args.expected_launch_bundle_sha256),
                )
            if args.action == "verify-resume":
                verdict = build_resume_verdict(
                    contract,
                    environment=os.environ,
                    verify_model=True,
                )
                print(json.dumps(verdict, sort_keys=True))
                return 0
            if args.action == "verify-update-canary":
                verdict = build_update_canary_verdict(
                    contract,
                    verify_model=True,
                )
                _write_once(Path(args.output), verdict)
                print(json.dumps(verdict, indent=2, sort_keys=True))
                return 0
            if args.action == "request-training-pause":
                request = _request_training_pause(contract)
                print(json.dumps(request, indent=2, sort_keys=True))
                return 0
            if args.action == "request-training-resume":
                request = _request_training_resume(contract)
                print(json.dumps(request, indent=2, sort_keys=True))
                return 0
            receipt = validate_contract(
                contract,
                verify_model=not args.skip_model,
            )
    except (OSError, PreregistrationError, TypeError, ValueError) as exc:
        print(f"resident recurrent GRPO preregistration: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
