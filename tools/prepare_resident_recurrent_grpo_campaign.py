#!/usr/bin/env python3
"""Freeze, verify, and execute the resident recurrent-GRPO preregistration.

The contract is deliberately created before the long resident run. It binds
the model, recurrent graph, training-only task corpus, executable sources,
resource limits, causal factorial, powered confirmatory design, and claims
that remain unavailable until independent evidence accepts them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from core.runtime.atomic_writer import atomic_write_bytes_if_absent  # noqa: E402
from tools import run_detached_step  # noqa: E402
from tools.train_grpo import _build_task_split, _dataset_payload  # noqa: E402

CONTRACT_SCHEMA = "aura.resident_recurrent_grpo_preregistration.v1"
DEFAULT_CAMPAIGN_ID = "resident-32b-recurrent-grpo-cp259"
CAMPAIGN_ID = DEFAULT_CAMPAIGN_ID
DEFAULT_MODEL = (
    "training/fused-model/Aura-32B-crsm-closeout-jul1-20260701-215118"
)
DEFAULT_SPEC = (
    "config/latent_cortex/resident_32b_recurrent_grpo_execution_spec.json"
)
DEFAULT_CONTRACT = (
    "config/latent_cortex/resident_32b_recurrent_grpo_preregistration.json"
)
DEFAULT_ROOT = (
    "artifacts/closeout/latent_cortex/cp259_resident_32b_recurrent_grpo"
)
NOT_BEFORE = "2026-07-21T17:00:00-07:00"
TRAINING_SEED = 2026072102
CONFIRMATORY_OBSERVATIONS_PER_DOMAIN = 411
TRAINING_PARAMETERS: Mapping[str, Any] = {
    "task_source": "recurrence_curriculum",
    "domains": list(RECURRENCE_TRAINING_FAMILIES),
    "depths": [2, 4, 8],
    "train_per_cell": 8,
    "holdout_per_cell": 1,
    "group_size": 4,
    "temperature": 1.0,
    "max_tokens": 320,
    "kl_coefficient": 0.02,
    "format_credit": 0.0,
    "trajectory_credit": True,
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
    "calibrate_minutes": 20.0,
    "cot": True,
    "max_minutes": 1440.0,
    "memory_fraction": 0.42,
    "seed": TRAINING_SEED,
}
SOURCE_ROLES: Mapping[str, str] = {
    "campaign_contract": "tools/prepare_resident_recurrent_grpo_campaign.py",
    "trainer": "tools/train_grpo.py",
    "grpo": "core/learning/grpo.py",
    "adaptive_curriculum": "core/learning/adaptive_curriculum.py",
    "training_curriculum": "core/learning/recurrence_curriculum.py",
    "checkpoint": "core/learning/grpo_training_state.py",
    "artifact_schema": "core/learning/recurrent_grpo_artifact_schema.py",
    "recurrent_grpo": "core/learning/recurrent_grpo.py",
    "recurrent_objective": "core/learning/recurrence_native_objective_v2.py",
    "execution_spec": "core/brain/llm/latent_cortex/execution_spec.py",
    "latent_engine": "core/brain/llm/latent_cortex/engine.py",
    "recurrence": "core/brain/llm/latent_cortex/recurrence.py",
    "adapter": "core/brain/llm/latent_cortex/recurrence_adapter.py",
    "adapter_identity": (
        "core/brain/llm/latent_cortex/recurrent_grpo_adapter_identity.py"
    ),
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
    candidate = REPO_ROOT / pure
    try:
        resolved = candidate.resolve(strict=must_exist)
    except OSError as exc:
        raise PreregistrationError(f"{role}_path_invalid") from exc
    for root in _repository_roots(REPO_ROOT):
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


def _training_argv(
    *, campaign_id: str, model: str, output: str, execution_spec: str
) -> list[str]:
    params = TRAINING_PARAMETERS
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
    if params["cot"]:
        argv.append("--cot")
    return argv


def _dataset_commitment() -> dict[str, Any]:
    params = TRAINING_PARAMETERS
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
        "excluded_evaluation_families": list(
            CURRENT_EXCLUDED_TRAINING_FAMILIES
        ),
    }


def build_contract(
    *,
    campaign_id: str = DEFAULT_CAMPAIGN_ID,
    model: str = DEFAULT_MODEL,
    execution_spec: str = DEFAULT_SPEC,
    artifact_root: str = DEFAULT_ROOT,
    committed_at: str,
    model_identity: Mapping[str, Any] | None = None,
    behavior_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not campaign_id.startswith("resident-32b-recurrent-grpo-cp"):
        _fail("campaign_id_invalid")
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
        "detached_training": f"{artifact_root}/detached-training",
        "frozen_adapter": f"{artifact_root}/frozen-adapter",
        "directional_campaign": f"{artifact_root}/directional-campaign",
        "confirmatory_campaign": f"{artifact_root}/confirmatory-campaign",
        "external_comparison": f"{artifact_root}/external-frontier",
    }
    for role, value in paths.items():
        _repo_path(value, role=role, must_exist=False)
    resolved_model_identity = dict(
        model_identity or full_weight_checkpoint_identity(model_path)
    )
    resolved_behavior_identity = dict(
        behavior_identity or model_behavior_bundle_identity(model_path)
    )
    sources = {role: _binding(path) for role, path in SOURCE_ROLES.items()}
    training_argv = _training_argv(
        campaign_id=campaign_id,
        model=model,
        output=paths["training_output"],
        execution_spec=execution_spec,
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
            (
                "resident_full_stack > "
                "resident_full_stack_no_latent_opt"
            ),
            (
                "resident_full_stack > "
                "resident_full_stack_no_fast_weights"
            ),
            (
                "resident_full_stack > "
                "resident_full_stack_no_branch_exchange"
            ),
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
    confirmatory_tasks = (
        len(FRONTIER_DOMAINS) * CONFIRMATORY_OBSERVATIONS_PER_DOMAIN
    )
    material = {
        "schema": CONTRACT_SCHEMA,
        "campaign_id": campaign_id,
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
            "execution_mode": "recurrent",
            "parameters": dict(TRAINING_PARAMETERS),
            "argv": training_argv,
            "dataset": _dataset_commitment(),
            "resume_contract": "exact_identity_bound_checkpoint",
            "completion_required": {
                "schema": "aura.recurrent_grpo_training_completion.v1",
                "complete": True,
                "halt_reason": "max_steps",
                "causal_gain_proven": False,
            },
            "resource_envelope": {
                "host_memory_bytes": 68_719_476_736,
                "mlx_memory_fraction": 0.42,
                "exclusive_resident_model_owner": True,
                "detached_timeout_s": 93_600,
                "multi_hour_soak": False,
            },
        },
        "evaluation": {
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
        },
        "hypotheses": {
            "positive_interaction": (
                "(adapter_rlc-adapter_vanilla) > "
                "(base_rlc-base_vanilla)"
            ),
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
        "required_stage_order": [
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
        ],
    }
    return {**material, "contract_sha256": _document_sha(material)}


def validate_contract(
    contract: Mapping[str, Any], *, verify_model: bool = True
) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "campaign_id",
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
    if (
        not isinstance(campaign_id, str)
        or not campaign_id.startswith("resident-32b-recurrent-grpo-cp")
    ):
        _fail("campaign_identity_mismatch")
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
    )
    if (
        training.get("execution_mode") != "recurrent"
        or training.get("parameters") != TRAINING_PARAMETERS
        or training.get("argv") != expected_argv
        or training.get("dataset") != _dataset_commitment()
        or training.get("resume_contract") != "exact_identity_bound_checkpoint"
    ):
        _fail("training_contract_mismatch")
    dataset = training["dataset"]
    if (
        dataset.get("families")
        != list(CURRENT_EXCLUDED_TRAINING_FAMILIES)
        or dataset.get("excluded_evaluation_families")
        != list(CURRENT_EXCLUDED_TRAINING_FAMILIES)
        or dataset.get("train_holdout_id_overlap") != 0
        or dataset.get("train_holdout_prompt_overlap") != 0
    ):
        _fail("training_evaluation_separation_invalid")
    evaluation = contract.get("evaluation")
    if not isinstance(evaluation, Mapping):
        _fail("evaluation_contract_invalid")
    confirmatory = evaluation.get("powered_confirmatory")
    if (
        evaluation.get("registry_version") != CURRENT_REGISTRY_VERSION
        or evaluation.get("domains") != list(FRONTIER_DOMAINS)
        or not isinstance(confirmatory, Mapping)
        or confirmatory.get("observations_per_domain")
        != CONFIRMATORY_OBSERVATIONS_PER_DOMAIN
        or confirmatory.get("task_count")
        != len(FRONTIER_DOMAINS) * CONFIRMATORY_OBSERVATIONS_PER_DOMAIN
        or confirmatory.get("cell_count")
        != len(FRONTIER_DOMAINS)
        * CONFIRMATORY_OBSERVATIONS_PER_DOMAIN
        * 6
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
        "contract_sha256": claimed_sha,
        "model_verified": bool(verify_model),
        "training_tasks": dataset["train_tasks"],
        "holdout_tasks": dataset["holdout_tasks"],
        "confirmatory_tasks": confirmatory["task_count"],
        "confirmatory_cells": confirmatory["cell_count"],
        "claim_eligible": False,
    }


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
        or complete.get("execution_spec_sha256")
        != contract["execution_spec"]["semantic_sha256"]
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


def _run_training(contract: Mapping[str, Any]) -> int:
    validate_contract(contract, verify_model=True)
    not_before = int(contract["launch_not_before_unix"])
    if time.time() < not_before:
        _fail("presentation_window_still_active")
    from tools import train_grpo

    argv = list(contract["training"]["argv"])
    previous = list(sys.argv)
    try:
        sys.argv = [argv[0], *argv[1:]]
        result = train_grpo.main()
    finally:
        sys.argv = previous
    if result != 0:
        return int(result)
    training_root = _repo_path(
        str(contract["paths"]["training_output"]),
        role="training_output",
    )
    produced_dataset = (training_root / "dataset_manifest.json").read_bytes()
    if _sha256(produced_dataset) != contract["training"]["dataset"]["sha256"]:
        _fail("produced_dataset_commitment_mismatch")
    return 0


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


def _launch_training(contract_path: Path, *, resume: bool) -> int:
    contract = _strict_json(contract_path)
    validate_contract(contract, verify_model=True)
    if time.time() < int(contract["launch_not_before_unix"]):
        _fail("presentation_window_still_active")
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
        ]
    )
    return run_detached_step.main(argv)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--contract", default=DEFAULT_CONTRACT)
    prepare.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    prepare.add_argument("--model", default=DEFAULT_MODEL)
    prepare.add_argument("--execution-spec", default=DEFAULT_SPEC)
    prepare.add_argument("--artifact-root", default=DEFAULT_ROOT)
    prepare.add_argument("--committed-at", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--contract", default=DEFAULT_CONTRACT)
    verify.add_argument("--skip-model", action="store_true")
    run = subparsers.add_parser("run-training")
    run.add_argument("--contract", default=DEFAULT_CONTRACT)
    preflight = subparsers.add_parser("run-answer-channel-preflight")
    preflight.add_argument("--contract", default=DEFAULT_CONTRACT)
    preflight_launch = subparsers.add_parser("launch-answer-channel-preflight")
    preflight_launch.add_argument("--contract", default=DEFAULT_CONTRACT)
    launch = subparsers.add_parser("launch-training")
    launch.add_argument("--contract", default=DEFAULT_CONTRACT)
    launch.add_argument("--resume", action="store_true")
    resume = subparsers.add_parser("verify-resume")
    resume.add_argument("--contract", default=DEFAULT_CONTRACT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "prepare":
            contract = build_contract(
                campaign_id=args.campaign_id,
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
                return _run_training(contract)
            if args.action == "run-answer-channel-preflight":
                return _run_answer_channel_preflight(contract)
            if args.action == "launch-answer-channel-preflight":
                return _launch_answer_channel_preflight(Path(args.contract))
            if args.action == "launch-training":
                return _launch_training(Path(args.contract), resume=args.resume)
            if args.action == "verify-resume":
                verdict = build_resume_verdict(
                    contract,
                    environment=os.environ,
                    verify_model=True,
                )
                print(json.dumps(verdict, sort_keys=True))
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
