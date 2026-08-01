#!/usr/bin/env python3
"""Materialize a production recurrent-GRPO launch from external trust roots."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import secrets
import stat
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_trust import (
    EVIDENCE_VERIFIER,
    TASK_ISSUER,
    externally_custodied_roles,
    operationally_isolated_roles,
    validate_campaign_trust_policy,
)
from core.learning.grpo import GRPOConfig
from core.learning.recurrent_grpo import RecurrentGRPOConfig, RecurrentSamplingConfig
from core.learning.recurrent_training_prompt import (
    render_recurrent_training_prompt,
)
from core.learning.verified_recurrent_transition_repository import (
    CAMPAIGN_FINALIZER_ID,
    DURABLE_REPLAY_LOADER_ID,
    INDEPENDENT_SCORER_ID,
    PRODUCTION_EVIDENCE_PRODUCER_ID,
    TOKEN_CODEC_ID,
    finalize_verified_recurrent_transition_campaign,
    load_recurrent_replay_packages,
    produce_verified_recurrent_transition_group,
    recurrent_trace_token_decoder,
    recurrent_trace_token_encoder,
    score_verified_recurrent_training_task,
)
from core.learning.verified_token_trace import (
    build_resident_tokenizer_trace_adapter,
)
from core.learning.verified_training_task import build_verified_training_task
from core.learning.verified_transition_causal_campaign import (
    CausalCampaignScheduleEntry,
    VerifiedTransitionCausalCampaignLedger,
    build_causal_campaign_manifest,
)
from core.learning.verified_transition_episode import canonical_json_bytes
from core.learning.verified_transition_launch_bundle import (
    VERIFIED_TRANSITION_LAUNCH_BUNDLE_SCHEMA,
    validate_verified_transition_launch_archive,
)
from core.learning.verified_transition_measurement_chain import (
    recurrent_grpo_config_contract,
)
from core.learning.verified_transition_policy_probe import (
    INITIAL_RECURRENT_POLICY_PROBE_SCHEMA_V2,
    INITIAL_RECURRENT_POLICY_PROBE_SCHEMA_V3,
    build_initial_policy_state_custody,
    inspect_initial_adapter_snapshot,
    inspect_initial_optimizer_snapshot,
    validate_initial_policy_state_custody,
    validate_initial_recurrent_policy_probe,
)
from core.learning.verified_transition_policy_state_replay import (
    build_policy_state_replay_contract,
    validate_policy_state_replay_contract,
)
from core.learning.verified_transition_production_factory import (
    JIT_PROVIDER_CONFIG_SCHEMA,
    CommandRoleSignerBroker,
    sampling_config_contract_document,
)
from core.learning.verified_transition_provider import (
    TASK_COMMITMENT_SCHEMA,
    build_verified_transition_provider_contract,
    callable_source_sha256,
)
from core.learning.verified_transition_reward import TransitionRewardConfig
from core.runtime.atomic_writer import (
    atomic_write_bytes_if_absent,
    ensure_private_directory,
)
from core.runtime.file_read_gateway import read_stable_bytes
from tools import prepare_resident_recurrent_grpo_campaign as prereg
from tools.run_verified_recurrent_grpo_training import (
    verified_recurrent_runtime_components,
)

SIGNER_CONFIG_SCHEMA = "aura.verified_transition.external_signer_config.v1"
MATERIALIZATION_INTENT_SCHEMA = "aura.verified_transition.launch_materialization_intent.v2"
MATERIALIZATION_RECEIPT_SCHEMA = "aura.verified_transition.launch_materialization_receipt.v2"
_CUSTODY_PROBE_SCHEMAS = frozenset(
    {
        INITIAL_RECURRENT_POLICY_PROBE_SCHEMA_V2,
        INITIAL_RECURRENT_POLICY_PROBE_SCHEMA_V3,
    }
)
_SIGNER_CONFIG_KEYS = frozenset(
    {
        "schema",
        "identity",
        "executable",
        "executable_sha256",
        "release_manifest",
        "custody_evidence",
        "arguments",
        "timeout_millis",
        "inherited_environment_names",
    }
)
_MATERIALIZATION_INTENT_KEYS = frozenset(
    {
        "schema",
        "campaign_id",
        "preregistration_contract_sha256",
        "initial_policy_probe_sha256",
        "trust_policy_sha256",
        "trust_root_sha256",
        "task_issuer_signer_config_sha256",
        "evidence_verifier_signer_config_sha256",
        "created_at_unix_ns",
        "intent_sha256",
    }
)
_MATERIALIZATION_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "campaign_id",
        "preregistration_contract_sha256",
        "initial_policy_probe_sha256",
        "provider_contract_sha256",
        "trust_policy_sha256",
        "campaign_schedule_root_sha256",
        "materialization_intent_sha256",
        "task_count",
        "bundle_path",
        "bundle_sha256",
        "created_at_unix_ns",
        "reopened",
        "claim_boundary",
        "receipt_sha256",
    }
)


class LaunchMaterializationError(RuntimeError):
    """One production launch could not be safely materialized."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise LaunchMaterializationError(code)


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _research_json_bytes(value: Any) -> bytes:
    """Canonical finite JSON for preregistration documents that contain floats."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise LaunchMaterializationError("research_document_not_canonical") from exc


def _stable_seed(base_seed: int, *parts: Any) -> int:
    payload = canonical_json_bytes([int(base_seed), *parts])
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{role}_invalid")
    return value


def _strict_owned_bytes(
    path: str | Path,
    *,
    role: str,
    max_bytes: int,
) -> bytes:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = prereg.REPO_ROOT / candidate
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current /= component
        if current.is_symlink():
            _fail(f"{role}_symlink_rejected")
    try:
        resolved = candidate.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise LaunchMaterializationError(f"{role}_unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        _fail(f"{role}_storage_invalid")
    try:
        return read_stable_bytes(resolved, max_bytes=max_bytes)
    except (OSError, ValueError) as exc:
        raise LaunchMaterializationError(f"{role}_unreadable") from exc


def _strict_json(
    path: str | Path,
    *,
    role: str,
    max_bytes: int = 64 << 20,
    require_canonical: bool = True,
) -> tuple[dict[str, Any], bytes, Path]:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = prereg.REPO_ROOT / candidate
    raw = _strict_owned_bytes(candidate, role=role, max_bytes=max_bytes)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in items:
            if key in document:
                _fail(f"{role}_duplicate_key")
            document[key] = value
        return document

    try:
        document = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _value: _fail(f"{role}_nonfinite"),
        )
    except LaunchMaterializationError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise LaunchMaterializationError(f"{role}_invalid") from exc
    canonical = _research_json_bytes(document) if isinstance(document, dict) else b""
    if not isinstance(document, dict) or (
        require_canonical and raw not in {canonical, canonical + b"\n"}
    ):
        _fail(f"{role}_schema_invalid")
    return document, raw, candidate.resolve(strict=True)


def _publish(path: Path, payload: bytes, *, role: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    if not atomic_write_bytes_if_absent(path, payload, mode=0o600):
        try:
            existing = read_stable_bytes(path, max_bytes=max(1, len(payload) + 1))
        except (OSError, ValueError) as exc:
            raise LaunchMaterializationError(f"{role}_publication_conflict") from exc
        if existing != payload:
            _fail(f"{role}_publication_conflict")


def _binding(path: Path) -> dict[str, Any]:
    payload = _strict_owned_bytes(path, role="launch_artifact", max_bytes=256 << 20)
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _load_bound_tokenizer(model_path: Path) -> Any:
    try:
        from mlx_lm.utils import load_tokenizer

        config_raw = _strict_owned_bytes(
            model_path / "config.json",
            role="model_config",
            max_bytes=4 << 20,
        )
        config = json.loads(config_raw)
        eos = config.get("eos_token_id") if isinstance(config, Mapping) else None
        eos_values = eos if isinstance(eos, list) else [eos]
        if not eos_values or any(
            type(token) is not int or not 0 <= token < 2**31 for token in eos_values
        ):
            _fail("model_tokenizer_eos_invalid")
        return load_tokenizer(model_path, eos_token_ids=eos)
    except LaunchMaterializationError:
        raise
    except (
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise LaunchMaterializationError("model_tokenizer_unavailable") from exc


def _load_signer_config(
    signer_config_path: str | Path,
) -> dict[str, Any]:
    document, _raw, _path = _strict_json(
        signer_config_path,
        role="signer_config",
        max_bytes=1 << 20,
    )
    if set(document) != _SIGNER_CONFIG_KEYS or document.get("schema") != SIGNER_CONFIG_SCHEMA:
        _fail("signer_config_schema_invalid")
    timeout = document.get("timeout_millis")
    arguments = document.get("arguments")
    environment = document.get("inherited_environment_names")
    if (
        type(timeout) is not int
        or not 100 <= timeout <= 300_000
        or not isinstance(arguments, list)
        or not isinstance(environment, list)
    ):
        _fail("signer_config_invalid")
    return document


def _build_signer(
    document: Mapping[str, Any],
) -> CommandRoleSignerBroker:
    return CommandRoleSignerBroker(
        identity=document.get("identity"),
        executable=document.get("executable"),
        executable_sha256=document.get("executable_sha256"),
        release_manifest=document.get("release_manifest"),
        custody_evidence=document.get("custody_evidence"),
        arguments=document.get("arguments"),
        timeout_seconds=int(document["timeout_millis"]) / 1000,
        inherited_environment_names=document.get("inherited_environment_names"),
    )


def _load_signer(
    signer_config_path: str | Path,
) -> tuple[dict[str, Any], CommandRoleSignerBroker]:
    document = _load_signer_config(signer_config_path)
    broker = _build_signer(document)
    return document, broker


def _validate_signer_role_separation(
    task_broker: CommandRoleSignerBroker,
    verifier_broker: CommandRoleSignerBroker,
) -> None:
    if (
        task_broker.identity == verifier_broker.identity
        or task_broker.custody_evidence_sha256 == verifier_broker.custody_evidence_sha256
    ):
        _fail("signer_role_separation_required")


def _validate_preregistration_envelope(
    contract: Mapping[str, Any],
) -> None:
    if (
        not isinstance(contract, Mapping)
        or not isinstance(contract.get("campaign_id"), str)
        or not isinstance(contract.get("paths"), Mapping)
        or not isinstance(contract["paths"].get("verified_launch_bundle"), str)
    ):
        _fail("preregistration_contract_schema_invalid")
    unsigned = dict(contract)
    claimed = unsigned.pop("contract_sha256", None)
    _sha256(claimed, role="preregistration_contract_sha256")
    canonical = _research_json_bytes(unsigned)
    if claimed not in {
        hashlib.sha256(canonical).hexdigest(),
        hashlib.sha256(canonical + b"\n").hexdigest(),
    }:
        _fail("preregistration_contract_digest_mismatch")


def _intervention_state_replay_required(
    contract: Mapping[str, Any],
) -> bool:
    training = contract.get("training")
    artifact = (
        training.get("verified_trajectory_config_artifact")
        if isinstance(training, Mapping)
        else None
    )
    config = artifact.get("config") if isinstance(artifact, Mapping) else None
    return isinstance(config, Mapping) and isinstance(config.get("intervention_config"), Mapping)


def _validate_probe_warm_start_binding(
    contract: Mapping[str, Any],
    probe: Mapping[str, Any],
) -> None:
    warm_start = contract.get("warm_start")
    probe_warm_start = probe.get("warm_start_receipt")
    if warm_start is None:
        if probe_warm_start is not None:
            _fail("initial_policy_probe_warm_start_unexpected")
        return
    if (
        not isinstance(warm_start, Mapping)
        or not isinstance(probe_warm_start, Mapping)
        or probe_warm_start.get("contract_sha256")
        != warm_start.get("contract_sha256")
        or probe_warm_start.get("policy_after_sha256")
        != probe.get("initial_policy_sha256")
        or probe_warm_start.get("claim_eligible") is not False
        or probe_warm_start.get("causal_preflight_required") is not True
    ):
        _fail("initial_policy_probe_warm_start_mismatch")


def _validate_materialized_initial_state(
    *,
    probe: Mapping[str, Any],
    provider_config: Mapping[str, Any],
) -> dict[str, Any] | None:
    custody_value = provider_config.get("initial_policy_state_custody")
    if probe.get("schema") not in _CUSTODY_PROBE_SCHEMAS:
        if custody_value is not None:
            _fail("initial_policy_state_custody_unexpected")
        return None
    custody = validate_initial_policy_state_custody(custody_value)
    artifact = inspect_initial_adapter_snapshot(
        custody["initial_adapter_path"],
        execution_spec_sha256=str(probe["execution_spec_sha256"]),
    )
    optimizer_artifact = inspect_initial_optimizer_snapshot(custody["initial_optimizer_path"])
    if (
        custody["initial_policy_probe_sha256"] != probe["receipt_sha256"]
        or custody["initial_policy_sha256"] != probe["initial_policy_sha256"]
        or custody["execution_spec_sha256"] != probe["execution_spec_sha256"]
        or custody["adapter_initialization"] != probe["adapter_initialization"]
        or custody["optimizer_initialization"] != probe["optimizer_initialization"]
        or custody["initial_adapter_artifact"] != artifact
        or custody["initial_optimizer_artifact"] != optimizer_artifact
        or probe["initial_adapter_artifact"] != artifact
        or probe["initial_optimizer_artifact"] != optimizer_artifact
    ):
        _fail("initial_policy_state_custody_mismatch")
    return custody


def _expected_policy_state_replay_contract(
    *,
    contract: Mapping[str, Any],
    probe: Mapping[str, Any],
    custody: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Reconstruct the complete external-replay input contract from custody."""

    required = _intervention_state_replay_required(contract)
    if not required:
        return None
    if custody is None:
        _fail("intervention_initial_policy_state_custody_required")
    training = contract.get("training")
    model = contract.get("model")
    spec_binding = contract.get("execution_spec")
    sources = contract.get("sources")
    if not all(isinstance(value, Mapping) for value in (training, model, spec_binding, sources)):
        _fail("policy_state_replay_preregistration_inputs_invalid")
    parameters = training.get("parameters")
    trajectory_artifact = training.get("verified_trajectory_config_artifact")
    envelope = training.get("resource_envelope")
    if not all(isinstance(value, Mapping) for value in (parameters, trajectory_artifact, envelope)):
        _fail("policy_state_replay_training_inputs_invalid")
    trajectory = trajectory_artifact.get("config")
    if not isinstance(trajectory, Mapping):
        _fail("policy_state_replay_trajectory_config_missing")
    spec_document, spec_raw, spec_path = _strict_json(
        str(spec_binding.get("path")),
        role="policy_state_replay_execution_spec",
        max_bytes=16 << 20,
        require_canonical=False,
    )
    if hashlib.sha256(spec_raw).hexdigest() != spec_binding.get("sha256") or len(
        spec_raw
    ) != spec_binding.get("size_bytes"):
        _fail("policy_state_replay_execution_spec_binding_mismatch")
    try:
        group_size = int(parameters["group_size"])
        kl_coefficient = float(parameters["kl_coefficient"])
        verifier_budget = int(envelope["detached_timeout_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LaunchMaterializationError("policy_state_replay_training_parameters_invalid") from exc
    grpo = GRPOConfig(
        group_size=group_size,
        kl_coefficient=kl_coefficient,
    )
    recurrent_config = recurrent_grpo_config_contract(
        RecurrentGRPOConfig(
            kl_coefficient=kl_coefficient,
            advantage_clip=grpo.advantage_clip,
        )
    )
    absolute_sources: dict[str, dict[str, Any]] = {}
    for role, binding in sorted(sources.items()):
        if not isinstance(binding, Mapping):
            _fail("policy_state_replay_source_binding_invalid")
        source_path = prereg._repo_path(str(binding.get("path")), role=f"source_{role}")
        absolute_sources[str(role)] = {
            "path": str(source_path),
            "sha256": binding.get("sha256"),
            "size_bytes": binding.get("size_bytes"),
        }
    expected = build_policy_state_replay_contract(
        preregistration_contract_sha256=str(contract["contract_sha256"]),
        initial_policy_sha256=str(probe["initial_policy_sha256"]),
        model_path=prereg._repo_path(str(model.get("path")), role="model"),
        base_checkpoint=model.get("base_checkpoint"),
        behavior_bundle=model.get("behavior_bundle"),
        execution_spec_path=spec_path,
        execution_spec_document=spec_document,
        source_bindings=absolute_sources,
        initial_policy_state_custody=custody,
        recurrent_grpo_config=recurrent_config,
        verified_trajectory_config=trajectory,
        external_verifier_max_seconds=verifier_budget,
    )
    if expected["execution_spec"]["semantic_sha256"] != spec_binding.get(
        "semantic_sha256"
    ) or expected["initial_policy_state_custody_sha256"] != custody.get("custody_sha256"):
        _fail("policy_state_replay_contract_cross_binding_mismatch")
    return expected


def _validate_materialized_policy_state_replay(
    *,
    contract: Mapping[str, Any],
    probe: Mapping[str, Any],
    provider_config: Mapping[str, Any],
    custody: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    expected = _expected_policy_state_replay_contract(
        contract=contract,
        probe=probe,
        custody=custody,
    )
    observed = provider_config.get("policy_state_replay_contract")
    if expected is None:
        if observed is not None:
            _fail("policy_state_replay_contract_unexpected")
        return None
    if validate_policy_state_replay_contract(observed, verify_files=True) != expected:
        _fail("policy_state_replay_contract_mismatch")
    return expected


def _recover_completed_launch(
    *,
    contract: Mapping[str, Any],
    probe: Mapping[str, Any],
    root_raw: bytes,
    policy_document: Mapping[str, Any],
    task_signer_document: Mapping[str, Any],
    verifier_signer_document: Mapping[str, Any],
    launch_root: Path,
    completion_path: Path,
) -> dict[str, Any]:
    receipt, _raw, _path = _strict_json(
        completion_path,
        role="materialization_receipt",
        max_bytes=1 << 20,
    )
    if (
        set(receipt) != _MATERIALIZATION_RECEIPT_KEYS
        or receipt.get("schema") != MATERIALIZATION_RECEIPT_SCHEMA
    ):
        _fail("materialization_receipt_schema_invalid")
    unsigned_receipt = dict(receipt)
    observed_receipt_sha256 = unsigned_receipt.pop("receipt_sha256", None)
    created_at = receipt.get("created_at_unix_ns")
    bundle_path = Path(str(receipt.get("bundle_path", "")))
    expected_bundle_path = launch_root / "launch-bundle.json"
    if (
        observed_receipt_sha256 != _digest(unsigned_receipt)
        or type(created_at) is not int
        or created_at <= 0
        or created_at % 1_000_000_000 != 0
        or not bundle_path.is_absolute()
        or bundle_path != expected_bundle_path
        or receipt.get("reopened") is not True
        or receipt.get("claim_boundary")
        not in {
            "launch_custody_only_no_training_or_reasoning_gain_claim",
            "host_isolated_research_launch_external_claim_custody_still_required",
        }
    ):
        _fail("materialization_receipt_invalid")

    intent, _raw, _path = _strict_json(
        launch_root / "materialization-intent.json",
        role="materialization_intent",
        max_bytes=1 << 20,
    )
    expected_intent_body = {
        "schema": MATERIALIZATION_INTENT_SCHEMA,
        "campaign_id": contract["campaign_id"],
        "preregistration_contract_sha256": contract["contract_sha256"],
        "initial_policy_probe_sha256": probe["receipt_sha256"],
        "trust_policy_sha256": _digest(policy_document),
        "trust_root_sha256": hashlib.sha256(root_raw).hexdigest(),
        "task_issuer_signer_config_sha256": _digest(task_signer_document),
        "evidence_verifier_signer_config_sha256": _digest(verifier_signer_document),
        "created_at_unix_ns": created_at,
    }
    expected_intent = {
        **expected_intent_body,
        "intent_sha256": _digest(expected_intent_body),
    }
    if (
        set(intent) != _MATERIALIZATION_INTENT_KEYS
        or intent != expected_intent
        or receipt.get("materialization_intent_sha256") != expected_intent["intent_sha256"]
    ):
        _fail("materialization_intent_mismatch")

    bundle_sha256 = _sha256(
        receipt.get("bundle_sha256"),
        role="materialization_bundle_sha256",
    )
    digest_raw = _strict_owned_bytes(
        launch_root / "launch-bundle.sha256",
        role="launch_bundle_digest",
        max_bytes=65,
    )
    if digest_raw != f"{bundle_sha256}\n".encode("ascii"):
        _fail("launch_bundle_digest_mismatch")
    archive = validate_verified_transition_launch_archive(
        bundle_path,
        expected_bundle_sha256=bundle_sha256,
        expected_preregistration_sha256=str(contract["contract_sha256"]),
        policy_validation_unix=created_at // 1_000_000_000,
    )
    if (
        archive.preregistration != dict(contract)
        or archive.trust_policy.document != dict(policy_document)
        or archive.trust_root != root_raw
    ):
        _fail("materialization_archive_input_mismatch")
    expected_claim_boundary = (
        "launch_custody_only_no_training_or_reasoning_gain_claim"
        if externally_custodied_roles(archive.trust_policy)
        else "host_isolated_research_launch_external_claim_custody_still_required"
    )
    if receipt.get("claim_boundary") != expected_claim_boundary:
        _fail("materialization_receipt_custody_boundary_mismatch")
    signer_inputs = {
        TASK_ISSUER: task_signer_document,
        EVIDENCE_VERIFIER: verifier_signer_document,
    }
    for role, signer_input in signer_inputs.items():
        archived_signer = archive.signer_documents[role]
        expected_fields = {key: signer_input[key] for key in _SIGNER_CONFIG_KEYS if key != "schema"}
        if any(archived_signer.get(key) != value for key, value in expected_fields.items()):
            _fail(f"{role}_archived_signer_config_mismatch")

    provider_contract = archive.provider_contract
    custody = _validate_materialized_initial_state(
        probe=probe,
        provider_config=archive.provider_config,
    )
    if _intervention_state_replay_required(contract) and custody is None:
        _fail("intervention_initial_policy_state_custody_required")
    _validate_materialized_policy_state_replay(
        contract=contract,
        probe=probe,
        provider_config=archive.provider_config,
        custody=custody,
    )
    if (
        receipt.get("campaign_id") != contract["campaign_id"]
        or receipt.get("preregistration_contract_sha256") != contract["contract_sha256"]
        or receipt.get("initial_policy_probe_sha256") != probe["receipt_sha256"]
        or receipt.get("provider_contract_sha256") != provider_contract["contract_sha256"]
        or receipt.get("trust_policy_sha256") != archive.trust_policy.policy_sha256
        or receipt.get("campaign_schedule_root_sha256")
        != provider_contract["campaign_schedule_root_sha256"]
        or receipt.get("task_count") != len(provider_contract["task_schedule"])
        or receipt.get("bundle_sha256") != archive.external_bundle_sha256
    ):
        _fail("materialization_receipt_reconstruction_mismatch")
    if (
        probe["campaign_id"] != contract["campaign_id"]
        or probe["initial_policy_sha256"] != provider_contract["initial_policy_sha256"]
        or probe["dataset_sha256"] != provider_contract["dataset_sha256"]
        or probe["tokenizer_bundle"] != provider_contract["tokenizer_bundle"]
    ):
        _fail("materialization_probe_archive_mismatch")
    return receipt


def _task_material(
    *,
    contract: Mapping[str, Any],
    tokenizer: Any,
    execution_spec_sha256: str,
    answer_nonces: Mapping[str, bytes] | None = None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, bytes],
]:
    params = contract["training"]["parameters"]
    train, _holdout, _source = prereg._build_task_split(
        task_source=str(params["task_source"]),
        domains=list(params["domains"]),
        depths=list(params["depths"]),
        train_per_cell=int(params["train_per_cell"]),
        holdout_per_cell=int(params["holdout_per_cell"]),
        seed=int(params["seed"]),
    )
    if len(train) != int(params["max_steps"]):
        _fail("training_schedule_not_one_pass")
    schedule: list[dict[str, Any]] = []
    commitments: dict[str, dict[str, Any]] = {}
    nonces: dict[str, bytes] = {}
    expected_task_ids = {task.task_id for task in train}
    if answer_nonces is not None and set(answer_nonces) != expected_task_ids:
        _fail("persisted_answer_nonce_scope_mismatch")
    for sequence, task in enumerate(train):
        nonce = (
            answer_nonces[task.task_id] if answer_nonces is not None else secrets.token_bytes(32)
        )
        public, _sealed = build_verified_training_task(task, answer_nonce=nonce)
        public_document = public.to_dict()
        _prompt, prompt_tokens = render_recurrent_training_prompt(
            tokenizer,
            task,
            include_chain_of_thought=bool(params["cot"]),
        )
        trainer_seed = _stable_seed(
            int(params["seed"]),
            "group",
            sequence + 1,
            task.task_id,
        )
        sample_seeds = [
            _stable_seed(
                int(params["seed"]),
                "verified-sample",
                sequence,
                index,
                task.task_id,
            )
            for index in range(int(params["group_size"]))
        ]
        if len(set(sample_seeds)) != len(sample_seeds):
            _fail("sample_seed_collision")
        commitment = {
            "schema": TASK_COMMITMENT_SCHEMA,
            "sequence": sequence,
            "task_id": task.task_id,
            "trainer_sample_seed": trainer_seed,
            "immutable_task_sha256": _digest(public_document),
            "prompt_tokens_sha256": _digest(list(prompt_tokens)),
            "recurrent_execution_spec_sha256": execution_spec_sha256,
            "sample_seeds": sample_seeds,
        }
        schedule.append(commitment)
        commitments[task.task_id] = public_document
        nonces[task.task_id] = nonce
    if len(commitments) != len(schedule):
        _fail("training_task_identity_collision")
    return schedule, commitments, nonces


def materialize_launch(
    *,
    preregistration_path: str | Path,
    initial_policy_probe_path: str | Path,
    trust_policy_path: str | Path,
    trust_root_path: str | Path,
    task_issuer_signer_config_path: str | Path,
    evidence_verifier_signer_config_path: str | Path,
    now_unix_ns: int | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Publish and independently reconstruct one immutable production launch."""

    contract, _prereg_raw, _prereg_path = _strict_json(
        preregistration_path,
        role="preregistration_contract",
    )
    _validate_preregistration_envelope(contract)
    probe_document, _probe_raw, _probe_path = _strict_json(
        initial_policy_probe_path,
        role="initial_policy_probe",
    )
    probe = validate_initial_recurrent_policy_probe(probe_document)
    if (
        probe["campaign_id"] != contract["campaign_id"]
        or probe["dataset_sha256"] != contract["training"]["dataset"]["sha256"]
        or probe["execution_spec_sha256"] != contract["execution_spec"]["semantic_sha256"]
        or probe["base_checkpoint"] != contract["model"]["base_checkpoint"]
        or probe["model_behavior_bundle"] != contract["model"]["behavior_bundle"]
    ):
        _fail("initial_policy_probe_contract_mismatch")
    _validate_probe_warm_start_binding(contract, probe)
    for role, binding in probe["source_bindings"].items():
        if contract["sources"].get(role) != binding:
            _fail("initial_policy_probe_source_mismatch")

    root_raw = _strict_owned_bytes(
        trust_root_path,
        role="trust_root",
        max_bytes=1 << 20,
    )
    policy_document, _policy_raw, _policy_path = _strict_json(
        trust_policy_path,
        role="trust_policy",
        max_bytes=16 << 20,
    )
    task_signer_document = _load_signer_config(task_issuer_signer_config_path)
    verifier_signer_document = _load_signer_config(evidence_verifier_signer_config_path)
    launch_root = prereg._repo_path(
        str(Path(contract["paths"]["verified_launch_bundle"]).parent),
        role="verified_launch_root",
        must_exist=False,
    )
    completion_path = launch_root / "materialization-receipt.json"
    if completion_path.exists():
        return _recover_completed_launch(
            contract=contract,
            probe=probe,
            root_raw=root_raw,
            policy_document=policy_document,
            task_signer_document=task_signer_document,
            verifier_signer_document=verifier_signer_document,
            launch_root=launch_root,
            completion_path=completion_path,
        )

    if (
        _intervention_state_replay_required(contract)
        and probe["schema"] not in _CUSTODY_PROBE_SCHEMAS
    ):
        _fail("intervention_initial_policy_state_custody_required")

    prereg.validate_contract(contract, verify_model=True)
    observed_second = (time.time_ns() if now_unix_ns is None else now_unix_ns) // 1_000_000_000
    policy = validate_campaign_trust_policy(
        policy_document,
        trusted_root_public_key_pem=root_raw,
        expected_campaign_name=str(contract["campaign_id"]),
        expected_protocol_sha256=str(contract["contract_sha256"]),
        now_unix=observed_second,
    )
    if not operationally_isolated_roles(policy):
        _fail("operational_role_custody_required")
    external_custody = externally_custodied_roles(policy)
    task_broker = _build_signer(task_signer_document)
    verifier_broker = _build_signer(verifier_signer_document)
    brokers = {
        TASK_ISSUER: task_broker,
        EVIDENCE_VERIFIER: verifier_broker,
    }
    _validate_signer_role_separation(task_broker, verifier_broker)
    for role, broker in brokers.items():
        pin = policy.role_pin(role)
        if (
            pin["implementation_sha256"] != broker.implementation_sha256
            or pin["release_sha256"] != broker.release_sha256
            or pin["custody_evidence_sha256"] != broker.custody_evidence_sha256
        ):
            _fail(f"{role}_signer_custody_mismatch")

    model_path = prereg._repo_path(str(contract["model"]["path"]), role="model")
    resolved_tokenizer = tokenizer or _load_bound_tokenizer(model_path)
    token_adapter = build_resident_tokenizer_trace_adapter(
        resolved_tokenizer,
        str(model_path),
    )
    if token_adapter.bundle_identity != probe["tokenizer_bundle"]:
        _fail("initial_policy_probe_tokenizer_mismatch")

    ensure_private_directory(launch_root)
    initial_policy_state_custody = None
    if probe["schema"] in _CUSTODY_PROBE_SCHEMAS:
        source_artifact = probe["initial_adapter_artifact"]
        source_optimizer_artifact = probe["initial_optimizer_artifact"]
        try:
            source_snapshot = (_probe_path.parent / source_artifact["path"]).resolve(strict=True)
            source_optimizer_snapshot = (
                _probe_path.parent / source_optimizer_artifact["path"]
            ).resolve(strict=True)
        except OSError as exc:
            raise LaunchMaterializationError(
                "initial_policy_probe_adapter_artifact_unavailable"
            ) from exc
        observed_source_artifact = inspect_initial_adapter_snapshot(
            source_snapshot,
            execution_spec_sha256=str(probe["execution_spec_sha256"]),
        )
        if observed_source_artifact != source_artifact:
            _fail("initial_policy_probe_adapter_artifact_mismatch")
        observed_source_optimizer_artifact = inspect_initial_optimizer_snapshot(
            source_optimizer_snapshot
        )
        if observed_source_optimizer_artifact != source_optimizer_artifact:
            _fail("initial_policy_probe_optimizer_artifact_mismatch")
        copied_snapshot = launch_root / source_artifact["path"]
        copied_optimizer_snapshot = launch_root / source_optimizer_artifact["path"]
        _publish(
            copied_snapshot,
            read_stable_bytes(
                source_snapshot,
                max_bytes=int(source_artifact["size_bytes"]),
            ),
            role="initial_adapter_snapshot",
        )
        _publish(
            copied_optimizer_snapshot,
            read_stable_bytes(
                source_optimizer_snapshot,
                max_bytes=int(source_optimizer_artifact["size_bytes"]),
            ),
            role="initial_optimizer_snapshot",
        )
        copied_artifact = inspect_initial_adapter_snapshot(
            copied_snapshot,
            execution_spec_sha256=str(probe["execution_spec_sha256"]),
        )
        if copied_artifact != source_artifact:
            _fail("materialized_initial_adapter_artifact_mismatch")
        copied_optimizer_artifact = inspect_initial_optimizer_snapshot(copied_optimizer_snapshot)
        if copied_optimizer_artifact != source_optimizer_artifact:
            _fail("materialized_initial_optimizer_artifact_mismatch")
        initial_policy_state_custody = build_initial_policy_state_custody(
            initial_policy_probe_sha256=str(probe["receipt_sha256"]),
            initial_policy_sha256=str(probe["initial_policy_sha256"]),
            execution_spec_sha256=str(probe["execution_spec_sha256"]),
            adapter_initialization=probe["adapter_initialization"],
            optimizer_initialization=probe["optimizer_initialization"],
            initial_adapter_artifact=copied_artifact,
            initial_optimizer_artifact=copied_optimizer_artifact,
            initial_adapter_path=copied_snapshot,
            initial_optimizer_path=copied_optimizer_snapshot,
        )
    requested_ns = time.time_ns() if now_unix_ns is None else now_unix_ns
    requested_ns = (requested_ns // 1_000_000_000) * 1_000_000_000
    intent_path = launch_root / "materialization-intent.json"
    intent_body = {
        "schema": MATERIALIZATION_INTENT_SCHEMA,
        "campaign_id": contract["campaign_id"],
        "preregistration_contract_sha256": contract["contract_sha256"],
        "initial_policy_probe_sha256": probe["receipt_sha256"],
        "trust_policy_sha256": policy.policy_sha256,
        "trust_root_sha256": hashlib.sha256(root_raw).hexdigest(),
        "task_issuer_signer_config_sha256": _digest(task_signer_document),
        "evidence_verifier_signer_config_sha256": _digest(verifier_signer_document),
        "created_at_unix_ns": requested_ns,
    }
    intended = {**intent_body, "intent_sha256": _digest(intent_body)}
    if intent_path.exists():
        existing_intent, _raw, _path = _strict_json(
            intent_path,
            role="materialization_intent",
            max_bytes=1 << 20,
        )
        existing_without_time = dict(existing_intent)
        existing_created_at = existing_without_time.get("created_at_unix_ns")
        if type(existing_created_at) is not int:
            _fail("materialization_intent_invalid")
        expected_with_existing_time = {
            **intent_body,
            "created_at_unix_ns": existing_created_at,
        }
        expected_intent = {
            **expected_with_existing_time,
            "intent_sha256": _digest(expected_with_existing_time),
        }
        if existing_intent != expected_intent:
            _fail("materialization_intent_mismatch")
        intended = existing_intent
        materialization_ns = existing_created_at
    else:
        _publish(
            intent_path,
            canonical_json_bytes(intended) + b"\n",
            role="materialization_intent",
        )
        materialization_ns = requested_ns

    nonce_path = launch_root / "task-answer-nonces.json"
    persisted_nonces: dict[str, bytes] | None = None
    if nonce_path.exists():
        nonce_document, _raw, _path = _strict_json(
            nonce_path,
            role="task_answer_nonces",
            max_bytes=256 << 20,
        )
        encoded = nonce_document.get("nonces_b64")
        if nonce_document.get(
            "schema"
        ) != "aura.verified_transition.task_answer_nonces.v1" or not isinstance(encoded, dict):
            _fail("task_answer_nonces_schema_invalid")
        try:
            persisted_nonces = {
                task_id: base64.b64decode(value, validate=True)
                for task_id, value in encoded.items()
            }
        except (TypeError, ValueError) as exc:
            raise LaunchMaterializationError("task_answer_nonces_invalid") from exc
    schedule, commitments, nonces = _task_material(
        contract=contract,
        tokenizer=resolved_tokenizer,
        execution_spec_sha256=str(contract["execution_spec"]["semantic_sha256"]),
        answer_nonces=persisted_nonces,
    )
    nonce_document = {
        "schema": "aura.verified_transition.task_answer_nonces.v1",
        "nonces_b64": {
            task_id: base64.b64encode(nonce).decode("ascii")
            for task_id, nonce in sorted(nonces.items())
        },
    }
    commitment_document = {
        "schema": "aura.verified_transition.task_commitments.v1",
        "tasks": commitments,
    }
    _publish(
        nonce_path,
        canonical_json_bytes(nonce_document) + b"\n",
        role="task_answer_nonces",
    )
    _publish(
        launch_root / "task-commitments.json",
        canonical_json_bytes(commitment_document) + b"\n",
        role="task_commitments",
    )
    custody_root = ensure_private_directory(launch_root / "custody")
    roots = {
        role: str(ensure_private_directory(custody_root / role))
        for role in (
            "campaign",
            "transition_artifacts",
            "updates",
            "replay_artifacts",
        )
    }
    output_root = prereg._repo_path(
        str(contract["paths"]["training_output"]),
        role="training_output",
        must_exist=False,
    )
    transaction_root = output_root / "verified-transition-transactions"
    sampling = RecurrentSamplingConfig(
        max_tokens=int(contract["training"]["parameters"]["max_tokens"])
    )
    policy_state_replay_contract = _expected_policy_state_replay_contract(
        contract=contract,
        probe=probe,
        custody=initial_policy_state_custody,
    )
    provider_config = {
        "evidence_timeout_ms": 300_000,
        "training_argv": list(contract["training"]["argv"]),
        "training_argv_sha256": _digest(list(contract["training"]["argv"])),
        "jit_plan": {
            "schema": JIT_PROVIDER_CONFIG_SCHEMA,
            "reward_config_sha256": _digest(TransitionRewardConfig().to_dict()),
            "sampling_config": sampling_config_contract_document(sampling),
            "branch_count": int(contract["training"]["parameters"]["group_size"]),
            "signer_broker_identity": task_broker.identity,
            "signer_broker_source_sha256": task_broker.source_sha256,
            "plan_store_root": str(
                (Path(roots["replay_artifacts"]) / "jit-plans").resolve(strict=False)
            ),
            "trainer_output_root": str(output_root.resolve(strict=False)),
            "transaction_root": str(transaction_root.resolve(strict=False)),
        },
        **(
            {"initial_policy_state_custody": (initial_policy_state_custody)}
            if initial_policy_state_custody is not None
            else {}
        ),
        **(
            {"policy_state_replay_contract": policy_state_replay_contract}
            if policy_state_replay_contract is not None
            else {}
        ),
    }
    validated_custody = _validate_materialized_initial_state(
        probe=probe,
        provider_config=provider_config,
    )
    _validate_materialized_policy_state_replay(
        contract=contract,
        probe=probe,
        provider_config=provider_config,
        custody=validated_custody,
    )
    components = verified_recurrent_runtime_components()
    provider_contract = build_verified_transition_provider_contract(
        provider_config=provider_config,
        evidence_producer_identity=PRODUCTION_EVIDENCE_PRODUCER_ID,
        evidence_producer_source_sha256=callable_source_sha256(
            produce_verified_recurrent_transition_group
        ),
        durable_artifact_loader_identity=DURABLE_REPLAY_LOADER_ID,
        durable_artifact_loader_source_sha256=callable_source_sha256(
            load_recurrent_replay_packages
        ),
        campaign_finalizer_identity=CAMPAIGN_FINALIZER_ID,
        campaign_finalizer_source_sha256=callable_source_sha256(
            finalize_verified_recurrent_transition_campaign
        ),
        trust_policy_sha256=policy.policy_sha256,
        trust_root_key_id=policy.root_key_id,
        campaign_id=str(contract["campaign_id"]),
        initial_policy_sha256=str(probe["initial_policy_sha256"]),
        scorer_identity=INDEPENDENT_SCORER_ID,
        scorer_source_sha256=callable_source_sha256(score_verified_recurrent_training_task),
        token_codec_identity=TOKEN_CODEC_ID,
        token_encoder_source_sha256=callable_source_sha256(recurrent_trace_token_encoder),
        token_decoder_source_sha256=callable_source_sha256(recurrent_trace_token_decoder),
        tokenizer_bundle=token_adapter.bundle_identity,
        dataset_sha256=str(contract["training"]["dataset"]["sha256"]),
        task_schedule=schedule,
        ledger_roots=roots,
        frozen_at_unix_ns=materialization_ns,
    )
    planned_ns = materialization_ns
    causal_manifest = build_causal_campaign_manifest(
        campaign_id=str(contract["campaign_id"]),
        provider_contract_sha256=str(provider_contract["contract_sha256"]),
        campaign_schedule_root_sha256=str(provider_contract["campaign_schedule_root_sha256"]),
        trust_policy_sha256=policy.policy_sha256,
        initial_policy_sha256=str(probe["initial_policy_sha256"]),
        schedule=[
            CausalCampaignScheduleEntry(
                sequence=row["sequence"],
                task_id=row["task_id"],
                task_commitment_sha256=_digest(row),
            )
            for row in schedule
        ],
        planned_at_unix_ns=planned_ns,
    )
    manifest_attestation = task_broker.attest(
        policy,
        role=TASK_ISSUER,
        payload=causal_manifest,
        signed_at_unix=planned_ns // 1_000_000_000,
        purpose=f"{contract['campaign_id']}:campaign-manifest",
    )
    campaign_open_path = Path(roots["campaign"]) / "campaign.open.json"
    if campaign_open_path.exists():
        campaign_ledger = VerifiedTransitionCausalCampaignLedger.open(
            roots["campaign"],
            policy=policy,
        )
        campaign_ledger.validate_open_manifest(
            expected_manifest=causal_manifest,
            policy=policy,
        )
    else:
        VerifiedTransitionCausalCampaignLedger.create(
            roots["campaign"],
            campaign_manifest=causal_manifest,
            campaign_manifest_attestation=manifest_attestation,
            policy=policy,
        )

    published: dict[str, Path] = {}
    preregistration_copy = launch_root / "preregistration-contract.json"
    _publish(
        preregistration_copy,
        _prereg_raw,
        role="preregistration-contract_json",
    )
    published["preregistration-contract.json"] = preregistration_copy
    artifact_documents = {
        "provider-contract.json": (provider_contract, True),
        "provider-config.json": (provider_config, True),
        "trust-policy.json": (policy.document, True),
        "task-commitments.json": (
            commitment_document,
            True,
        ),
        "task-answer-nonces.json": (
            nonce_document,
            True,
        ),
    }
    for name, (document, newline) in artifact_documents.items():
        path = launch_root / name
        payload = canonical_json_bytes(document) + (b"\n" if newline else b"")
        _publish(path, payload, role=name.replace(".", "_"))
        published[name] = path
    root_copy = launch_root / "trust-root.pem"
    _publish(root_copy, root_raw, role="trust_root")
    signer_bundles = {
        "task_issuer": {
            key: task_signer_document[key] for key in _SIGNER_CONFIG_KEYS if key != "schema"
        }
        | {
            "release_sha256": task_broker.release_sha256,
            "custody_evidence_sha256": (task_broker.custody_evidence_sha256),
        },
        "evidence_verifier": {
            key: verifier_signer_document[key] for key in _SIGNER_CONFIG_KEYS if key != "schema"
        }
        | {
            "release_sha256": verifier_broker.release_sha256,
            "custody_evidence_sha256": (verifier_broker.custody_evidence_sha256),
        },
    }
    unsigned_bundle = {
        "schema": VERIFIED_TRANSITION_LAUNCH_BUNDLE_SCHEMA,
        "campaign_name": str(contract["campaign_id"]),
        "preregistration_contract": _binding(published["preregistration-contract.json"]),
        "provider_contract": _binding(published["provider-contract.json"]),
        "provider_config": _binding(published["provider-config.json"]),
        "trust_policy": _binding(published["trust-policy.json"]),
        "trust_root": _binding(root_copy),
        "campaign_ledger_root": roots["campaign"],
        "signers": signer_bundles,
        "task_commitments": _binding(published["task-commitments.json"]),
        "task_answer_nonces": _binding(published["task-answer-nonces.json"]),
    }
    bundle = {**unsigned_bundle, "bundle_sha256": _digest(unsigned_bundle)}
    bundle_path = launch_root / "launch-bundle.json"
    bundle_raw = canonical_json_bytes(bundle) + b"\n"
    _publish(bundle_path, bundle_raw, role="launch_bundle")
    external_bundle_sha256 = hashlib.sha256(bundle_raw).hexdigest()
    digest_path = launch_root / "launch-bundle.sha256"
    _publish(
        digest_path,
        f"{external_bundle_sha256}\n".encode("ascii"),
        role="launch_bundle_digest",
    )

    from core.learning.verified_transition_launch_bundle import (
        load_verified_transition_provider_factory,
    )

    reopened = load_verified_transition_provider_factory(
        bundle_path,
        expected_bundle_sha256=external_bundle_sha256,
        expected_preregistration_sha256=str(contract["contract_sha256"]),
        components=components,
        now_unix=observed_second,
    )
    if reopened.contract_sha256 != provider_contract[
        "contract_sha256"
    ] or reopened.training_argv != tuple(contract["training"]["argv"]):
        _fail("launch_bundle_reopen_mismatch")
    receipt_body = {
        "schema": MATERIALIZATION_RECEIPT_SCHEMA,
        "campaign_id": contract["campaign_id"],
        "preregistration_contract_sha256": contract["contract_sha256"],
        "initial_policy_probe_sha256": probe["receipt_sha256"],
        "provider_contract_sha256": provider_contract["contract_sha256"],
        "trust_policy_sha256": policy.policy_sha256,
        "campaign_schedule_root_sha256": provider_contract["campaign_schedule_root_sha256"],
        "materialization_intent_sha256": intended["intent_sha256"],
        "task_count": len(schedule),
        "bundle_path": str(bundle_path),
        "bundle_sha256": external_bundle_sha256,
        "created_at_unix_ns": planned_ns,
        "reopened": True,
        "claim_boundary": (
            "launch_custody_only_no_training_or_reasoning_gain_claim"
            if external_custody
            else "host_isolated_research_launch_external_claim_custody_still_required"
        ),
    }
    receipt = {**receipt_body, "receipt_sha256": _digest(receipt_body)}
    _publish(
        completion_path,
        canonical_json_bytes(receipt) + b"\n",
        role="materialization_receipt",
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--initial-policy-probe", required=True)
    parser.add_argument("--trust-policy", required=True)
    parser.add_argument("--trust-root", required=True)
    parser.add_argument("--task-issuer-signer-config", required=True)
    parser.add_argument("--evidence-verifier-signer-config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = materialize_launch(
            preregistration_path=args.preregistration,
            initial_policy_probe_path=args.initial_policy_probe,
            trust_policy_path=args.trust_policy,
            trust_root_path=args.trust_root,
            task_issuer_signer_config_path=(args.task_issuer_signer_config),
            evidence_verifier_signer_config_path=(args.evidence_verifier_signer_config),
        )
    except (
        LaunchMaterializationError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"verified recurrent launch materialization: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
