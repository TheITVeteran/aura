"""Root-verified construction of the resident verified-transition provider.

The launch document contains paths and byte identities, never private signing
material.  A separately supplied root-key binding authenticates the campaign
policy; the policy in turn pins the external signer and every runtime role.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import os
import stat
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never

from core.brain.llm.latent_cortex.campaign_trust import (
    VerifiedCampaignTrustPolicy,
    validate_campaign_trust_policy,
)
from core.learning.durable_external_verifier_job import (
    DurableExternalVerifierJob,
)
from core.learning.verified_transition_causal_campaign import (
    VerifiedTransitionCausalCampaignLedger,
)
from core.learning.verified_transition_episode import canonical_json_bytes
from core.learning.verified_transition_policy_state_replay import (
    validate_policy_state_replay_contract,
)
from core.learning.verified_transition_production_factory import (
    CommandRoleSignerBroker,
    ProductionVerifiedTransitionProviderFactory,
)
from core.learning.verified_transition_provider import (
    callable_source_sha256,
    validate_verified_transition_provider_contract,
)
from core.runtime.atomic_writer import ensure_private_directory
from core.runtime.file_read_gateway import read_stable_bytes

VERIFIED_TRANSITION_LAUNCH_BUNDLE_SCHEMA = "aura.verified_transition.production_launch_bundle.v3"
_BUNDLE_KEYS = frozenset(
    {
        "schema",
        "campaign_name",
        "preregistration_contract",
        "provider_contract",
        "provider_config",
        "trust_policy",
        "trust_root",
        "campaign_ledger_root",
        "signers",
        "task_commitments",
        "task_answer_nonces",
        "bundle_sha256",
    }
)
_SIGNER_ROLES = frozenset({"task_issuer", "evidence_verifier"})
_FILE_BINDING_KEYS = frozenset({"path", "sha256", "size_bytes"})
_SIGNER_KEYS = frozenset(
    {
        "identity",
        "executable",
        "executable_sha256",
        "release_manifest",
        "release_sha256",
        "custody_evidence",
        "custody_evidence_sha256",
        "arguments",
        "timeout_millis",
        "inherited_environment_names",
    }
)


class VerifiedTransitionLaunchBundleError(RuntimeError):
    """Stable failure at the root-verified launch boundary."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise VerifiedTransitionLaunchBundleError(code)


def _sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{role}_invalid")
    return value


def _identifier(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 256:
        _fail(f"{role}_invalid")
    return value


def _canonical_document(
    raw: bytes,
    *,
    role: str,
    allow_bare_canonical: bool = False,
) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                _fail(f"{role}_duplicate_key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _value: _fail(f"{role}_nonfinite"),
        )
    except VerifiedTransitionLaunchBundleError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise VerifiedTransitionLaunchBundleError(f"{role}_invalid") from exc
    canonical = canonical_json_bytes(value)
    accepted = {canonical + b"\n"}
    if allow_bare_canonical:
        accepted.add(canonical)
    if not isinstance(value, dict) or raw not in accepted:
        _fail(f"{role}_noncanonical")
    return value


def _owned_regular_file(path: str | Path, *, role: str, max_bytes: int) -> bytes:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        _fail(f"{role}_path_invalid")
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current = current / component
        try:
            if current.is_symlink():
                _fail(f"{role}_path_symlink_rejected")
        except OSError as exc:
            raise VerifiedTransitionLaunchBundleError(f"{role}_unavailable") from exc
    try:
        resolved = candidate.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise VerifiedTransitionLaunchBundleError(f"{role}_unavailable") from exc
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
        raise VerifiedTransitionLaunchBundleError(f"{role}_unreadable") from exc


def _bound_bytes(binding: Any, *, role: str, max_bytes: int) -> bytes:
    if not isinstance(binding, Mapping) or set(binding) != _FILE_BINDING_KEYS:
        _fail(f"{role}_binding_invalid")
    payload = _owned_regular_file(binding.get("path"), role=role, max_bytes=max_bytes)
    size = binding.get("size_bytes")
    if type(size) is not int or size < 1 or size != len(payload):
        _fail(f"{role}_size_mismatch")
    if hashlib.sha256(payload).hexdigest() != _sha256(binding.get("sha256"), role=f"{role}_sha256"):
        _fail(f"{role}_digest_mismatch")
    return payload


@dataclass(frozen=True, slots=True)
class VerifiedTransitionRuntimeComponents:
    """Fixed production callables selected by code, not by launch JSON."""

    evidence_producer: Callable[..., Any]
    evidence_producer_identity: str
    durable_artifact_loader: Callable[..., Any]
    durable_artifact_loader_identity: str
    campaign_finalizer: Callable[..., Any]
    campaign_finalizer_identity: str
    independent_scorer: Callable[..., Any]
    scorer_identity: str
    token_encoder: Callable[..., Any]
    token_decoder: Callable[..., Any]
    token_codec_identity: str


@dataclass(frozen=True, slots=True)
class VerifiedTransitionLaunchArchive:
    """Signer-independent reconstruction of one completed launch bundle."""

    bundle: dict[str, Any]
    preregistration: dict[str, Any]
    provider_contract: dict[str, Any]
    provider_config: dict[str, Any]
    trust_policy: VerifiedCampaignTrustPolicy
    trust_root: bytes
    signer_documents: dict[str, dict[str, Any]]
    task_commitments: dict[str, dict[str, Any]]
    task_answer_nonces: dict[str, bytes]
    campaign_ledger: VerifiedTransitionCausalCampaignLedger
    external_bundle_sha256: str


def _validate_archival_signers(
    value: Any,
    *,
    policy: VerifiedCampaignTrustPolicy,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != _SIGNER_ROLES:
        _fail("launch_signers_schema_invalid")
    documents: dict[str, dict[str, Any]] = {}
    for role in sorted(_SIGNER_ROLES):
        signer = value.get(role)
        if not isinstance(signer, Mapping) or set(signer) != _SIGNER_KEYS:
            _fail(f"launch_{role}_signer_schema_invalid")
        document = dict(signer)
        timeout = document.get("timeout_millis")
        arguments = document.get("arguments")
        environment = document.get("inherited_environment_names")
        if (
            type(timeout) is not int
            or not 100 <= timeout <= 300_000
            or not isinstance(arguments, list)
            or any(
                not isinstance(argument, str) or "\x00" in argument or len(argument) > 4096
                for argument in arguments
            )
            or not isinstance(environment, list)
            or len(set(environment)) != len(environment)
            or any(
                not isinstance(name, str) or not name or not name.replace("_", "a").isalnum()
                for name in environment
            )
        ):
            _fail(f"launch_{role}_signer_arguments_invalid")
        _identifier(
            document.get("identity"),
            role=f"launch_{role}_signer_identity",
        )
        for field in ("executable", "release_manifest", "custody_evidence"):
            raw_path = document.get(field)
            if (
                not isinstance(raw_path, str)
                or not Path(raw_path).is_absolute()
                or Path(raw_path).expanduser() != Path(raw_path)
            ):
                _fail(f"launch_{role}_{field}_path_invalid")
        executable_sha256 = _sha256(
            document.get("executable_sha256"),
            role=f"launch_{role}_signer_executable_sha256",
        )
        release_sha256 = _sha256(
            document.get("release_sha256"),
            role=f"launch_{role}_signer_release_sha256",
        )
        custody_sha256 = _sha256(
            document.get("custody_evidence_sha256"),
            role=f"launch_{role}_signer_custody_sha256",
        )
        pin = policy.role_pin(role)
        if (
            executable_sha256 != pin["implementation_sha256"]
            or release_sha256 != pin["release_sha256"]
            or custody_sha256 != pin["custody_evidence_sha256"]
        ):
            _fail(f"launch_{role}_signer_policy_mismatch")
        documents[role] = document
    if (
        documents["task_issuer"]["identity"] == documents["evidence_verifier"]["identity"]
        or documents["task_issuer"]["custody_evidence_sha256"]
        == documents["evidence_verifier"]["custody_evidence_sha256"]
    ):
        _fail("launch_signer_role_separation_required")
    return documents


def validate_verified_transition_launch_archive(
    bundle_path: str | Path,
    *,
    expected_bundle_sha256: str,
    expected_preregistration_sha256: str,
    policy_validation_unix: int,
) -> VerifiedTransitionLaunchArchive:
    """Reconstruct immutable launch custody without executing signer commands."""

    raw = _owned_regular_file(bundle_path, role="launch_bundle", max_bytes=64 << 20)
    external_bundle_sha256 = hashlib.sha256(raw).hexdigest()
    if external_bundle_sha256 != _sha256(
        expected_bundle_sha256, role="expected_launch_bundle_sha256"
    ):
        _fail("launch_bundle_external_digest_mismatch")
    bundle = _canonical_document(raw, role="launch_bundle")
    if (
        set(bundle) != _BUNDLE_KEYS
        or bundle.get("schema") != VERIFIED_TRANSITION_LAUNCH_BUNDLE_SCHEMA
    ):
        _fail("launch_bundle_schema_invalid")
    unsigned = dict(bundle)
    claimed_bundle_sha = unsigned.pop("bundle_sha256")
    if claimed_bundle_sha != hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest():
        _fail("launch_bundle_internal_digest_mismatch")

    preregistration = _canonical_document(
        _bound_bytes(
            bundle["preregistration_contract"],
            role="preregistration_contract",
            max_bytes=64 << 20,
        ),
        role="preregistration_contract",
        allow_bare_canonical=True,
    )
    preregistration_unsigned = dict(preregistration)
    claimed_preregistration = preregistration_unsigned.pop("contract_sha256", None)
    if (
        claimed_preregistration
        != hashlib.sha256(canonical_json_bytes(preregistration_unsigned)).hexdigest()
    ):
        _fail("launch_preregistration_internal_digest_mismatch")
    if claimed_preregistration != _sha256(
        expected_preregistration_sha256,
        role="expected_preregistration_sha256",
    ):
        _fail("launch_preregistration_external_digest_mismatch")
    campaign_name = _identifier(bundle.get("campaign_name"), role="campaign_name")
    if (
        _identifier(
            preregistration.get("campaign_id"),
            role="preregistration_campaign_id",
        )
        != campaign_name
    ):
        _fail("launch_preregistration_campaign_mismatch")

    contract = validate_verified_transition_provider_contract(
        _canonical_document(
            _bound_bytes(
                bundle["provider_contract"],
                role="provider_contract",
                max_bytes=64 << 20,
            ),
            role="provider_contract",
        )
    )
    config = _canonical_document(
        _bound_bytes(
            bundle["provider_config"],
            role="provider_config",
            max_bytes=16 << 20,
        ),
        role="provider_config",
    )
    if contract["provider"]["config"] != config:
        _fail("launch_provider_config_mismatch")
    policy_document = _canonical_document(
        _bound_bytes(
            bundle["trust_policy"],
            role="trust_policy",
            max_bytes=16 << 20,
        ),
        role="trust_policy",
    )
    trust_root = _bound_bytes(bundle["trust_root"], role="trust_root", max_bytes=1 << 20)
    policy = validate_campaign_trust_policy(
        policy_document,
        trusted_root_public_key_pem=trust_root,
        expected_campaign_name=campaign_name,
        expected_protocol_sha256=claimed_preregistration,
        expected_policy_sha256=contract["trust_policy_sha256"],
        now_unix=policy_validation_unix,
    )
    if (
        contract["campaign_id"] != campaign_name
        or policy.document["campaign_name"] != campaign_name
        or contract["trust_root_key_id"] != policy.root_key_id
    ):
        _fail("launch_campaign_identity_mismatch")
    signer_documents = _validate_archival_signers(
        bundle.get("signers"),
        policy=policy,
    )

    commitments_document = _canonical_document(
        _bound_bytes(
            bundle["task_commitments"],
            role="task_commitments",
            max_bytes=256 << 20,
        ),
        role="task_commitments",
    )
    nonces_document = _canonical_document(
        _bound_bytes(
            bundle["task_answer_nonces"],
            role="task_answer_nonces",
            max_bytes=256 << 20,
        ),
        role="task_answer_nonces",
    )
    if (
        set(commitments_document) != {"schema", "tasks"}
        or commitments_document.get("schema") != "aura.verified_transition.task_commitments.v1"
        or set(nonces_document) != {"schema", "nonces_b64"}
        or nonces_document.get("schema") != "aura.verified_transition.task_answer_nonces.v1"
    ):
        _fail("launch_task_material_schema_invalid")
    commitments = commitments_document.get("tasks")
    encoded_nonces = nonces_document.get("nonces_b64")
    if not isinstance(commitments, dict) or not isinstance(encoded_nonces, dict):
        _fail("launch_task_material_invalid")
    try:
        nonces = {
            task_id: base64.b64decode(value, validate=True)
            for task_id, value in encoded_nonces.items()
        }
    except (TypeError, ValueError) as exc:
        raise VerifiedTransitionLaunchBundleError("launch_task_nonce_invalid") from exc
    schedule = contract["task_schedule"]
    task_ids = [row["task_id"] for row in schedule]
    if (
        len(set(task_ids)) != len(task_ids)
        or set(commitments) != set(task_ids)
        or set(nonces) != set(task_ids)
    ):
        _fail("launch_task_material_scope_mismatch")
    for row in schedule:
        task_id = row["task_id"]
        if (
            not isinstance(commitments[task_id], Mapping)
            or hashlib.sha256(canonical_json_bytes(commitments[task_id])).hexdigest()
            != row["immutable_task_sha256"]
            or not 32 <= len(nonces[task_id]) <= 256
        ):
            _fail("launch_task_material_commitment_mismatch")

    ledger_root = Path(_identifier(bundle.get("campaign_ledger_root"), role="campaign_ledger_root"))
    if not ledger_root.is_absolute() or ledger_root.resolve(strict=False) != ledger_root:
        _fail("campaign_ledger_root_invalid")
    ledger = VerifiedTransitionCausalCampaignLedger.open(ledger_root, policy=policy)
    ledger_manifest = ledger.campaign_manifest()
    ledger_identity = {
        "campaign_id": campaign_name,
        "provider_contract_sha256": contract["contract_sha256"],
        "campaign_schedule_root_sha256": contract["campaign_schedule_root_sha256"],
        "trust_policy_sha256": policy.policy_sha256,
        "initial_policy_sha256": contract["initial_policy_sha256"],
    }
    if any(ledger_manifest.get(field) != expected for field, expected in ledger_identity.items()):
        _fail("launch_causal_campaign_identity_mismatch")
    return VerifiedTransitionLaunchArchive(
        bundle=bundle,
        preregistration=preregistration,
        provider_contract=contract,
        provider_config=config,
        trust_policy=policy,
        trust_root=trust_root,
        signer_documents=signer_documents,
        task_commitments={task_id: dict(document) for task_id, document in commitments.items()},
        task_answer_nonces=nonces,
        campaign_ledger=ledger,
        external_bundle_sha256=external_bundle_sha256,
    )


def _validate_component_identities(
    contract: Mapping[str, Any], components: VerifiedTransitionRuntimeComponents
) -> None:
    provider = contract["provider"]
    scorer = contract["scorer"]
    codec = contract["token_codec"]
    checks = (
        (
            components.evidence_producer_identity,
            callable_source_sha256(components.evidence_producer),
            provider["evidence_producer_identity"],
            provider["evidence_producer_source_sha256"],
            "evidence_producer",
        ),
        (
            components.durable_artifact_loader_identity,
            callable_source_sha256(components.durable_artifact_loader),
            provider["durable_artifact_loader_identity"],
            provider["durable_artifact_loader_source_sha256"],
            "durable_artifact_loader",
        ),
        (
            components.campaign_finalizer_identity,
            callable_source_sha256(components.campaign_finalizer),
            provider["campaign_finalizer_identity"],
            provider["campaign_finalizer_source_sha256"],
            "campaign_finalizer",
        ),
        (
            components.scorer_identity,
            callable_source_sha256(components.independent_scorer),
            scorer["identity"],
            scorer["source_sha256"],
            "scorer",
        ),
        (
            components.token_codec_identity,
            callable_source_sha256(components.token_encoder),
            codec["identity"],
            codec["encoder_source_sha256"],
            "token_encoder",
        ),
        (
            components.token_codec_identity,
            callable_source_sha256(components.token_decoder),
            codec["identity"],
            codec["decoder_source_sha256"],
            "token_decoder",
        ),
    )
    for identity, source, expected_identity, expected_source, role in checks:
        callable_value = {
            "evidence_producer": components.evidence_producer,
            "durable_artifact_loader": components.durable_artifact_loader,
            "campaign_finalizer": components.campaign_finalizer,
            "scorer": components.independent_scorer,
            "token_encoder": components.token_encoder,
            "token_decoder": components.token_decoder,
        }[role]
        if (
            not inspect.isfunction(callable_value)
            or callable_value.__closure__ is not None
            or callable_value.__module__ in {None, "__main__"}
        ):
            _fail(f"launch_{role}_unstable_callable")
        if identity != expected_identity or source != expected_source:
            _fail(f"launch_{role}_identity_mismatch")


def _policy_state_replay_job(
    archive: VerifiedTransitionLaunchArchive,
) -> DurableExternalVerifierJob | None:
    value = archive.provider_config.get("policy_state_replay_contract")
    if value is None:
        return None
    contract = validate_policy_state_replay_contract(
        value,
        verify_files=True,
        verify_model=False,
    )
    sources = contract["source_bindings"]

    def source_path(role: str) -> Path:
        binding = sources.get(role)
        if not isinstance(binding, Mapping):
            _fail(f"launch_policy_state_replay_{role}_missing")
        path = Path(str(binding.get("path")))
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            _fail(f"launch_policy_state_replay_{role}_invalid")
        payload = read_stable_bytes(path, max_bytes=32 * 1024 * 1024)
        if len(payload) != binding.get("size_bytes") or hashlib.sha256(
            payload
        ).hexdigest() != binding.get("sha256"):
            _fail(f"launch_policy_state_replay_{role}_mismatch")
        return path.resolve(strict=True)

    worker = source_path("transition_policy_state_replay_worker")
    detached_runner = source_path("detached_supervisor")
    resume_helper = source_path("transition_policy_state_replay_resume")
    python = Path(sys.executable).resolve(strict=True)
    python_sha256 = hashlib.sha256(
        read_stable_bytes(python, max_bytes=512 * 1024 * 1024)
    ).hexdigest()
    replay_root = Path(archive.provider_contract["ledger_roots"]["replay_artifacts"])
    job_root = ensure_private_directory(replay_root / "external-policy-state-replay-jobs")
    return DurableExternalVerifierJob(
        job_root=job_root,
        executable=python,
        executable_sha256=python_sha256,
        cwd=worker.parents[1],
        detached_runner=detached_runner,
        resume_helper=resume_helper,
        arguments=(str(worker), "run"),
        timeout_seconds=contract["external_verifier_max_seconds"],
        result_max_bytes=256 * 1024 * 1024,
        request_max_bytes=512 * 1024 * 1024,
        require_sleep_protection=True,
    )


def load_verified_transition_provider_factory(
    bundle_path: str | Path,
    *,
    expected_bundle_sha256: str,
    expected_preregistration_sha256: str,
    components: VerifiedTransitionRuntimeComponents,
    now_unix: int,
) -> ProductionVerifiedTransitionProviderFactory:
    """Validate every launch byte and construct the post-load factory."""

    archive = validate_verified_transition_launch_archive(
        bundle_path,
        expected_bundle_sha256=expected_bundle_sha256,
        expected_preregistration_sha256=expected_preregistration_sha256,
        policy_validation_unix=now_unix,
    )
    contract = archive.provider_contract
    policy = archive.trust_policy
    _validate_component_identities(contract, components)
    policy_state_replay_job = _policy_state_replay_job(archive)
    brokers: dict[str, CommandRoleSignerBroker] = {}
    for role in sorted(_SIGNER_ROLES):
        signer = archive.signer_documents[role]
        timeout = signer["timeout_millis"]
        arguments = signer["arguments"]
        environment = signer["inherited_environment_names"]
        broker = CommandRoleSignerBroker(
            identity=_identifier(
                signer.get("identity"),
                role=f"launch_{role}_signer_identity",
            ),
            executable=signer.get("executable"),
            executable_sha256=_sha256(
                signer.get("executable_sha256"),
                role=f"launch_{role}_signer_executable_sha256",
            ),
            release_manifest=signer.get("release_manifest"),
            custody_evidence=signer.get("custody_evidence"),
            arguments=arguments,
            timeout_seconds=timeout / 1000,
            inherited_environment_names=environment,
            durable_policy_state_replay_job=(
                policy_state_replay_job if role == "evidence_verifier" else None
            ),
        )
        if (
            broker.implementation_sha256 != signer["executable_sha256"]
            or broker.release_sha256 != signer["release_sha256"]
            or broker.custody_evidence_sha256 != signer["custody_evidence_sha256"]
        ):
            _fail(f"launch_{role}_signer_artifact_mismatch")
        brokers[role] = broker
    return ProductionVerifiedTransitionProviderFactory(
        contract=contract,
        provider_config=archive.provider_config,
        campaign_ledger=archive.campaign_ledger,
        campaign_trust_policy=policy,
        evidence_producer=components.evidence_producer,
        evidence_producer_identity=components.evidence_producer_identity,
        durable_artifact_loader=components.durable_artifact_loader,
        durable_artifact_loader_identity=components.durable_artifact_loader_identity,
        campaign_finalizer=components.campaign_finalizer,
        campaign_finalizer_identity=components.campaign_finalizer_identity,
        independent_scorer=components.independent_scorer,
        scorer_identity=components.scorer_identity,
        token_encoder=components.token_encoder,
        token_decoder=components.token_decoder,
        token_codec_identity=components.token_codec_identity,
        task_issuer_signer_broker=brokers["task_issuer"],
        evidence_verifier_signer_broker=brokers["evidence_verifier"],
        task_commitments=archive.task_commitments,
        task_answer_nonces=archive.task_answer_nonces,
    )


__all__ = [
    "VERIFIED_TRANSITION_LAUNCH_BUNDLE_SCHEMA",
    "VerifiedTransitionLaunchArchive",
    "VerifiedTransitionLaunchBundleError",
    "VerifiedTransitionRuntimeComponents",
    "load_verified_transition_provider_factory",
    "validate_verified_transition_launch_archive",
]
