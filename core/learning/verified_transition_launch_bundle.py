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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never

from core.brain.llm.latent_cortex.campaign_trust import (
    VerifiedCampaignTrustPolicy,
    validate_campaign_trust_policy,
)
from core.learning.verified_transition_causal_campaign import (
    VerifiedTransitionCausalCampaignLedger,
)
from core.learning.verified_transition_episode import canonical_json_bytes
from core.learning.verified_transition_production_factory import (
    CommandRoleSignerBroker,
    ProductionVerifiedTransitionProviderFactory,
)
from core.learning.verified_transition_provider import (
    callable_source_sha256,
    validate_verified_transition_provider_contract,
)
from core.runtime.file_read_gateway import read_stable_bytes

VERIFIED_TRANSITION_LAUNCH_BUNDLE_SCHEMA = (
    "aura.verified_transition.production_launch_bundle.v1"
)
_BUNDLE_KEYS = frozenset(
    {
        "schema",
        "campaign_name",
        "provider_contract",
        "provider_config",
        "trust_policy",
        "trust_root",
        "campaign_ledger_root",
        "signer",
        "task_commitments",
        "task_answer_nonces",
        "bundle_sha256",
    }
)
_FILE_BINDING_KEYS = frozenset({"path", "sha256", "size_bytes"})
_SIGNER_KEYS = frozenset(
    {
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
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
    ):
        _fail(f"{role}_invalid")
    return value


def _canonical_document(raw: bytes, *, role: str) -> dict[str, Any]:
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
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
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
            raise VerifiedTransitionLaunchBundleError(
                f"{role}_unavailable"
            ) from exc
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
    if hashlib.sha256(payload).hexdigest() != _sha256(
        binding.get("sha256"), role=f"{role}_sha256"
    ):
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


def load_verified_transition_provider_factory(
    bundle_path: str | Path,
    *,
    expected_bundle_sha256: str,
    components: VerifiedTransitionRuntimeComponents,
    now_unix: int,
) -> ProductionVerifiedTransitionProviderFactory:
    """Validate every launch byte and construct the post-load factory."""

    raw = _owned_regular_file(bundle_path, role="launch_bundle", max_bytes=64 << 20)
    if hashlib.sha256(raw).hexdigest() != _sha256(
        expected_bundle_sha256, role="expected_launch_bundle_sha256"
    ):
        _fail("launch_bundle_external_digest_mismatch")
    bundle = _canonical_document(raw, role="launch_bundle")
    if set(bundle) != _BUNDLE_KEYS or bundle.get("schema") != VERIFIED_TRANSITION_LAUNCH_BUNDLE_SCHEMA:
        _fail("launch_bundle_schema_invalid")
    unsigned = dict(bundle)
    claimed_bundle_sha = unsigned.pop("bundle_sha256")
    if claimed_bundle_sha != hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest():
        _fail("launch_bundle_internal_digest_mismatch")

    contract = validate_verified_transition_provider_contract(
        _canonical_document(
            _bound_bytes(bundle["provider_contract"], role="provider_contract", max_bytes=64 << 20),
            role="provider_contract",
        )
    )
    config = _canonical_document(
        _bound_bytes(bundle["provider_config"], role="provider_config", max_bytes=16 << 20),
        role="provider_config",
    )
    policy_document = _canonical_document(
        _bound_bytes(bundle["trust_policy"], role="trust_policy", max_bytes=16 << 20),
        role="trust_policy",
    )
    trust_root = _bound_bytes(bundle["trust_root"], role="trust_root", max_bytes=1 << 20)
    policy: VerifiedCampaignTrustPolicy = validate_campaign_trust_policy(
        policy_document,
        trusted_root_public_key_pem=trust_root,
        expected_campaign_name=_identifier(
            bundle.get("campaign_name"), role="campaign_name"
        ),
        expected_policy_sha256=contract["trust_policy_sha256"],
        now_unix=now_unix,
    )
    _validate_component_identities(contract, components)

    signer = bundle.get("signer")
    if not isinstance(signer, Mapping) or set(signer) != _SIGNER_KEYS:
        _fail("launch_signer_schema_invalid")
    timeout = signer.get("timeout_millis")
    if type(timeout) is not int or not 100 <= timeout <= 300_000:
        _fail("launch_signer_timeout_invalid")
    arguments = signer.get("arguments")
    environment = signer.get("inherited_environment_names")
    if not isinstance(arguments, list) or not isinstance(environment, list):
        _fail("launch_signer_arguments_invalid")
    broker = CommandRoleSignerBroker(
        identity=_identifier(signer.get("identity"), role="launch_signer_identity"),
        executable=signer.get("executable"),
        executable_sha256=_sha256(
            signer.get("executable_sha256"), role="launch_signer_executable_sha256"
        ),
        release_manifest=signer.get("release_manifest"),
        custody_evidence=signer.get("custody_evidence"),
        arguments=arguments,
        timeout_seconds=timeout / 1000,
        inherited_environment_names=environment,
    )

    commitments_document = _canonical_document(
        _bound_bytes(bundle["task_commitments"], role="task_commitments", max_bytes=256 << 20),
        role="task_commitments",
    )
    nonces_document = _canonical_document(
        _bound_bytes(bundle["task_answer_nonces"], role="task_answer_nonces", max_bytes=256 << 20),
        role="task_answer_nonces",
    )
    if commitments_document.get("schema") != "aura.verified_transition.task_commitments.v1":
        _fail("launch_task_commitments_schema_invalid")
    if nonces_document.get("schema") != "aura.verified_transition.task_answer_nonces.v1":
        _fail("launch_task_answer_nonces_schema_invalid")
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

    ledger_root = Path(
        _identifier(
            bundle.get("campaign_ledger_root"), role="campaign_ledger_root"
        )
    )
    if not ledger_root.is_absolute() or ledger_root.resolve(strict=False) != ledger_root:
        _fail("campaign_ledger_root_invalid")
    ledger = VerifiedTransitionCausalCampaignLedger.open(ledger_root, policy=policy)
    return ProductionVerifiedTransitionProviderFactory(
        contract=contract,
        provider_config=config,
        campaign_ledger=ledger,
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
        signer_broker=broker,
        task_commitments=commitments,
        task_answer_nonces=nonces,
    )


__all__ = [
    "VERIFIED_TRANSITION_LAUNCH_BUNDLE_SCHEMA",
    "VerifiedTransitionLaunchBundleError",
    "VerifiedTransitionRuntimeComponents",
    "load_verified_transition_provider_factory",
]
