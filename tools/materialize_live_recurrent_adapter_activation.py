#!/usr/bin/env python3
"""Prepare and publish a certified live recurrent-adapter activation.

The workflow is deliberately split. ``prepare`` validates the positive,
claim-eligible evidence and emits exact bytes for an externally held evidence-
verifier key. ``finalize`` verifies that detached signature, reopens every
runtime/package identity, performs a complete admission dry run, and only then
publishes the immutable activation pointer. This tool never loads a private key
and never fuses role/depth-conditioned banks into static base weights.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import stat
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    CampaignPlan,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.campaign_launch_bundle import (  # noqa: E402
    ADAPTER_FREEZE_SCHEMA,
    CampaignLaunchBundleError,
    verify_adapter_freeze,
)
from core.brain.llm.latent_cortex.campaign_trust import (  # noqa: E402
    EVIDENCE_VERIFIER,
    CampaignTrustError,
    assemble_role_attestation,
    externally_custodied_roles,
    prepare_role_signature_request,
    validate_campaign_trust_policy,
    verify_role_attestation,
)
from core.brain.llm.latent_cortex.live_adapter_activation import (  # noqa: E402
    ROLE_CONDITIONED_MANIFEST_SCHEMA,
    LiveAdapterActivationError,
    admit_live_adapter_activation,
    build_live_adapter_activation,
    build_live_adapter_pointer,
    read_live_adapter_trust_root,
    validate_positive_live_adapter_evidence,
)
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (  # noqa: E402
    full_weight_checkpoint_identity,
    model_behavior_bundle_identity,
    personality_bundle_identity,
    runtime_environment_identity,
)
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402

PREPARATION_SCHEMA = "aura.latent_cortex.live_adapter_activation_preparation.v1"
EVIDENCE_BUNDLE_SCHEMA = "aura.latent_cortex.live_adapter_evidence_bundle.v1"
SIGNER_PACKET_SCHEMA = "aura.latent_cortex.live_adapter_signer_packet.v1"
COMMAND_SIGNER_REQUEST_SCHEMA = "aura.external_role_signer.request.v1"
COMMAND_SIGNER_RESPONSE_SCHEMA = "aura.external_role_signer.response.v1"
PUBLICATION_SCHEMA = "aura.latent_cortex.live_adapter_activation_publication.v1"
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_KEY_BYTES = 1024 * 1024
MAX_SIGNATURE_BYTES = 1024 * 1024


class ActivationMaterializationError(RuntimeError):
    """A production activation could not be prepared or published."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise ActivationMaterializationError(code)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("activation_materialization_duplicate_json_key")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Never:
    raise ActivationMaterializationError(
        f"activation_materialization_nonfinite_json:{value}"
    )


def _read_json(path: Path, *, role: str) -> tuple[dict[str, Any], bytes, Path]:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink():
        _fail(f"activation_materialization_{role}_symlink_rejected")
    try:
        resolved = lexical.resolve(strict=True)
        payload = read_stable_bytes(resolved, max_bytes=MAX_JSON_BYTES)
        document = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except ActivationMaterializationError:
        raise
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ActivationMaterializationError(
            f"activation_materialization_{role}_unavailable"
        ) from exc
    canonical = canonical_json_bytes(document)
    if not isinstance(document, dict) or payload not in {canonical, canonical + b"\n"}:
        _fail(f"activation_materialization_{role}_noncanonical")
    return document, payload, resolved


def _binding(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _portable_binding(root: Path, path: Path, payload: bytes) -> dict[str, Any]:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ActivationMaterializationError(
            "activation_materialization_portable_path_escape"
        ) from exc
    portable = PurePosixPath(relative)
    if portable.is_absolute() or not portable.parts or any(
        part in {"", ".", ".."} for part in portable.parts
    ):
        _fail("activation_materialization_portable_path_invalid")
    return {
        "path": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _same_content(left: Any, right: Any) -> bool:
    return (
        isinstance(left, Mapping)
        and isinstance(right, Mapping)
        and left.get("sha256") == right.get("sha256")
        and left.get("size_bytes") == right.get("size_bytes")
    )


def _ensure_private_directory(path: Path) -> Path:
    directory = path.expanduser().absolute()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = directory.stat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        _fail("activation_materialization_output_parent_not_private")
    return directory


def _atomic_create_or_verify_bytes(path: Path, payload: bytes) -> Path:
    destination = path.expanduser().absolute()
    _ensure_private_directory(destination.parent)
    if destination.is_symlink():
        _fail("activation_materialization_output_symlink_rejected")
    if destination.exists():
        if read_stable_bytes(destination, max_bytes=MAX_JSON_BYTES) == payload:
            return destination.resolve(strict=True)
        _fail("activation_materialization_output_collision")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail("activation_materialization_short_write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, destination, follow_symlinks=False)
    except FileExistsError:
        if read_stable_bytes(destination, max_bytes=MAX_JSON_BYTES) != payload:
            _fail("activation_materialization_output_collision")
    finally:
        temporary.unlink(missing_ok=True)
    directory = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return destination.resolve(strict=True)


def _atomic_create_or_verify(path: Path, document: Mapping[str, Any]) -> Path:
    return _atomic_create_or_verify_bytes(
        path,
        canonical_json_bytes(document) + b"\n",
    )


def _copy_evidence(
    destination: Path,
    *,
    filename: str,
    payload: bytes,
) -> tuple[Path, dict[str, Any]]:
    copied = _atomic_create_or_verify_bytes(destination / filename, payload)
    return copied, _binding(copied, payload)


def _portable_evidence_bundle(
    *,
    destination: Path,
    plan: CampaignPlan,
    plan_raw: bytes,
    verdict_raw: bytes,
    verifier_attestation_raw: bytes,
    policy_raw: bytes,
    trusted_root: bytes,
    adapter_manifest_raw: bytes,
    adapter_freeze: Mapping[str, Any],
    adapter_package_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    evidence_dir = destination / "evidence"
    copied: dict[str, dict[str, Any]] = {}
    portable_documents: dict[str, dict[str, Any]] = {}
    for role, filename, payload in (
        ("campaign_plan", "campaign-plan.json", plan_raw),
        ("independent_verdict", "independent-verdict.json", verdict_raw),
        (
            "independent_verifier_attestation",
            "independent-verifier-attestation.json",
            verifier_attestation_raw,
        ),
        ("campaign_policy", "campaign-policy.json", policy_raw),
        ("trusted_root", "trusted-root.pem", trusted_root),
        ("adapter_manifest", "recurrence-adapter-manifest.json", adapter_manifest_raw),
        (
            "adapter_freeze",
            "adapter-freeze.json",
            canonical_json_bytes(adapter_freeze) + b"\n",
        ),
    ):
        copied_path, copied[role] = _copy_evidence(
            evidence_dir,
            filename=filename,
            payload=payload,
        )
        portable_documents[role] = _portable_binding(
            evidence_dir,
            copied_path,
            payload,
        )
    material = {
        "schema": EVIDENCE_BUNDLE_SCHEMA,
        "campaign_name": plan.campaign_name,
        "plan_sha256": plan.plan_sha256,
        "adapter_id": adapter_freeze["adapter_id"],
        "adapter_package_path": str(adapter_package_path),
        "documents": portable_documents,
        "adapter_content_root_sha256": adapter_freeze["content_root_sha256"],
        "adapter_artifacts": adapter_freeze["artifacts"],
        "adapter_weights_embedded": False,
        "adapter_weights_binding": "adapter_freeze_artifact_inventory",
        "review_contract": {
            "portable_without_private_keys": True,
            "all_documentary_signing_inputs_embedded": True,
            "adapter_payload_identity_embedded": True,
            "adapter_payload_bytes_embedded": False,
            "large_adapter_payload_remains_in_approved_package": True,
            "publication_authority": False,
        },
    }
    bundle = {
        **material,
        "bundle_sha256": hashlib.sha256(canonical_json_bytes(material)).hexdigest(),
    }
    bundle_path = _atomic_create_or_verify(
        evidence_dir / "evidence-bundle.json",
        bundle,
    )
    copied["bundle"] = _binding(
        bundle_path,
        canonical_json_bytes(bundle) + b"\n",
    )
    return bundle, copied


def verify_portable_evidence_bundle(
    bundle_path: Path,
    *,
    activation: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Independently reopen a transferable evidence bundle."""

    bundle, bundle_raw, bundle_path = _read_json(
        bundle_path,
        role="evidence_bundle",
    )
    material = dict(bundle)
    claimed = material.pop("bundle_sha256", None)
    documents = bundle.get("documents")
    expected_review_contract = {
        "portable_without_private_keys": True,
        "all_documentary_signing_inputs_embedded": True,
        "adapter_payload_identity_embedded": True,
        "adapter_payload_bytes_embedded": False,
        "large_adapter_payload_remains_in_approved_package": True,
        "publication_authority": False,
    }
    if (
        bundle.get("schema") != EVIDENCE_BUNDLE_SCHEMA
        or claimed != hashlib.sha256(canonical_json_bytes(material)).hexdigest()
        or bundle.get("campaign_name") != activation.get("campaign_name")
        or bundle.get("adapter_id") != activation.get("adapter_id")
        or not isinstance(documents, Mapping)
        or bundle.get("review_contract") != expected_review_contract
        or bundle.get("adapter_weights_embedded") is not False
        or bundle.get("adapter_weights_binding")
        != "adapter_freeze_artifact_inventory"
        or set(documents)
        != {
            "campaign_plan",
            "independent_verdict",
            "campaign_policy",
            "trusted_root",
            "adapter_manifest",
            "adapter_freeze",
            "independent_verifier_attestation",
        }
    ):
        _fail("activation_materialization_evidence_bundle_invalid")
    reopened: dict[str, dict[str, Any]] = {}
    reopened_documents: dict[str, dict[str, Any]] = {}
    bundle_root = bundle_path.parent
    for role in (
        "campaign_plan",
        "independent_verdict",
        "independent_verifier_attestation",
        "campaign_policy",
        "adapter_manifest",
        "adapter_freeze",
    ):
        binding = documents.get(role)
        if not isinstance(binding, Mapping):
            _fail("activation_materialization_evidence_bundle_invalid")
        portable_path = PurePosixPath(str(binding.get("path")))
        if portable_path.is_absolute() or any(
            part in {"", ".", ".."} for part in portable_path.parts
        ):
            _fail("activation_materialization_evidence_bundle_path_invalid")
        candidate = bundle_root.joinpath(*portable_path.parts)
        document, payload, path = _read_json(
            candidate,
            role=f"evidence_bundle_{role}",
        )
        if _portable_binding(bundle_root, path, payload) != dict(binding):
            _fail("activation_materialization_evidence_bundle_binding_mismatch")
        reopened[role] = _binding(path, payload)
        reopened_documents[role] = document
    root_binding = documents.get("trusted_root")
    if not isinstance(root_binding, Mapping):
        _fail("activation_materialization_evidence_bundle_invalid")
    portable_root_path = PurePosixPath(str(root_binding.get("path")))
    if portable_root_path.is_absolute() or any(
        part in {"", ".", ".."} for part in portable_root_path.parts
    ):
        _fail("activation_materialization_evidence_bundle_path_invalid")
    root_path = bundle_root.joinpath(*portable_root_path.parts)
    root_raw = read_live_adapter_trust_root(root_path)
    if _portable_binding(bundle_root, root_path.resolve(strict=True), root_raw) != dict(
        root_binding
    ):
        _fail("activation_materialization_evidence_bundle_binding_mismatch")
    plan_document = reopened_documents["campaign_plan"]
    verdict_document = reopened_documents["independent_verdict"]
    verifier_attestation = reopened_documents["independent_verifier_attestation"]
    policy_document = reopened_documents["campaign_policy"]
    manifest_document = reopened_documents["adapter_manifest"]
    freeze_document = reopened_documents["adapter_freeze"]
    freeze_material = dict(freeze_document)
    freeze_sha256 = freeze_material.pop("certificate_sha256", None)
    if (
        not _same_content(reopened["campaign_plan"], activation.get("campaign_plan", {}))
        or not _same_content(
            reopened["independent_verdict"],
            activation.get("independent_verdict", {}),
        )
        or reopened["adapter_manifest"].get("sha256")
        != activation.get("adapter_manifest_sha256")
        or manifest_document.get("adapter_id") != activation.get("adapter_id")
        or freeze_document.get("schema") != ADAPTER_FREEZE_SCHEMA
        or freeze_sha256
        != hashlib.sha256(canonical_json_bytes(freeze_material)).hexdigest()
        or freeze_document.get("identity_receipt", {}).get("adapter_id")
        != activation.get("adapter_id")
        or freeze_document.get("content_root_sha256")
        != bundle.get("adapter_content_root_sha256")
        or freeze_document.get("artifacts") != bundle.get("adapter_artifacts")
        or bundle.get("adapter_package_path")
        != activation.get("adapter_package_path")
    ):
        _fail("activation_materialization_evidence_bundle_activation_mismatch")
    try:
        plan = CampaignPlan.from_dict(plan_document)
        verified_policy = validate_campaign_trust_policy(
            policy_document,
            trusted_root_public_key_pem=root_raw,
            expected_campaign_name=str(activation.get("campaign_name")),
            expected_policy_sha256=str(activation.get("policy_sha256")),
            now_unix=int(activation.get("not_before_unix")),
        )
    except (CampaignTrustError, TypeError, ValueError) as exc:
        raise ActivationMaterializationError(
            "activation_materialization_evidence_bundle_trust_invalid"
        ) from exc
    plan_metadata = plan.to_dict().get("metadata")
    adapter_identity = (
        plan_metadata.get("adapter_identity")
        if isinstance(plan_metadata, Mapping)
        else None
    )
    plan_identity = (
        adapter_identity.get("identity_receipt")
        if isinstance(adapter_identity, Mapping)
        else None
    )
    freeze_identity = freeze_document.get("identity_receipt")
    if (
        plan.plan_sha256 != bundle.get("plan_sha256")
        or not externally_custodied_roles(verified_policy)
        or not isinstance(plan_identity, Mapping)
        or freeze_identity != plan_identity
        or freeze_document.get("adapter_id") != activation.get("adapter_id")
        or plan_identity.get("manifest_sha256")
        != activation.get("adapter_manifest_sha256")
        or plan_identity.get("composite_identity_sha256")
        != activation.get("adapter_composite_identity_sha256")
        or plan_identity.get("base_checkpoint_fingerprint")
        != activation.get("base_checkpoint_fingerprint")
        or plan_identity.get("model_behavior_bundle_sha256")
        != activation.get("model_behavior_bundle_sha256")
    ):
        _fail("activation_materialization_evidence_bundle_plan_mismatch")
    verifier_request = verdict_document.get("verifier_attestation_request")
    verifier_payload = (
        verifier_request.get("payload")
        if isinstance(verifier_request, Mapping)
        else None
    )
    if (
        not isinstance(verifier_request, Mapping)
        or not isinstance(verifier_payload, Mapping)
        or verifier_request.get("role") != EVIDENCE_VERIFIER
        or verifier_request.get("signer_id")
        != verified_policy.role_pin(EVIDENCE_VERIFIER)["signer_id"]
        or verifier_request.get("policy_sha256") != verified_policy.policy_sha256
        or verifier_request.get("payload_sha256")
        != hashlib.sha256(canonical_json_bytes(verifier_payload)).hexdigest()
        or verdict_document.get("verifier_attestation_sha256")
        != hashlib.sha256(canonical_json_bytes(verifier_attestation)).hexdigest()
    ):
        _fail("activation_materialization_final_verifier_attestation_binding_invalid")
    try:
        verify_role_attestation(
            verified_policy,
            verifier_attestation,
            role=EVIDENCE_VERIFIER,
            expected_payload=verifier_payload,
        )
    except (CampaignTrustError, TypeError, ValueError) as exc:
        raise ActivationMaterializationError(
            "activation_materialization_final_verifier_attestation_invalid"
        ) from exc
    validate_positive_live_adapter_evidence(
        activation,
        campaign_plan=plan_document,
        independent_verdict=verdict_document,
        independent_verdict_path=Path(reopened["independent_verdict"]["path"]),
    )
    reopened["trusted_root"] = _binding(root_path.resolve(strict=True), root_raw)
    return bundle, reopened, reopened_documents


def _portable_signer_packet(
    *,
    destination: Path,
    plan: CampaignPlan,
    activation_path: Path,
    activation_raw: bytes,
    request_path: Path,
    request_raw: bytes,
    evidence_bundle_path: Path,
    evidence_bundle_raw: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifacts = {
        "activation": _portable_binding(destination, activation_path, activation_raw),
        "signature_request": _portable_binding(destination, request_path, request_raw),
        "evidence_bundle": _portable_binding(
            destination,
            evidence_bundle_path,
            evidence_bundle_raw,
        ),
    }
    material = {
        "schema": SIGNER_PACKET_SCHEMA,
        "campaign_name": plan.campaign_name,
        "plan_sha256": plan.plan_sha256,
        "artifacts": artifacts,
        "signer_contract": {
            "role": EVIDENCE_VERIFIER,
            "operation": "activate_recurrent_adapter",
            "purpose": "verified-recurrent-adapter-activation",
            "private_key_in_packet": False,
            "publication_authority": False,
            "copy_directory_as_unit": True,
        },
    }
    packet = {
        **material,
        "packet_sha256": hashlib.sha256(canonical_json_bytes(material)).hexdigest(),
    }
    packet_path = _atomic_create_or_verify(
        destination / "external-signer-packet.json",
        packet,
    )
    return packet, _binding(
        packet_path,
        canonical_json_bytes(packet) + b"\n",
    )


def verify_portable_signer_packet(packet_path: Path) -> dict[str, Any]:
    """Verify a copied signer packet without relying on source-host paths."""

    packet, packet_raw, packet_path = _read_json(
        packet_path,
        role="signer_packet",
    )
    material = dict(packet)
    claimed = material.pop("packet_sha256", None)
    artifacts = packet.get("artifacts")
    expected_signer_contract = {
        "role": EVIDENCE_VERIFIER,
        "operation": "activate_recurrent_adapter",
        "purpose": "verified-recurrent-adapter-activation",
        "private_key_in_packet": False,
        "publication_authority": False,
        "copy_directory_as_unit": True,
    }
    if (
        packet.get("schema") != SIGNER_PACKET_SCHEMA
        or claimed != hashlib.sha256(canonical_json_bytes(material)).hexdigest()
        or packet.get("signer_contract") != expected_signer_contract
        or not isinstance(artifacts, Mapping)
        or set(artifacts) != {"activation", "signature_request", "evidence_bundle"}
    ):
        _fail("activation_materialization_signer_packet_invalid")
    packet_root = packet_path.parent
    reopened_documents: dict[str, dict[str, Any]] = {}
    reopened_bindings: dict[str, dict[str, Any]] = {}
    for role in ("activation", "signature_request", "evidence_bundle"):
        binding = artifacts.get(role)
        if not isinstance(binding, Mapping):
            _fail("activation_materialization_signer_packet_invalid")
        portable_path = PurePosixPath(str(binding.get("path")))
        if portable_path.is_absolute() or any(
            part in {"", ".", ".."} for part in portable_path.parts
        ):
            _fail("activation_materialization_signer_packet_path_invalid")
        document, payload, path = _read_json(
            packet_root.joinpath(*portable_path.parts),
            role=f"signer_packet_{role}",
        )
        if _portable_binding(packet_root, path, payload) != dict(binding):
            _fail("activation_materialization_signer_packet_binding_mismatch")
        reopened_documents[role] = document
        reopened_bindings[role] = _binding(path, payload)
    activation = reopened_documents["activation"]
    signature_request = reopened_documents["signature_request"]
    signed_payload = signature_request.get("signed_payload")
    if (
        packet.get("campaign_name") != activation.get("campaign_name")
        or not isinstance(signed_payload, Mapping)
        or signed_payload.get("payload") != activation
        or signed_payload.get("role") != EVIDENCE_VERIFIER
        or signed_payload.get("operation") != "activate_recurrent_adapter"
        or signed_payload.get("purpose") != "verified-recurrent-adapter-activation"
    ):
        _fail("activation_materialization_signer_packet_payload_mismatch")
    bundle_path = Path(reopened_bindings["evidence_bundle"]["path"])
    bundle, bundled_bindings, _bundled_documents = verify_portable_evidence_bundle(
        bundle_path,
        activation=activation,
    )
    receipt_material = {
        "schema": "aura.latent_cortex.live_adapter_signer_packet_verification.v1",
        "packet_sha256": packet["packet_sha256"],
        "activation_sha256": activation.get("activation_sha256"),
        "signature_request_sha256": signature_request.get("request_sha256"),
        "evidence_bundle_sha256": bundle.get("bundle_sha256"),
        "campaign_name": activation.get("campaign_name"),
        "adapter_id": activation.get("adapter_id"),
        "plan_sha256": bundle.get("plan_sha256"),
        "passed": True,
        "publication_authority": False,
    }
    return {
        **receipt_material,
        "receipt_sha256": hashlib.sha256(
            canonical_json_bytes(receipt_material)
        ).hexdigest(),
        "_packet": packet,
        "_activation": activation,
        "_signature_request": signature_request,
        "_artifact_bindings": reopened_bindings,
        "_bundled_bindings": bundled_bindings,
    }


def _validate_portable_signer_packet(
    preparation: Mapping[str, Any],
    *,
    activation: Mapping[str, Any],
    signature_request: Mapping[str, Any],
) -> dict[str, Any]:
    packet_binding = preparation.get("portable_signer_packet")
    if not isinstance(packet_binding, Mapping):
        _fail("activation_materialization_signer_packet_missing")
    receipt = verify_portable_signer_packet(
        Path(str(packet_binding.get("path"))),
    )
    packet_path = Path(str(packet_binding.get("path"))).resolve(strict=True)
    packet_raw = read_stable_bytes(packet_path, max_bytes=MAX_JSON_BYTES)
    if (
        _binding(packet_path, packet_raw) != dict(packet_binding)
        or receipt["_activation"] != activation
        or receipt["_signature_request"] != signature_request
        or not _same_content(
            receipt["_artifact_bindings"]["evidence_bundle"],
            preparation.get("portable_evidence_bundle", {}),
        )
    ):
        _fail("activation_materialization_signer_packet_preparation_mismatch")
    return receipt


def _signature(path: Path, *, expected_request_sha256: str) -> str:
    payload = read_stable_bytes(path.expanduser().absolute(), max_bytes=MAX_SIGNATURE_BYTES)
    try:
        document = json.loads(payload, object_pairs_hook=_strict_object)
    except (UnicodeError, ValueError, json.JSONDecodeError):
        document = None
    if isinstance(document, dict) and set(document) == {"signature_b64"}:
        encoded = document["signature_b64"]
    elif (
        isinstance(document, dict)
        and set(document) == {"schema", "request_sha256", "signature_b64"}
        and document.get("schema") == COMMAND_SIGNER_RESPONSE_SCHEMA
        and document.get("request_sha256") == expected_request_sha256
    ):
        encoded = document["signature_b64"]
    else:
        try:
            encoded = payload.decode("ascii").strip()
        except UnicodeError:
            encoded = ""
        if len(payload) == 64:
            encoded = base64.b64encode(payload).decode("ascii")
    if not isinstance(encoded, str):
        _fail("activation_materialization_signature_invalid")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ActivationMaterializationError(
            "activation_materialization_signature_invalid"
        ) from exc
    if len(raw) != 64 or base64.b64encode(raw).decode("ascii") != encoded:
        _fail("activation_materialization_signature_invalid")
    return encoded


def _contained_package(path: Path, roots: Sequence[Path]) -> Path:
    package = path.expanduser().absolute()
    if package.is_symlink():
        _fail("activation_materialization_package_symlink_rejected")
    try:
        package = package.resolve(strict=True)
    except OSError as exc:
        raise ActivationMaterializationError(
            "activation_materialization_package_unavailable"
        ) from exc
    if not package.is_dir():
        _fail("activation_materialization_package_invalid")
    admitted = False
    for raw_root in roots:
        root = raw_root.expanduser().absolute()
        if root.is_symlink():
            _fail("activation_materialization_approved_root_symlink_rejected")
        root = root.resolve(strict=True)
        if package == root or package.is_relative_to(root):
            admitted = True
            break
    if not admitted:
        _fail("activation_materialization_package_outside_approved_roots")
    return package


def prepare_activation(
    *,
    campaign_plan_path: Path,
    independent_verdict_path: Path,
    independent_verifier_attestation_path: Path,
    campaign_policy_path: Path,
    trusted_root_path: Path,
    adapter_package_path: Path,
    approved_adapter_roots: Sequence[Path],
    output_dir: Path,
    signed_at_unix: int,
    not_before_unix: int,
    expires_at_unix: int,
) -> dict[str, Any]:
    destination = _ensure_private_directory(output_dir)
    plan_document, plan_raw, _plan_path = _read_json(campaign_plan_path, role="plan")
    verdict, verdict_raw, _verdict_path = _read_json(
        independent_verdict_path,
        role="verdict",
    )
    _verifier_attestation, verifier_attestation_raw, _verifier_attestation_path = (
        _read_json(
            independent_verifier_attestation_path,
            role="verifier_attestation",
        )
    )
    policy_document, policy_raw, _policy_path = _read_json(
        campaign_policy_path,
        role="policy",
    )
    root = read_live_adapter_trust_root(trusted_root_path)
    plan = CampaignPlan.from_dict(plan_document)
    policy = validate_campaign_trust_policy(
        policy_document,
        trusted_root_public_key_pem=root,
        expected_campaign_name=plan.campaign_name,
        now_unix=signed_at_unix,
    )
    if not externally_custodied_roles(policy):
        _fail("activation_materialization_external_custody_required")
    if not (
        policy.document["not_before_unix"] <= not_before_unix
        <= signed_at_unix
        < expires_at_unix
        <= policy.document["expires_at_unix"]
    ):
        _fail("activation_materialization_window_outside_policy")
    metadata = plan.to_dict().get("metadata")
    adapter = metadata.get("adapter_identity") if isinstance(metadata, Mapping) else None
    identity = adapter.get("identity_receipt") if isinstance(adapter, Mapping) else None
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("claim_eligible") is not True
        or not isinstance(adapter, Mapping)
        or adapter.get("format") != ROLE_CONDITIONED_MANIFEST_SCHEMA
        or not isinstance(identity, Mapping)
    ):
        _fail("activation_materialization_claim_eligible_adapter_required")

    package = _contained_package(adapter_package_path, approved_adapter_roots)
    manifest, manifest_raw, _manifest_path = _read_json(
        package / "recurrence_adapter_manifest.json",
        role="adapter_manifest",
    )
    if (
        manifest.get("schema") != ROLE_CONDITIONED_MANIFEST_SCHEMA
        or manifest.get("adapter_id") != identity.get("adapter_id")
        or hashlib.sha256(manifest_raw).hexdigest() != identity.get("manifest_sha256")
    ):
        _fail("activation_materialization_adapter_manifest_mismatch")
    try:
        adapter_freeze = verify_adapter_freeze(package)
    except (CampaignLaunchBundleError, OSError, ValueError) as exc:
        raise ActivationMaterializationError(
            "activation_materialization_adapter_freeze_invalid"
        ) from exc
    if (
        adapter_freeze.get("adapter_id") != identity.get("adapter_id")
        or adapter_freeze.get("identity_receipt") != identity
    ):
        _fail("activation_materialization_adapter_freeze_identity_mismatch")
    _bundle, evidence = _portable_evidence_bundle(
        destination=destination,
        plan=plan,
        plan_raw=plan_raw,
        verdict_raw=verdict_raw,
        verifier_attestation_raw=verifier_attestation_raw,
        policy_raw=policy_raw,
        trusted_root=root,
        adapter_manifest_raw=manifest_raw,
        adapter_freeze=adapter_freeze,
        adapter_package_path=package,
    )
    activation = build_live_adapter_activation(
        campaign_name=plan.campaign_name,
        policy_sha256=policy.policy_sha256,
        adapter_id=str(identity.get("adapter_id")),
        adapter_package_path=package,
        adapter_manifest_sha256=str(identity.get("manifest_sha256")),
        adapter_composite_identity_sha256=str(identity.get("composite_identity_sha256")),
        base_checkpoint_fingerprint=str(identity.get("base_checkpoint_fingerprint")),
        model_behavior_bundle_sha256=str(identity.get("model_behavior_bundle_sha256")),
        campaign_plan=evidence["campaign_plan"],
        independent_verdict=evidence["independent_verdict"],
        not_before_unix=not_before_unix,
        expires_at_unix=expires_at_unix,
    )
    validate_positive_live_adapter_evidence(
        activation,
        campaign_plan=plan_document,
        independent_verdict=verdict,
        independent_verdict_path=Path(evidence["independent_verdict"]["path"]),
    )
    request = prepare_role_signature_request(
        policy,
        role=EVIDENCE_VERIFIER,
        payload=activation,
        signed_at_unix=signed_at_unix,
        operation="activate_recurrent_adapter",
        purpose="verified-recurrent-adapter-activation",
    )
    activation_path = _atomic_create_or_verify(destination / "activation.json", activation)
    request_path = _atomic_create_or_verify(
        destination / "activation-signature-request.json",
        request,
    )
    activation_raw = canonical_json_bytes(activation) + b"\n"
    request_raw = canonical_json_bytes(request) + b"\n"
    evidence_bundle_path = Path(evidence["bundle"]["path"])
    evidence_bundle_raw = read_stable_bytes(
        evidence_bundle_path,
        max_bytes=MAX_JSON_BYTES,
    )
    _packet, signer_packet_binding = _portable_signer_packet(
        destination=destination,
        plan=plan,
        activation_path=activation_path,
        activation_raw=activation_raw,
        request_path=request_path,
        request_raw=request_raw,
        evidence_bundle_path=evidence_bundle_path,
        evidence_bundle_raw=evidence_bundle_raw,
    )
    signer_command = {
        "schema": COMMAND_SIGNER_REQUEST_SCHEMA,
        "purpose": "verified-recurrent-adapter-activation",
        "signature_request": request,
        "verification_packet_path": "external-signer-packet.json",
    }
    signer_command_path = _atomic_create_or_verify(
        destination / "external-signer-command.json",
        signer_command,
    )
    signer_command_binding = _binding(
        signer_command_path,
        canonical_json_bytes(signer_command) + b"\n",
    )
    material = {
        "schema": PREPARATION_SCHEMA,
        "campaign_name": plan.campaign_name,
        "plan_sha256": plan.plan_sha256,
        "policy_sha256": policy.policy_sha256,
        "adapter_id": activation["adapter_id"],
        "adapter_package_path": str(package),
        "activation": _binding(activation_path, activation_raw),
        "signature_request": _binding(
            request_path,
            request_raw,
        ),
        "campaign_policy": evidence["campaign_policy"],
        "trusted_root_sha256": hashlib.sha256(root).hexdigest(),
        "portable_evidence_bundle": evidence["bundle"],
        "portable_signer_packet": signer_packet_binding,
        "external_signer_command": signer_command_binding,
        "adapter_freeze": evidence["adapter_freeze"],
        "publication_allowed": False,
        "required_next_step": "external_evidence_verifier_command_then_finalize",
    }
    preparation = {
        **material,
        "preparation_sha256": hashlib.sha256(canonical_json_bytes(material)).hexdigest(),
    }
    _atomic_create_or_verify(destination / "preparation.json", preparation)
    return preparation


def finalize_activation(
    *,
    preparation_path: Path,
    signature_path: Path,
    campaign_policy_path: Path,
    trusted_root_path: Path,
    model_path: Path,
    personality_adapter_path: Path | None,
    approved_adapter_roots: Sequence[Path],
    pointer_path: Path,
    publication_receipt_path: Path,
    now_unix: int,
) -> dict[str, Any]:
    preparation, _preparation_raw, _resolved_preparation = _read_json(
        preparation_path,
        role="preparation",
    )
    material = dict(preparation)
    claimed = material.pop("preparation_sha256", None)
    if (
        preparation.get("schema") != PREPARATION_SCHEMA
        or claimed != hashlib.sha256(canonical_json_bytes(material)).hexdigest()
        or preparation.get("publication_allowed") is not False
    ):
        _fail("activation_materialization_preparation_invalid")
    activation_binding = preparation.get("activation")
    request_binding = preparation.get("signature_request")
    if not isinstance(activation_binding, Mapping) or not isinstance(
        request_binding, Mapping
    ):
        _fail("activation_materialization_preparation_binding_invalid")
    activation, activation_raw, activation_path = _read_json(
        Path(str(activation_binding.get("path"))),
        role="activation",
    )
    request, request_raw, request_path = _read_json(
        Path(str(request_binding.get("path"))),
        role="signature_request",
    )
    if (
        _binding(activation_path, activation_raw) != dict(activation_binding)
        or _binding(request_path, request_raw) != dict(request_binding)
    ):
        _fail("activation_materialization_preparation_binding_mismatch")
    signer_packet_receipt = _validate_portable_signer_packet(
        preparation,
        activation=activation,
        signature_request=request,
    )
    bundled_bindings = signer_packet_receipt["_bundled_bindings"]
    policy_binding = bundled_bindings["campaign_policy"]
    policy_document, policy_raw, policy_resolved = _read_json(
        Path(str(policy_binding["path"])),
        role="bundled_policy",
    )
    supplied_policy, supplied_policy_raw, _supplied_policy_path = _read_json(
        campaign_policy_path,
        role="supplied_policy",
    )
    root_binding = bundled_bindings["trusted_root"]
    bundled_root_path = Path(str(root_binding["path"]))
    root = read_live_adapter_trust_root(bundled_root_path)
    supplied_root = read_live_adapter_trust_root(trusted_root_path)
    if (
        _binding(policy_resolved, policy_raw) != policy_binding
        or supplied_policy != policy_document
        or hashlib.sha256(supplied_policy_raw).hexdigest() != policy_binding["sha256"]
        or len(supplied_policy_raw) != policy_binding["size_bytes"]
        or supplied_root != root
        or hashlib.sha256(root).hexdigest() != preparation.get("trusted_root_sha256")
    ):
        _fail("activation_materialization_trust_binding_mismatch")
    policy = validate_campaign_trust_policy(
        policy_document,
        trusted_root_public_key_pem=root,
        expected_campaign_name=str(preparation.get("campaign_name")),
        expected_policy_sha256=str(preparation.get("policy_sha256")),
        now_unix=now_unix,
    )
    attestation = assemble_role_attestation(
        policy,
        request,
        signature_b64=_signature(
            signature_path,
            expected_request_sha256=str(request.get("request_sha256")),
        ),
        role=EVIDENCE_VERIFIER,
    )
    attestation_path = _atomic_create_or_verify(
        publication_receipt_path.expanduser().absolute().parent
        / f"activation-attestation-{activation['activation_sha256'][:16]}.json",
        attestation,
    )
    attestation_raw = canonical_json_bytes(attestation) + b"\n"
    pointer = build_live_adapter_pointer(
        activation=activation,
        campaign_policy=policy_binding,
        activation_attestation=_binding(attestation_path, attestation_raw),
    )
    candidate_path = pointer_path.expanduser().absolute().with_name(
        f".{pointer_path.name}.{os.getpid()}.candidate"
    )
    try:
        _atomic_create_or_verify(candidate_path, pointer)
        admission = admit_live_adapter_activation(
            candidate_path,
            trusted_root_public_key_pem=root,
            approved_adapter_roots=approved_adapter_roots,
            actual_base_checkpoint=full_weight_checkpoint_identity(model_path),
            actual_model_behavior_bundle=model_behavior_bundle_identity(model_path),
            actual_personality_adapter=personality_bundle_identity(personality_adapter_path),
            actual_runtime_environment=runtime_environment_identity(),
            now_unix=now_unix,
        )
    finally:
        candidate_path.unlink(missing_ok=True)
    published_pointer = _atomic_create_or_verify(pointer_path, pointer)
    publication_material = {
        "schema": PUBLICATION_SCHEMA,
        "campaign_name": activation["campaign_name"],
        "adapter_id": activation["adapter_id"],
        "activation_sha256": activation["activation_sha256"],
        "pointer": _binding(
            published_pointer,
            canonical_json_bytes(pointer) + b"\n",
        ),
        "admission_receipt": admission,
        "published": True,
        "static_weight_fusion_performed": False,
        "activation_mode": "signed_default_on_role_and_depth_conditioned_adapter",
    }
    publication = {
        **publication_material,
        "publication_sha256": hashlib.sha256(
            canonical_json_bytes(publication_material)
        ).hexdigest(),
    }
    _atomic_create_or_verify(publication_receipt_path, publication)
    return publication


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--campaign-plan", type=Path, required=True)
    prepare.add_argument("--independent-verdict", type=Path, required=True)
    prepare.add_argument(
        "--independent-verifier-attestation",
        type=Path,
        required=True,
    )
    prepare.add_argument("--campaign-policy", type=Path, required=True)
    prepare.add_argument("--trusted-root", type=Path, required=True)
    prepare.add_argument("--adapter-package", type=Path, required=True)
    prepare.add_argument("--approved-adapter-root", type=Path, action="append", required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--signed-at-unix", type=int, required=True)
    prepare.add_argument("--not-before-unix", type=int, required=True)
    prepare.add_argument("--expires-at-unix", type=int, required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--preparation", type=Path, required=True)
    finalize.add_argument("--signature", type=Path, required=True)
    finalize.add_argument("--campaign-policy", type=Path, required=True)
    finalize.add_argument("--trusted-root", type=Path, required=True)
    finalize.add_argument("--model", type=Path, required=True)
    finalize.add_argument("--personality-adapter", type=Path)
    finalize.add_argument("--approved-adapter-root", type=Path, action="append", required=True)
    finalize.add_argument("--pointer-out", type=Path, required=True)
    finalize.add_argument("--publication-receipt-out", type=Path, required=True)
    finalize.add_argument("--now-unix", type=int, default=0)
    verify_packet = commands.add_parser("verify-packet")
    verify_packet.add_argument("--packet", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_activation(
                campaign_plan_path=args.campaign_plan,
                independent_verdict_path=args.independent_verdict,
                independent_verifier_attestation_path=(
                    args.independent_verifier_attestation
                ),
                campaign_policy_path=args.campaign_policy,
                trusted_root_path=args.trusted_root,
                adapter_package_path=args.adapter_package,
                approved_adapter_roots=args.approved_adapter_root,
                output_dir=args.output_dir,
                signed_at_unix=args.signed_at_unix,
                not_before_unix=args.not_before_unix,
                expires_at_unix=args.expires_at_unix,
            )
        elif args.command == "finalize":
            result = finalize_activation(
                preparation_path=args.preparation,
                signature_path=args.signature,
                campaign_policy_path=args.campaign_policy,
                trusted_root_path=args.trusted_root,
                model_path=args.model,
                personality_adapter_path=args.personality_adapter,
                approved_adapter_roots=args.approved_adapter_root,
                pointer_path=args.pointer_out,
                publication_receipt_path=args.publication_receipt_out,
                now_unix=args.now_unix or int(time.time()),
            )
        else:
            verified = verify_portable_signer_packet(args.packet)
            result = {
                key: value
                for key, value in verified.items()
                if not key.startswith("_")
            }
        sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
        return 0
    except (
        ActivationMaterializationError,
        CampaignTrustError,
        LiveAdapterActivationError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        error = {
            "schema": "aura.latent_cortex.live_adapter_activation_materialization_error.v1",
            "ok": False,
            "reason": getattr(exc, "code", str(exc)) or type(exc).__name__,
        }
        sys.stdout.buffer.write(canonical_json_bytes(error) + b"\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
