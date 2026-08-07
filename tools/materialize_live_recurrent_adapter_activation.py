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
from pathlib import Path
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    CampaignPlan,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.campaign_trust import (  # noqa: E402
    EVIDENCE_VERIFIER,
    CampaignTrustError,
    assemble_role_attestation,
    externally_custodied_roles,
    prepare_role_signature_request,
    validate_campaign_trust_policy,
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


def _atomic_create_or_verify(path: Path, document: Mapping[str, Any]) -> Path:
    destination = path.expanduser().absolute()
    payload = canonical_json_bytes(document) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent_metadata = destination.parent.stat()
    if (
        not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != os.getuid()
        or stat.S_IMODE(parent_metadata.st_mode) & 0o077
    ):
        _fail("activation_materialization_output_parent_not_private")
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


def _signature(path: Path) -> str:
    payload = read_stable_bytes(path.expanduser().absolute(), max_bytes=MAX_SIGNATURE_BYTES)
    try:
        document = json.loads(payload, object_pairs_hook=_strict_object)
    except (UnicodeError, ValueError, json.JSONDecodeError):
        document = None
    if isinstance(document, dict) and set(document) == {"signature_b64"}:
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
    campaign_policy_path: Path,
    trusted_root_path: Path,
    adapter_package_path: Path,
    approved_adapter_roots: Sequence[Path],
    output_dir: Path,
    signed_at_unix: int,
    not_before_unix: int,
    expires_at_unix: int,
) -> dict[str, Any]:
    plan_document, plan_raw, plan_path = _read_json(campaign_plan_path, role="plan")
    verdict, verdict_raw, verdict_path = _read_json(
        independent_verdict_path,
        role="verdict",
    )
    policy_document, policy_raw, policy_path = _read_json(
        campaign_policy_path,
        role="policy",
    )
    root = read_stable_bytes(
        trusted_root_path.expanduser().absolute(),
        max_bytes=MAX_KEY_BYTES,
    )
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
    activation = build_live_adapter_activation(
        campaign_name=plan.campaign_name,
        policy_sha256=policy.policy_sha256,
        adapter_id=str(identity.get("adapter_id")),
        adapter_package_path=package,
        adapter_manifest_sha256=str(identity.get("manifest_sha256")),
        adapter_composite_identity_sha256=str(identity.get("composite_identity_sha256")),
        base_checkpoint_fingerprint=str(identity.get("base_checkpoint_fingerprint")),
        model_behavior_bundle_sha256=str(identity.get("model_behavior_bundle_sha256")),
        campaign_plan=_binding(plan_path, plan_raw),
        independent_verdict=_binding(verdict_path, verdict_raw),
        not_before_unix=not_before_unix,
        expires_at_unix=expires_at_unix,
    )
    validate_positive_live_adapter_evidence(
        activation,
        campaign_plan=plan_document,
        independent_verdict=verdict,
        independent_verdict_path=verdict_path,
    )
    request = prepare_role_signature_request(
        policy,
        role=EVIDENCE_VERIFIER,
        payload=activation,
        signed_at_unix=signed_at_unix,
        operation="activate_recurrent_adapter",
        purpose="verified-recurrent-adapter-activation",
    )
    destination = output_dir.expanduser().absolute()
    activation_path = _atomic_create_or_verify(destination / "activation.json", activation)
    request_path = _atomic_create_or_verify(
        destination / "activation-signature-request.json",
        request,
    )
    material = {
        "schema": PREPARATION_SCHEMA,
        "campaign_name": plan.campaign_name,
        "plan_sha256": plan.plan_sha256,
        "policy_sha256": policy.policy_sha256,
        "adapter_id": activation["adapter_id"],
        "adapter_package_path": str(package),
        "activation": _binding(activation_path, canonical_json_bytes(activation) + b"\n"),
        "signature_request": _binding(
            request_path,
            canonical_json_bytes(request) + b"\n",
        ),
        "campaign_policy": _binding(policy_path, policy_raw),
        "trusted_root_sha256": hashlib.sha256(root).hexdigest(),
        "publication_allowed": False,
        "required_next_step": "detached_evidence_verifier_signature_then_finalize",
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
    policy_document, policy_raw, policy_resolved = _read_json(
        campaign_policy_path,
        role="policy",
    )
    root = read_live_adapter_trust_root(trusted_root_path)
    if (
        _binding(policy_resolved, policy_raw) != preparation.get("campaign_policy")
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
        signature_b64=_signature(signature_path),
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
        campaign_policy=_binding(policy_resolved, policy_raw),
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_activation(
                campaign_plan_path=args.campaign_plan,
                independent_verdict_path=args.independent_verdict,
                campaign_policy_path=args.campaign_policy,
                trusted_root_path=args.trusted_root,
                adapter_package_path=args.adapter_package,
                approved_adapter_roots=args.approved_adapter_root,
                output_dir=args.output_dir,
                signed_at_unix=args.signed_at_unix,
                not_before_unix=args.not_before_unix,
                expires_at_unix=args.expires_at_unix,
            )
        else:
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
