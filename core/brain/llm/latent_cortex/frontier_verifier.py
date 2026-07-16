"""Standalone verifier for raw Recursive Latent Cortex frontier evidence."""

from __future__ import annotations

import base64
import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.brain.llm.latent_cortex.frontier_artifacts import (
    FrontierArtifactError,
    read_strict_json_file,
    verify_raw_artifact_package,
)
from core.brain.llm.latent_cortex.frontier_certification import (
    CONJECTURE,
    INDEPENDENT_ATTESTATION_SCHEMA,
    PROVEN,
    canonical_sha256,
    evidence_payload_sha256,
    verify_frontier_gain_bundle,
)
from core.runtime.file_read_gateway import read_stable_bytes

TRUST_CONFIG_SCHEMA = "aura.latent_cortex.frontier_trust.v1"
VERIFICATION_CERTIFICATE_SCHEMA = "aura.latent_cortex.standalone_frontier_verification.v1"
ATTESTATION_REQUEST_SCHEMA = "aura.latent_cortex.independent_attestation_request.v1"
VERIFICATION_KERNEL_IDENTITY_SCHEMA = "aura.latent_cortex.verification_kernel_identity.v1"

_PIN_FIELDS = {"public_key_b64", "implementation_sha256", "release_sha256"}
_EXPECTED_PREATTESTATION_REASON = "independent_verifier_trust_pin_missing"
_SUPPORTED_COMPARISON = "resident_32b_vs_vanilla_same_checkpoint"
_MAX_IMPLEMENTATION_MODULE_BYTES = 16 * 1024 * 1024


class FrontierVerificationError(ValueError):
    """Stable input or precondition failure for the standalone verifier."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    error = FrontierVerificationError(code)
    raise error


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_pin(raw: Any, *, role: str) -> dict[str, str]:
    if not isinstance(raw, dict) or set(raw) != _PIN_FIELDS:
        _fail(f"{role}_trust_pin_invalid")
    public_key = raw.get("public_key_b64")
    if not isinstance(public_key, str) or not public_key or public_key != public_key.strip():
        _fail(f"{role}_trust_pin_invalid")
    try:
        decoded = base64.b64decode(public_key, validate=True)
    except (TypeError, ValueError):
        _fail(f"{role}_trust_pin_invalid")
    if len(decoded) != 32 or base64.b64encode(decoded).decode("ascii") != public_key:
        _fail(f"{role}_trust_pin_invalid")
    implementation = raw.get("implementation_sha256")
    release = raw.get("release_sha256")
    if not _is_sha256(implementation) or not _is_sha256(release):
        _fail(f"{role}_trust_pin_invalid")
    return {
        "public_key_b64": public_key,
        "implementation_sha256": implementation,
        "release_sha256": release,
    }


def validate_trust_config(raw: Any) -> dict[str, Any]:
    """Validate external pins and reject identity/key reuse across roles."""

    if not isinstance(raw, dict) or set(raw) != {
        "schema",
        "verification_kernel_sha256",
        "task_issuers",
        "verifiers",
    }:
        _fail("trust_config_schema_invalid")
    if raw.get("schema") != TRUST_CONFIG_SCHEMA:
        _fail("trust_config_schema_invalid")
    expected_kernel = raw.get("verification_kernel_sha256")
    if not _is_sha256(expected_kernel):
        _fail("verification_kernel_trust_pin_invalid")
    observed_kernel = verifier_implementation_sha256()
    if expected_kernel != observed_kernel:
        _fail("verification_kernel_trust_pin_mismatch")
    role_maps: dict[str, dict[str, dict[str, str]]] = {}
    for role in ("task_issuers", "verifiers"):
        entries = raw.get(role)
        if not isinstance(entries, dict) or not entries:
            _fail(f"{role}_trust_set_missing")
        validated: dict[str, dict[str, str]] = {}
        for signer_id, pin in entries.items():
            if (
                not isinstance(signer_id, str)
                or not signer_id
                or signer_id != signer_id.strip()
                or len(signer_id) > 200
            ):
                _fail(f"{role}_signer_id_invalid")
            validated[signer_id] = _validate_pin(pin, role=role)
        role_maps[role] = validated
    issuer_ids = set(role_maps["task_issuers"])
    verifier_ids = set(role_maps["verifiers"])
    if issuer_ids & verifier_ids:
        _fail("trust_role_identity_reused")
    issuer_keys = {pin["public_key_b64"] for pin in role_maps["task_issuers"].values()}
    verifier_keys = {pin["public_key_b64"] for pin in role_maps["verifiers"].values()}
    if issuer_keys & verifier_keys:
        _fail("trust_role_key_reused")
    return {
        **role_maps,
        "verification_kernel_sha256": observed_kernel,
    }


def verifier_implementation_sha256() -> str:
    """Fingerprint the complete local verification implementation."""

    from core.brain import frontier_evidence_v5
    from core.brain.llm.latent_cortex import (
        experiments,
        frontier_artifacts,
        frontier_certification,
    )
    from core.runtime import file_read_gateway

    repo_root = Path(__file__).resolve().parents[4]
    modules = (
        Path(__file__),
        Path(frontier_artifacts.__file__),
        Path(frontier_certification.__file__),
        Path(frontier_evidence_v5.__file__),
        Path(experiments.__file__),
        Path(file_read_gateway.__file__),
        repo_root / "tools" / "verify_latent_cortex_frontier.py",
    )
    rows = []
    for path in modules:
        resolved = path.resolve(strict=True)
        payload = read_stable_bytes(
            resolved,
            max_bytes=_MAX_IMPLEMENTATION_MODULE_BYTES,
        )
        rows.append(
            {
                "module": resolved.relative_to(repo_root).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    rows.sort(key=lambda row: row["module"])
    return canonical_sha256(rows)


def _load_inputs(
    *,
    bundle_path: Path,
    manifest_path: Path,
    trust_path: Path,
) -> tuple[
    dict[str, Any],
    bytes,
    dict[str, Any],
    bytes,
    dict[str, Any],
    bytes,
]:
    try:
        bundle, bundle_bytes = read_strict_json_file(bundle_path, role="frontier_bundle")
        manifest, manifest_bytes = read_strict_json_file(
            manifest_path,
            role="raw_artifact_manifest",
        )
        trust_raw, trust_bytes = read_strict_json_file(trust_path, role="trust_config")
    except FrontierArtifactError as exc:
        _fail(exc.code)
    if not isinstance(bundle, dict):
        _fail("frontier_bundle_not_mapping")
    trust = validate_trust_config(trust_raw)
    return bundle, bundle_bytes, manifest, manifest_bytes, trust, trust_bytes


def _finalize_certificate(certificate: dict[str, Any]) -> dict[str, Any]:
    certificate["certificate_sha256"] = canonical_sha256(certificate)
    return certificate


def _rejection_certificate(reason: str) -> dict[str, Any]:
    return _finalize_certificate(
        {
            "schema": VERIFICATION_CERTIFICATE_SCHEMA,
            "accepted": False,
            "release_accepted": False,
            "verification_status": "MALFORMED_OR_TAMPERED",
            "claim_tier": CONJECTURE,
            "statistical_tier": "UNVERIFIED",
            "ablation_status": "NOT_VERIFIED",
            "reasons": [reason],
            "bundle_file_sha256": "",
            "trust_config_sha256": "",
            "verifier_implementation_sha256": verifier_implementation_sha256(),
            "artifact_verification": {},
            "core_certificate": {},
        }
    )


def _validate_supported_comparison(bundle: Mapping[str, Any]) -> None:
    preregistration = bundle.get("preregistration")
    comparison = (
        preregistration.get("comparison_kind") if isinstance(preregistration, Mapping) else None
    )
    if comparison != _SUPPORTED_COMPARISON:
        _fail("external_frontier_comparable_telemetry_unimplemented")


def verify_frontier_evidence_package(
    *,
    bundle_path: Path,
    manifest_path: Path,
    artifact_root: Path,
    trust_path: Path,
) -> dict[str, Any]:
    """Verify raw files and the cryptographic/statistical certificate together."""

    try:
        bundle, bundle_bytes, manifest, manifest_bytes, trust, trust_bytes = _load_inputs(
            bundle_path=bundle_path,
            manifest_path=manifest_path,
            trust_path=trust_path,
        )
        _validate_supported_comparison(bundle)
        artifact_receipt = verify_raw_artifact_package(
            bundle,
            manifest,
            manifest_bytes,
            artifact_root=Path(artifact_root),
        )
        core_certificate = verify_frontier_gain_bundle(
            bundle,
            trusted_verifiers=trust["verifiers"],
            trusted_task_issuers=trust["task_issuers"],
        )
    except (FrontierArtifactError, FrontierVerificationError) as exc:
        return _rejection_certificate(exc.code)
    except (TypeError, ValueError, OverflowError, RecursionError, MemoryError):
        return _rejection_certificate("frontier_verifier_internal_validation_failure")

    accepted = artifact_receipt.get("accepted") is True and core_certificate.get("accepted") is True
    reasons = [] if accepted else list(core_certificate.get("reasons") or [])
    statistical_claim = core_certificate.get("statistical_claim")
    statistical_tier = (
        str(statistical_claim.get("tier") or "UNVERIFIED")
        if isinstance(statistical_claim, Mapping)
        else "UNVERIFIED"
    )
    certificate = {
        "schema": VERIFICATION_CERTIFICATE_SCHEMA,
        "accepted": accepted,
        "release_accepted": False,
        "verification_status": "ACCEPTED" if accepted else "VALID_EVIDENCE_NOT_PROVEN",
        "claim_tier": PROVEN if accepted else CONJECTURE,
        "statistical_tier": statistical_tier,
        "ablation_status": "NOT_VERIFIED",
        "reasons": sorted(set(str(reason) for reason in reasons)),
        "bundle_file_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "trust_config_sha256": hashlib.sha256(trust_bytes).hexdigest(),
        "verifier_implementation_sha256": verifier_implementation_sha256(),
        "artifact_verification": artifact_receipt,
        "core_certificate": core_certificate,
    }
    return _finalize_certificate(certificate)


def prepare_independent_attestation_request(
    *,
    bundle_path: Path,
    manifest_path: Path,
    artifact_root: Path,
    trust_path: Path,
    verifier_id: str,
    verified_at: float,
) -> dict[str, Any]:
    """Prepare exact bytes for an externally held Ed25519 verifier key."""

    if isinstance(verified_at, bool):
        _fail("attestation_verified_at_invalid")
    try:
        verified_at_value = float(verified_at)
    except (TypeError, ValueError, OverflowError):
        _fail("attestation_verified_at_invalid")
    if not math.isfinite(verified_at_value) or verified_at_value <= 0.0:
        _fail("attestation_verified_at_invalid")
    bundle, bundle_bytes, manifest, manifest_bytes, trust, trust_bytes = _load_inputs(
        bundle_path=bundle_path,
        manifest_path=manifest_path,
        trust_path=trust_path,
    )
    _validate_supported_comparison(bundle)
    if "independent_verifier" in bundle:
        _fail("attestation_prepare_requires_unsigned_bundle")
    try:
        artifact_receipt = verify_raw_artifact_package(
            bundle,
            manifest,
            manifest_bytes,
            artifact_root=Path(artifact_root),
        )
        preattestation = verify_frontier_gain_bundle(
            bundle,
            trusted_verifiers=trust["verifiers"],
            trusted_task_issuers=trust["task_issuers"],
        )
    except FrontierArtifactError as exc:
        _fail(exc.code)
    reasons = set(str(reason) for reason in preattestation.get("reasons") or [])
    statistical_claim = preattestation.get("statistical_claim")
    if (
        reasons != {_EXPECTED_PREATTESTATION_REASON}
        or not isinstance(statistical_claim, Mapping)
        or statistical_claim.get("tier") != PROVEN
    ):
        _fail("attestation_evidence_not_preverified")
    pin = trust["verifiers"].get(verifier_id)
    if pin is None:
        _fail("attestation_verifier_not_trusted")
    producer_id = str(bundle.get("producer_id") or "")
    task_issuer_id = str(preattestation.get("task_issuer_id") or "")
    if verifier_id in {producer_id, task_issuer_id}:
        _fail("attestation_verifier_role_collision")
    trials = bundle.get("trials")
    latest_evidence_at = max(
        max(float(trial.get("task_generated_at") or 0.0) for trial in trials),
        max(float(trial.get("evaluation_started_at") or 0.0) for trial in trials),
    )
    if verified_at_value <= latest_evidence_at:
        _fail("attestation_verified_at_not_after_evidence")

    signed_payload = {
        "accepted": True,
        "claim_tier": PROVEN,
        "producer_id": producer_id,
        "evidence_payload_sha256": evidence_payload_sha256(bundle),
        "preregistration_sha256": bundle.get("preregistration_sha256"),
        "implementation_sha256": pin["implementation_sha256"],
        "verifier_release_sha256": pin["release_sha256"],
        "raw_artifact_manifest_sha256": bundle.get("raw_artifact_manifest_sha256"),
        "task_commitment_sha256": bundle.get("task_commitment_sha256"),
        "verified_at": verified_at_value,
    }
    request: dict[str, Any] = {
        "schema": ATTESTATION_REQUEST_SCHEMA,
        "envelope_schema": INDEPENDENT_ATTESTATION_SCHEMA,
        "signer": {
            "algorithm": "Ed25519",
            "signer_id": verifier_id,
            "public_key_b64": pin["public_key_b64"],
        },
        "signed_payload": signed_payload,
        "signed_payload_sha256": canonical_sha256(signed_payload),
        "bundle_file_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "trust_config_sha256": hashlib.sha256(trust_bytes).hexdigest(),
        "artifact_verification_receipt_sha256": artifact_receipt["receipt_sha256"],
        "verifier_implementation_sha256": verifier_implementation_sha256(),
    }
    request["request_sha256"] = canonical_sha256(request)
    return request


__all__ = [
    "ATTESTATION_REQUEST_SCHEMA",
    "FrontierVerificationError",
    "TRUST_CONFIG_SCHEMA",
    "VERIFICATION_CERTIFICATE_SCHEMA",
    "VERIFICATION_KERNEL_IDENTITY_SCHEMA",
    "prepare_independent_attestation_request",
    "validate_trust_config",
    "verifier_implementation_sha256",
    "verify_frontier_evidence_package",
]
