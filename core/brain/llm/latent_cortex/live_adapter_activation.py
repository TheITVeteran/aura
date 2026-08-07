"""Externally authorized admission for a live recurrent adapter.

Training completion is not promotion authority. A resident worker may attach a
recurrence-native adapter only when an externally rooted evidence-verifier
attestation signs an activation record that is bound to:

* a complete, independently verified same-checkpoint gain certificate;
* the frozen campaign plan and its exact adapter identity;
* the current base checkpoint, tokenizer behavior, personality, and runtime;
* a role-conditioned v3 package whose bytes are reopened at admission.

The adapter's own training manifest remains honest (``promotion_allowed`` is
false there). The separate activation attestation is the later scientific
authority that permits live use.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, Never

from core.brain.llm.latent_cortex.adapter_identity import (
    inspect_mlx_tensor_metadata,
)
from core.brain.llm.latent_cortex.campaign_journal import (
    CampaignPlan,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.campaign_trust import (
    EVIDENCE_VERIFIER,
    externally_custodied_roles,
    validate_campaign_trust_policy,
    verify_role_attestation,
)
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
    full_weight_checkpoint_identity,
    model_behavior_bundle_identity,
    personality_bundle_identity,
    runtime_environment_identity,
    strict_json_loads,
)
from core.brain.llm.latent_cortex.resident_adapter_loader import (
    load_resident_adapter,
)
from core.brain.llm.latent_cortex.resident_recurrent_sft_adapter_identity import (
    ROLE_CONDITIONED_MANIFEST_SCHEMA,
    declared_bindings,
    validate_resident_recurrent_sft_adapter_identity,
)
from core.runtime.file_read_gateway import read_stable_bytes

ACTIVATION_POINTER_SCHEMA: Final = "aura.latent_cortex.live_adapter_pointer.v1"
ACTIVATION_SCHEMA: Final = "aura.latent_cortex.live_adapter_activation.v1"
ACTIVATION_RECEIPT_SCHEMA: Final = (
    "aura.latent_cortex.live_adapter_activation_receipt.v1"
)
INDEPENDENT_VERDICT_SCHEMA: Final = (
    "aura.latent_cortex.independent_evidence_verdict.v2"
)
MAX_JSON_BYTES: Final = 64 * 1024 * 1024
MAX_ARTIFACT_BYTES: Final = 1 << 40
MAX_TRUST_ROOT_BYTES: Final = 1024 * 1024
_BINDING_KEYS = {"path", "sha256", "size_bytes"}
_RUNTIME_CONTRACT = {
    "activation_default": "on",
    "activation_scope": "recurrence_adapter_scope_only",
    "manifest_schema": ROLE_CONDITIONED_MANIFEST_SCHEMA,
    "depth_conditioning_required": True,
    "role_conditioning_required": True,
    "ordinary_decode_unchanged_outside_scope": True,
    "rollback_contract": "exact_original_module_graph_v1",
}


class LiveAdapterActivationError(RuntimeError):
    """A live adapter was not authorized by complete positive evidence."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise LiveAdapterActivationError(code)


def _sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"live_adapter_{role}_sha256_invalid")
    return value


def _positive_int(value: Any, *, role: str) -> int:
    if type(value) is not int or value <= 0:
        _fail(f"live_adapter_{role}_invalid")
    return value


def _identifier(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 240
    ):
        _fail(f"live_adapter_{role}_invalid")
    return value


def _reject_symlink_chain(path: str | Path, *, role: str) -> Path:
    lexical = Path(path).expanduser().absolute()
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise LiveAdapterActivationError(
                f"live_adapter_{role}_unavailable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            _fail(f"live_adapter_{role}_symlink_forbidden")
    try:
        return lexical.resolve(strict=True)
    except OSError as exc:
        raise LiveAdapterActivationError(
            f"live_adapter_{role}_unavailable"
        ) from exc


def _stable_bytes(
    path: str | Path,
    *,
    role: str,
    maximum: int = MAX_ARTIFACT_BYTES,
) -> tuple[bytes, Path]:
    resolved = _reject_symlink_chain(path, role=role)
    try:
        before = resolved.stat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) & 0o022
            or not 0 < before.st_size <= maximum
        ):
            _fail(f"live_adapter_{role}_storage_invalid")
        payload = read_stable_bytes(resolved, max_bytes=maximum)
        after = resolved.stat()
    except (OSError, ValueError) as exc:
        raise LiveAdapterActivationError(
            f"live_adapter_{role}_unreadable"
        ) from exc
    identity = lambda item: (  # noqa: E731 - compact immutable stat tuple
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after) or len(payload) != before.st_size:
        _fail(f"live_adapter_{role}_changed_while_reading")
    return payload, resolved


def _json_bytes(
    path: str | Path,
    *,
    role: str,
) -> tuple[dict[str, Any], bytes, Path]:
    payload, resolved = _stable_bytes(path, role=role, maximum=MAX_JSON_BYTES)
    try:
        value = strict_json_loads(payload, role=f"live_adapter_{role}")
    except ValueError as exc:
        raise LiveAdapterActivationError(
            f"live_adapter_{role}_json_invalid"
        ) from exc
    if not isinstance(value, Mapping):
        _fail(f"live_adapter_{role}_schema_invalid")
    return dict(value), payload, resolved


def read_live_adapter_trust_root(path: str | Path) -> bytes:
    """Read a private, stable, non-symlinked external trust root."""

    payload, _resolved = _stable_bytes(
        path,
        role="trust_root",
        maximum=MAX_TRUST_ROOT_BYTES,
    )
    return payload


def _binding(value: Any, *, role: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _BINDING_KEYS:
        _fail(f"live_adapter_{role}_binding_invalid")
    path = _identifier(value.get("path"), role=f"{role}_path")
    if not Path(path).is_absolute():
        _fail(f"live_adapter_{role}_path_not_absolute")
    return {
        "path": path,
        "sha256": _sha256(value.get("sha256"), role=role),
        "size_bytes": _positive_int(value.get("size_bytes"), role=f"{role}_size"),
    }


def _read_binding(value: Any, *, role: str) -> tuple[dict[str, Any], bytes, Path]:
    binding = _binding(value, role=role)
    payload, path = _stable_bytes(
        binding["path"],
        role=role,
        maximum=binding["size_bytes"],
    )
    if (
        len(payload) != binding["size_bytes"]
        or hashlib.sha256(payload).hexdigest() != binding["sha256"]
    ):
        _fail(f"live_adapter_{role}_binding_mismatch")
    return binding, payload, path


def _contained_directory(
    value: Any,
    *,
    roots: Sequence[str | Path],
) -> Path:
    path = _reject_symlink_chain(
        _identifier(value, role="package_path"),
        role="package",
    )
    if not path.is_dir():
        _fail("live_adapter_package_not_directory")
    admitted = False
    for root_value in roots:
        root = _reject_symlink_chain(root_value, role="approved_root")
        if not root.is_dir():
            _fail("live_adapter_approved_root_not_directory")
        try:
            path.relative_to(root)
            admitted = True
            break
        except ValueError:
            continue
    if not admitted:
        _fail("live_adapter_package_outside_approved_roots")
    return path


def _contained_artifact(
    root: Path,
    relative: Any,
    *,
    role: str,
    maximum: int = MAX_ARTIFACT_BYTES,
) -> bytes:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        _fail(f"live_adapter_{role}_relative_path_invalid")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or str(pure) != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        _fail(f"live_adapter_{role}_relative_path_invalid")
    resolved = _reject_symlink_chain(root.joinpath(*pure.parts), role=role)
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail(f"live_adapter_{role}_outside_package")
    payload, _path = _stable_bytes(resolved, role=role, maximum=maximum)
    return payload


def _validate_activation(value: Any) -> dict[str, Any]:
    keys = {
        "schema",
        "campaign_name",
        "policy_sha256",
        "adapter_id",
        "adapter_package_path",
        "adapter_manifest_sha256",
        "adapter_composite_identity_sha256",
        "base_checkpoint_fingerprint",
        "model_behavior_bundle_sha256",
        "campaign_plan",
        "independent_verdict",
        "runtime_contract",
        "not_before_unix",
        "expires_at_unix",
        "activation_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        _fail("live_adapter_activation_schema_invalid")
    activation = dict(value)
    material = dict(activation)
    claimed = material.pop("activation_sha256", None)
    if (
        activation.get("schema") != ACTIVATION_SCHEMA
        or claimed != hashlib.sha256(canonical_json_bytes(material)).hexdigest()
        or activation.get("runtime_contract") != _RUNTIME_CONTRACT
    ):
        _fail("live_adapter_activation_invalid")
    for role in (
        "policy",
        "adapter_manifest",
        "adapter_composite_identity",
        "base_checkpoint_fingerprint",
        "model_behavior_bundle",
    ):
        key = {
            "policy": "policy_sha256",
            "adapter_manifest": "adapter_manifest_sha256",
            "adapter_composite_identity": "adapter_composite_identity_sha256",
            "base_checkpoint_fingerprint": "base_checkpoint_fingerprint",
            "model_behavior_bundle": "model_behavior_bundle_sha256",
        }[role]
        _sha256(activation.get(key), role=role)
    _identifier(activation.get("campaign_name"), role="campaign_name")
    _identifier(activation.get("adapter_id"), role="adapter_id")
    _identifier(activation.get("adapter_package_path"), role="package_path")
    _binding(activation.get("campaign_plan"), role="campaign_plan")
    _binding(activation.get("independent_verdict"), role="independent_verdict")
    not_before = _positive_int(
        activation.get("not_before_unix"),
        role="not_before_unix",
    )
    expires = _positive_int(
        activation.get("expires_at_unix"),
        role="expires_at_unix",
    )
    if expires <= not_before:
        _fail("live_adapter_activation_window_invalid")
    return activation


def build_live_adapter_activation(
    *,
    campaign_name: str,
    policy_sha256: str,
    adapter_id: str,
    adapter_package_path: str | Path,
    adapter_manifest_sha256: str,
    adapter_composite_identity_sha256: str,
    base_checkpoint_fingerprint: str,
    model_behavior_bundle_sha256: str,
    campaign_plan: Mapping[str, Any],
    independent_verdict: Mapping[str, Any],
    not_before_unix: int,
    expires_at_unix: int,
) -> dict[str, Any]:
    """Build the exact externally attested production-activation payload."""

    material = {
        "schema": ACTIVATION_SCHEMA,
        "campaign_name": campaign_name,
        "policy_sha256": policy_sha256,
        "adapter_id": adapter_id,
        "adapter_package_path": str(Path(adapter_package_path).expanduser().absolute()),
        "adapter_manifest_sha256": adapter_manifest_sha256,
        "adapter_composite_identity_sha256": adapter_composite_identity_sha256,
        "base_checkpoint_fingerprint": base_checkpoint_fingerprint,
        "model_behavior_bundle_sha256": model_behavior_bundle_sha256,
        "campaign_plan": dict(campaign_plan),
        "independent_verdict": dict(independent_verdict),
        "runtime_contract": dict(_RUNTIME_CONTRACT),
        "not_before_unix": not_before_unix,
        "expires_at_unix": expires_at_unix,
    }
    return _validate_activation(
        {
            **material,
            "activation_sha256": hashlib.sha256(
                canonical_json_bytes(material)
            ).hexdigest(),
        }
    )


def _validate_pointer(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "activation",
        "campaign_policy",
        "activation_attestation",
        "pointer_sha256",
    }:
        _fail("live_adapter_pointer_schema_invalid")
    pointer = dict(value)
    material = dict(pointer)
    claimed = material.pop("pointer_sha256", None)
    if (
        pointer.get("schema") != ACTIVATION_POINTER_SCHEMA
        or claimed != hashlib.sha256(canonical_json_bytes(material)).hexdigest()
    ):
        _fail("live_adapter_pointer_invalid")
    return {
        **material,
        "activation": _validate_activation(pointer.get("activation")),
        "campaign_policy": _binding(pointer.get("campaign_policy"), role="campaign_policy"),
        "activation_attestation": _binding(
            pointer.get("activation_attestation"),
            role="activation_attestation",
        ),
        "pointer_sha256": claimed,
    }


def build_live_adapter_pointer(
    *,
    activation: Mapping[str, Any],
    campaign_policy: Mapping[str, Any],
    activation_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable pointer published only after detached signing."""

    material = {
        "schema": ACTIVATION_POINTER_SCHEMA,
        "activation": dict(activation),
        "campaign_policy": dict(campaign_policy),
        "activation_attestation": dict(activation_attestation),
    }
    return _validate_pointer(
        {
            **material,
            "pointer_sha256": hashlib.sha256(canonical_json_bytes(material)).hexdigest(),
        }
    )


def _validate_positive_verdict(
    verdict: Mapping[str, Any],
    *,
    activation: Mapping[str, Any],
    plan: CampaignPlan,
    certificate_path: Path,
) -> None:
    plan_metadata = plan.to_dict().get("metadata")
    adapter_identity = (
        plan_metadata.get("adapter_identity")
        if isinstance(plan_metadata, Mapping)
        else None
    )
    model_identity = (
        plan_metadata.get("model_identity")
        if isinstance(plan_metadata, Mapping)
        else None
    )
    identity_receipt = (
        adapter_identity.get("identity_receipt")
        if isinstance(adapter_identity, Mapping)
        else None
    )
    campaign_dir = certificate_path.parent
    if (
        verdict.get("schema") != INDEPENDENT_VERDICT_SCHEMA
        or verdict.get("campaign_dir") != str(campaign_dir)
        or verdict.get("passed") is not True
        or verdict.get("claim_tier") != "PROVEN"
        or verdict.get("verified_verdict") != "gain_proven"
        or verdict.get("failures") != []
        or verdict.get("recomputed_verdict") != "gain_preverified"
        or verdict.get("independent_verdict") != "gain_preverified"
        or verdict.get("published_verdict") != "gain_preverified"
        or verdict.get("production_semantic_grade_sha256")
        != verdict.get("independent_semantic_grade_sha256")
        or verdict.get("plan_sha256") != plan.plan_sha256
        or plan.campaign_name != activation["campaign_name"]
        or not isinstance(plan_metadata, Mapping)
        or plan_metadata.get("claim_eligible") is not True
        or not isinstance(identity_receipt, Mapping)
        or identity_receipt.get("composite_identity_sha256")
        != activation["adapter_composite_identity_sha256"]
        or identity_receipt.get("manifest_sha256")
        != activation["adapter_manifest_sha256"]
        or identity_receipt.get("base_checkpoint_fingerprint")
        != activation["base_checkpoint_fingerprint"]
        or identity_receipt.get("model_behavior_bundle_sha256")
        != activation["model_behavior_bundle_sha256"]
        or not isinstance(model_identity, Mapping)
        or model_identity.get("fingerprint")
        != activation["base_checkpoint_fingerprint"]
    ):
        _fail("live_adapter_positive_certificate_invalid")
    for role in ("answer_reveal", "worker_origins", "final_run"):
        evidence = verdict.get(role)
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("required") is not True
            or evidence.get("verified") is not True
        ):
            _fail(f"live_adapter_{role}_unverified")
    _sha256(
        verdict.get("verifier_attestation_sha256"),
        role="final_verifier_attestation",
    )
    for role in (
        "production_semantic_grade",
        "independent_semantic_grade",
        "production_grade_implementation",
        "independent_scoring_implementation",
        "verifier_implementation",
    ):
        _sha256(verdict.get(f"{role}_sha256"), role=role)


def admit_live_adapter_activation(
    pointer_path: str | Path,
    *,
    trusted_root_public_key_pem: bytes,
    approved_adapter_roots: Sequence[str | Path],
    actual_base_checkpoint: Mapping[str, Any],
    actual_model_behavior_bundle: Mapping[str, Any],
    actual_personality_adapter: Mapping[str, Any],
    actual_runtime_environment: Mapping[str, Any],
    now_unix: int,
) -> dict[str, Any]:
    """Authenticate, reopen, and revalidate a production adapter activation."""

    pointer, _pointer_raw, _pointer_path = _json_bytes(
        pointer_path,
        role="activation_pointer",
    )
    pointer = _validate_pointer(pointer)
    activation = pointer["activation"]
    if type(now_unix) is not int or not (
        activation["not_before_unix"]
        <= now_unix
        < activation["expires_at_unix"]
    ):
        _fail("live_adapter_activation_not_current")

    policy_binding, policy_raw, _policy_path = _read_binding(
        pointer.get("campaign_policy"),
        role="campaign_policy",
    )
    policy_document = strict_json_loads(
        policy_raw,
        role="live_adapter_campaign_policy",
    )
    if (
        policy_binding["sha256"] != activation["policy_sha256"]
        or not isinstance(policy_document, Mapping)
    ):
        _fail("live_adapter_campaign_policy_mismatch")
    policy = validate_campaign_trust_policy(
        policy_document,
        trusted_root_public_key_pem=trusted_root_public_key_pem,
        expected_campaign_name=activation["campaign_name"],
        expected_policy_sha256=activation["policy_sha256"],
        now_unix=activation["not_before_unix"],
    )
    if not externally_custodied_roles(policy):
        _fail("live_adapter_external_custody_required")

    _attestation_binding, attestation_raw, _attestation_path = _read_binding(
        pointer.get("activation_attestation"),
        role="activation_attestation",
    )
    attestation = strict_json_loads(
        attestation_raw,
        role="live_adapter_activation_attestation",
    )
    verify_role_attestation(
        policy,
        attestation,
        role=EVIDENCE_VERIFIER,
        expected_payload=activation,
        not_before_unix=activation["not_before_unix"],
        not_after_unix=activation["expires_at_unix"] - 1,
    )

    plan_binding, plan_raw, _plan_path = _read_binding(
        activation["campaign_plan"],
        role="campaign_plan",
    )
    try:
        plan_document = strict_json_loads(
            plan_raw,
            role="live_adapter_campaign_plan",
        )
        plan = CampaignPlan.from_dict(plan_document)
    except (TypeError, ValueError) as exc:
        raise LiveAdapterActivationError(
            "live_adapter_campaign_plan_invalid"
        ) from exc
    if plan_binding["sha256"] != hashlib.sha256(plan_raw).hexdigest():
        _fail("live_adapter_campaign_plan_mismatch")

    _verdict_binding, verdict_raw, verdict_path = _read_binding(
        activation["independent_verdict"],
        role="independent_verdict",
    )
    verdict = strict_json_loads(
        verdict_raw,
        role="live_adapter_independent_verdict",
    )
    if not isinstance(verdict, Mapping):
        _fail("live_adapter_independent_verdict_schema_invalid")
    _validate_positive_verdict(
        verdict,
        activation=activation,
        plan=plan,
        certificate_path=verdict_path,
    )

    package = _contained_directory(
        activation["adapter_package_path"],
        roots=approved_adapter_roots,
    )
    manifest_raw = _contained_artifact(
        package,
        "recurrence_adapter_manifest.json",
        role="package_manifest",
        maximum=MAX_JSON_BYTES,
    )
    if hashlib.sha256(manifest_raw).hexdigest() != activation["adapter_manifest_sha256"]:
        _fail("live_adapter_package_manifest_mismatch")
    manifest = strict_json_loads(
        manifest_raw,
        role="live_adapter_package_manifest",
    )
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema") != ROLE_CONDITIONED_MANIFEST_SCHEMA
        or manifest.get("adapter_id") != activation["adapter_id"]
    ):
        _fail("live_adapter_role_conditioned_package_required")
    artifacts = {
        role: _contained_artifact(
            package,
            binding["path"],
            role=f"package_{role}",
        )
        for role, binding in declared_bindings(manifest)
    }
    artifacts["training_completion.json"] = _contained_artifact(
        package,
        "training_completion.json",
        role="package_completion",
        maximum=MAX_JSON_BYTES,
    )
    identity = validate_resident_recurrent_sft_adapter_identity(
        manifest_raw,
        adapter_id=activation["adapter_id"],
        actual_base_checkpoint=actual_base_checkpoint,
        actual_model_behavior_bundle=actual_model_behavior_bundle,
        actual_personality_adapter=actual_personality_adapter,
        actual_runtime_environment=actual_runtime_environment,
        artifacts=artifacts,
        tensor_metadata=inspect_mlx_tensor_metadata(
            package / "adapter.safetensors"
        ),
    )
    if (
        identity.get("composite_identity_sha256")
        != activation["adapter_composite_identity_sha256"]
        or identity.get("manifest_sha256")
        != activation["adapter_manifest_sha256"]
        or identity.get("base_checkpoint_fingerprint")
        != activation["base_checkpoint_fingerprint"]
        or identity.get("model_behavior_bundle_sha256")
        != activation["model_behavior_bundle_sha256"]
    ):
        _fail("live_adapter_runtime_identity_mismatch")

    receipt_material = {
        "schema": ACTIVATION_RECEIPT_SCHEMA,
        "activation_sha256": activation["activation_sha256"],
        "pointer_sha256": pointer["pointer_sha256"],
        "policy_sha256": policy.policy_sha256,
        "campaign_name": activation["campaign_name"],
        "plan_sha256": plan.plan_sha256,
        "verdict_sha256": hashlib.sha256(verdict_raw).hexdigest(),
        "adapter_package_path": str(package),
        "adapter_identity": identity,
        "runtime_contract": dict(_RUNTIME_CONTRACT),
        "claim_tier": "PROVEN",
        "verified_verdict": "gain_proven",
    }
    return {
        **receipt_material,
        "receipt_sha256": hashlib.sha256(
            canonical_json_bytes(receipt_material)
        ).hexdigest(),
        "manifest": dict(manifest),
    }


def attach_certified_live_adapter(
    model: Any,
    *,
    model_path: str | Path,
    personality_adapter_path: str | Path | None,
    pointer_path: str | Path,
    trusted_root_public_key_pem: bytes,
    approved_adapter_roots: Sequence[str | Path],
    now_unix: int,
) -> dict[str, Any]:
    """Admit and transactionally attach one externally certified adapter.

    Identity is measured from the exact model assets used by the worker.
    ``load_resident_adapter`` performs its own independent byte, topology, and
    tensor readback checks and restores the original module graph on failure.
    """

    receipt = admit_live_adapter_activation(
        pointer_path,
        trusted_root_public_key_pem=trusted_root_public_key_pem,
        approved_adapter_roots=approved_adapter_roots,
        actual_base_checkpoint=full_weight_checkpoint_identity(model_path),
        actual_model_behavior_bundle=model_behavior_bundle_identity(model_path),
        actual_personality_adapter=personality_bundle_identity(
            personality_adapter_path
        ),
        actual_runtime_environment=runtime_environment_identity(),
        now_unix=now_unix,
    )
    identity = receipt["adapter_identity"]
    expected_projections = (
        identity.get("lora", {}).get("wrapped_projections")
        if isinstance(identity, Mapping)
        else None
    )
    manifest_lora = receipt["manifest"].get("lora")
    manifest_expected = (
        manifest_lora.get("wrapped_projections")
        if isinstance(manifest_lora, Mapping)
        else None
    )
    if (
        type(expected_projections) is not int
        or expected_projections <= 0
        or manifest_expected != expected_projections
    ):
        _fail("live_adapter_projection_contract_mismatch")
    loaded_projections = load_resident_adapter(
        model,
        receipt["adapter_package_path"],
        receipt["manifest"],
    )
    public_receipt = {
        key: value
        for key, value in receipt.items()
        if key != "manifest"
    }
    public_receipt["loaded_projection_count"] = loaded_projections
    return public_receipt


__all__ = [
    "ACTIVATION_POINTER_SCHEMA",
    "ACTIVATION_RECEIPT_SCHEMA",
    "ACTIVATION_SCHEMA",
    "LiveAdapterActivationError",
    "admit_live_adapter_activation",
    "attach_certified_live_adapter",
    "build_live_adapter_activation",
    "build_live_adapter_pointer",
    "read_live_adapter_trust_root",
]
