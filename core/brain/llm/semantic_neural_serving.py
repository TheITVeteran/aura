"""Evidence-bound activation for the qualified semantic neural machine."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

SEMANTIC_NEURAL_SERVING_SCHEMA: Final = "aura.semantic_neural_serving.v1"
SEMANTIC_NEURAL_SERVING_MODE: Final = "qualified_exact_semantic_v1"
PACKAGE_ID: Final = "cp556-resident-semantic-neural-serving"
REPO_ROOT: Final = Path(__file__).resolve().parents[3]
DEFAULT_ACTIVATION_PATH: Final = (
    REPO_ROOT / "artifacts/closeout/latent_cortex/cp556_semantic_neural_runtime/activation.json"
)
CP555_RESULT_PATH: Final = (
    REPO_ROOT / "artifacts/closeout/latent_cortex/cp555_resident_semantic_decode_canary/result.json"
)
CP555_VERIFICATION_PATH: Final = (
    REPO_ROOT
    / "artifacts/closeout/latent_cortex/cp555_resident_semantic_decode_canary/verification.json"
)
MEASURED_SOURCE_FILES: Final = (
    "core/brain/llm/latent_cortex/semantic_neural_decode_context.py",
    "core/learning/public_frontier_action_compiler.py",
    "core/learning/semantic_neural_machine.py",
)
ACTIVATION_SOURCE_FILES: Final = (
    *MEASURED_SOURCE_FILES,
    "core/brain/foreground_latent_runtime.py",
    "core/brain/latent_cortex_service.py",
    "core/brain/llm/latent_cortex/assets/systematic_neural_alu_v1/manifest.json",
    "core/brain/llm/latent_cortex/assets/systematic_neural_alu_v1/weights.safetensors",
    "core/brain/llm/latent_cortex/frontier_tasks.py",
    "core/brain/llm/latent_cortex/persistence.py",
    "core/brain/llm/latent_cortex/systematic_neural_alu.py",
    "core/brain/llm/qualified_recurrent_ingress.py",
    "core/brain/llm/semantic_neural_serving.py",
    "core/learning/semantic_neural_controls.py",
    "core/learning/recurrent_action_schema.py",
    "core/learning/recurrent_state_schema.py",
    "core/learning/systematic_neural_alu_training.py",
    "core/phases/response_generation_unitary.py",
)
ALLOWED_FAMILIES: Final = (
    "frontier_calibration",
    "frontier_coding",
    "frontier_misleading_premise",
)
_FALSE_VALUES: Final = frozenset({"0", "false", "no", "off", "disabled"})


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_bounded_json(path: Path, *, maximum_bytes: int) -> tuple[dict[str, Any], bytes]:
    resolved = path.expanduser().resolve(strict=True)
    metadata = resolved.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or path.is_symlink()
        or metadata.st_size <= 1
        or metadata.st_size > maximum_bytes
    ):
        raise RuntimeError(f"semantic serving evidence file is invalid: {path.name}")
    raw = resolved.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError(f"semantic serving evidence JSON is invalid: {path.name}") from None
    if not isinstance(payload, dict):
        raise RuntimeError(f"semantic serving evidence is not an object: {path.name}")
    return payload, raw


def _identity_for_model(model_path: Path) -> dict[str, str]:
    resolved = model_path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "config_sha256": _file_sha(resolved / "config.json"),
        "weights_index_sha256": _file_sha(resolved / "model.safetensors.index.json"),
    }


def _identity_for_manifest(manifest_path: Path) -> dict[str, Any]:
    payload, raw = _read_bounded_json(manifest_path, maximum_bytes=64 * 1024)
    active_path = (
        Path(str(payload.get("active_model_path") or "")).expanduser().resolve(strict=True)
    )
    return {
        "path": str(manifest_path.expanduser().resolve(strict=True)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "active_model_path": str(active_path),
        "schema_version": payload.get("schema_version"),
        "base_model": str(payload.get("base_model") or ""),
        "tag": str(payload.get("tag") or ""),
        "fused_at": payload.get("fused_at"),
    }


def _relative_evidence_path(repo_root: Path, path: Path) -> str:
    resolved = path.expanduser().resolve(strict=True)
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError:
        raise RuntimeError("semantic serving evidence is outside the repository") from None
    return relative.as_posix()


def _resolve_evidence_path(repo_root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise RuntimeError("semantic serving evidence path is not relative")
    candidate = (repo_root / value).resolve(strict=True)
    try:
        candidate.relative_to(repo_root)
    except ValueError:
        raise RuntimeError("semantic serving evidence escaped the repository") from None
    return candidate


def _verify_cp555_evidence(
    *,
    repo_root: Path,
    result_path: Path,
    verification_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], bytes, bytes]:
    result, result_raw = _read_bounded_json(result_path, maximum_bytes=4 * 1024 * 1024)
    verification, verification_raw = _read_bounded_json(
        verification_path,
        maximum_bytes=512 * 1024,
    )
    result_body = {key: value for key, value in result.items() if key != "receipt_sha256"}
    verification_body = {
        key: value for key, value in verification.items() if key != "verification_receipt_sha256"
    }
    if (
        result.get("receipt_sha256") != _sha(result_body)
        or verification.get("verification_receipt_sha256") != _sha(verification_body)
        or verification.get("verified") is not True
        or verification.get("artifact_sha256") != hashlib.sha256(result_raw).hexdigest()
        or verification.get("artifact_receipt_sha256") != result.get("receipt_sha256")
        or verification.get("gain_count", 0) <= 0
        or verification.get("regression_count") != 0
        or "resident model bound by" not in str(verification.get("claim_boundary") or "")
        or "not open-domain" not in str(verification.get("claim_boundary") or "")
    ):
        raise RuntimeError("CP555 semantic serving evidence is not admissible")
    measured_hashes = result.get("source_sha256s")
    if not isinstance(measured_hashes, dict) or any(
        measured_hashes.get(relative) != _file_sha(repo_root / relative)
        for relative in MEASURED_SOURCE_FILES
    ):
        raise RuntimeError("CP555 measured semantic source has drifted")
    return result, verification, result_raw, verification_raw


def build_semantic_neural_activation(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path = CP555_RESULT_PATH,
    verification_path: Path = CP555_VERIFICATION_PATH,
    resident_manifest_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    """Materialize a source/model/evidence-bound qualified activation."""

    root = repo_root.expanduser().resolve(strict=True)
    result, verification, result_raw, verification_raw = _verify_cp555_evidence(
        repo_root=root,
        result_path=result_path,
        verification_path=verification_path,
    )
    model_identity = _identity_for_model(model_path)
    manifest_identity = _identity_for_manifest(resident_manifest_path)
    if (
        model_identity != verification.get("model_identity")
        or manifest_identity != verification.get("resident_manifest_identity")
        or manifest_identity["active_model_path"] != model_identity["path"]
    ):
        raise RuntimeError("semantic serving resident identity differs from CP555")
    body = {
        "schema": SEMANTIC_NEURAL_SERVING_SCHEMA,
        "package_id": PACKAGE_ID,
        "mode": SEMANTIC_NEURAL_SERVING_MODE,
        "active_by_default": True,
        "allowed_families": list(ALLOWED_FAMILIES),
        "source_sha256s": {
            relative: _file_sha(root / relative) for relative in ACTIVATION_SOURCE_FILES
        },
        "model_identity": model_identity,
        "resident_manifest_identity": manifest_identity,
        "evidence": {
            "result_path": _relative_evidence_path(root, result_path),
            "result_sha256": hashlib.sha256(result_raw).hexdigest(),
            "result_receipt_sha256": result["receipt_sha256"],
            "verification_path": _relative_evidence_path(root, verification_path),
            "verification_sha256": hashlib.sha256(verification_raw).hexdigest(),
            "verification_receipt_sha256": verification["verification_receipt_sha256"],
            "gain_count": verification["gain_count"],
            "regression_count": verification["regression_count"],
            "paired_one_sided_exact_p": verification["paired_one_sided_exact_p"],
        },
        "claim_boundary": verification["claim_boundary"],
    }
    return {**body, "activation_sha256": _sha(body)}


def semantic_neural_activation_errors(
    activation: dict[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    model_path: Path | None = None,
    verify_live_identity: bool = True,
) -> list[str]:
    """Recompute every mutable dependency of a serving activation."""

    errors: list[str] = []
    body = {key: value for key, value in activation.items() if key != "activation_sha256"}
    if activation.get("schema") != SEMANTIC_NEURAL_SERVING_SCHEMA:
        errors.append("schema")
    if activation.get("package_id") != PACKAGE_ID:
        errors.append("package_id")
    if activation.get("mode") != SEMANTIC_NEURAL_SERVING_MODE:
        errors.append("mode")
    if activation.get("active_by_default") is not True:
        errors.append("active_by_default")
    if activation.get("allowed_families") != list(ALLOWED_FAMILIES):
        errors.append("allowed_families")
    if activation.get("activation_sha256") != _sha(body):
        errors.append("activation_sha256")
    root = repo_root.expanduser().resolve(strict=True)
    source_hashes = activation.get("source_sha256s")
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(ACTIVATION_SOURCE_FILES):
        errors.append("source_inventory")
    elif any(
        source_hashes.get(relative) != _file_sha(root / relative)
        for relative in ACTIVATION_SOURCE_FILES
    ):
        errors.append("source_drift")
    evidence = activation.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("evidence")
    else:
        try:
            result_path = _resolve_evidence_path(root, evidence["result_path"])
            verification_path = _resolve_evidence_path(
                root,
                evidence["verification_path"],
            )
            result, verification, result_raw, verification_raw = _verify_cp555_evidence(
                repo_root=root,
                result_path=result_path,
                verification_path=verification_path,
            )
            if (
                hashlib.sha256(result_raw).hexdigest() != evidence.get("result_sha256")
                or result.get("receipt_sha256") != evidence.get("result_receipt_sha256")
                or hashlib.sha256(verification_raw).hexdigest()
                != evidence.get("verification_sha256")
                or verification.get("verification_receipt_sha256")
                != evidence.get("verification_receipt_sha256")
            ):
                errors.append("evidence_drift")
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            errors.append("evidence_invalid")
    if verify_live_identity:
        try:
            manifest_identity = activation["resident_manifest_identity"]
            current_manifest = _identity_for_manifest(Path(manifest_identity["path"]))
            expected_model = activation["model_identity"]
            selected_model = (
                Path(model_path).expanduser().resolve(strict=True)
                if model_path is not None
                else Path(str(expected_model["path"])).resolve(strict=True)
            )
            if current_manifest != manifest_identity:
                errors.append("resident_manifest_drift")
            if _identity_for_model(selected_model) != expected_model:
                errors.append("model_identity_drift")
            if current_manifest["active_model_path"] != str(selected_model):
                errors.append("active_model_mismatch")
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            errors.append("live_identity_invalid")
    return sorted(set(errors))


def semantic_neural_serving_status(model_path: str | Path) -> dict[str, Any]:
    """Return active only while code, evidence, manifest, and model still agree."""

    if str(os.getenv("AURA_SEMANTIC_NEURAL_SERVING", "1")).strip().lower() in _FALSE_VALUES:
        return {
            "active": False,
            "reason": "semantic_neural_serving_disabled",
        }
    try:
        activation, _raw = _read_bounded_json(
            DEFAULT_ACTIVATION_PATH,
            maximum_bytes=512 * 1024,
        )
        root = REPO_ROOT.resolve(strict=True)
        evidence = activation["evidence"]
        resident_identity = activation["resident_manifest_identity"]
        selected_model = Path(model_path).expanduser().resolve(strict=True)
        dependencies = (
            DEFAULT_ACTIVATION_PATH,
            *tuple(root / relative for relative in ACTIVATION_SOURCE_FILES),
            _resolve_evidence_path(root, evidence["result_path"]),
            _resolve_evidence_path(root, evidence["verification_path"]),
            Path(resident_identity["path"]),
            selected_model / "config.json",
            selected_model / "model.safetensors.index.json",
        )
        signature_items: list[tuple[str, int, int, int, int]] = []
        for path in dependencies:
            resolved = path.expanduser().resolve(strict=True)
            metadata = resolved.stat()
            signature_items.append(
                (
                    str(resolved),
                    metadata.st_ino,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                )
            )
        return deepcopy(
            _cached_semantic_neural_serving_status(
                str(selected_model),
                tuple(signature_items),
            )
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "active": False,
            "reason": f"semantic_neural_activation_unavailable:{type(exc).__name__}",
        }


@lru_cache(maxsize=8)
def _cached_semantic_neural_serving_status(
    model_path: str,
    _dependency_signature: tuple[tuple[str, int, int, int, int], ...],
) -> dict[str, Any]:
    activation, _raw = _read_bounded_json(
        DEFAULT_ACTIVATION_PATH,
        maximum_bytes=512 * 1024,
    )
    errors = semantic_neural_activation_errors(
        activation,
        model_path=Path(model_path),
    )
    if errors:
        return {
            "active": False,
            "reason": "semantic_neural_activation_invalid:" + ",".join(errors),
        }
    public_receipt = {
        key: activation[key]
        for key in (
            "schema",
            "package_id",
            "mode",
            "allowed_families",
            "model_identity",
            "resident_manifest_identity",
            "claim_boundary",
            "activation_sha256",
        )
    }
    return {
        "active": True,
        "reason": "semantic_neural_serving_active",
        "receipt": public_receipt,
    }


__all__ = [
    "ACTIVATION_SOURCE_FILES",
    "ALLOWED_FAMILIES",
    "DEFAULT_ACTIVATION_PATH",
    "SEMANTIC_NEURAL_SERVING_MODE",
    "SEMANTIC_NEURAL_SERVING_SCHEMA",
    "build_semantic_neural_activation",
    "semantic_neural_activation_errors",
    "semantic_neural_serving_status",
]
