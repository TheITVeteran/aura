"""Identity receipts for resident Recursive Latent Cortex episodes."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.runtime.file_read_gateway import read_stable_bytes

WORKER_IDENTITY_SCHEMA = "aura.latent_cortex.worker_identity.v1"
RUNTIME_IDENTITY_SCHEMA = "aura.latent_cortex.runtime_identity.v1"


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _git_oid(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _stable_sha256(path: str | Path, *, max_bytes: int) -> str:
    return hashlib.sha256(read_stable_bytes(path, max_bytes=max_bytes)).hexdigest()


def canonical_model_path(model_path: str | Path) -> str:
    return os.path.realpath(os.path.expanduser(str(model_path)))


def latent_request_payload_sha256(
    *,
    prompt: Any,
    messages: Any,
    domain: str,
    config: Any,
    budget: Any,
    runtime_controls: Any,
) -> str:
    payload = {
        "prompt": prompt,
        "messages": messages,
        "domain": str(domain or "general"),
        "config": config,
        "budget": budget,
        "runtime_controls": runtime_controls,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_parameter_count(model: Any) -> int:
    """Count every resident parameter leaf once without copying tensor data."""

    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise ValueError("resident model does not expose a parameter tree")

    def leaves(node: Any):
        if isinstance(node, Mapping):
            for child in node.values():
                yield from leaves(child)
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                yield from leaves(child)
            return
        yield node

    total = 0
    for tensor in leaves(parameters()):
        size = getattr(tensor, "size", None)
        if callable(size):
            size = size()
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("resident model exposed an invalid parameter leaf size")
        total += size
    if total <= 0:
        raise ValueError("resident model exposed no countable parameters")
    return total


def build_worker_identity(
    model: Any,
    *,
    model_path: str | Path,
    worker_boot_id: str,
    worker_source_path: str | Path,
    affective_steering_active: bool = False,
    affective_steering_alpha: float = 0.0,
) -> dict[str, Any]:
    boot_id = str(worker_boot_id or "").strip().lower()
    if len(boot_id) != 32 or any(character not in "0123456789abcdef" for character in boot_id):
        raise ValueError("worker_boot_id must be a 128-bit lowercase hex identifier")
    return {
        "schema": WORKER_IDENTITY_SCHEMA,
        "worker_boot_id": boot_id,
        "worker_pid": os.getpid(),
        "worker_model_path": canonical_model_path(model_path),
        "worker_model_parameter_count": model_parameter_count(model),
        "worker_source_sha256": _stable_sha256(worker_source_path, max_bytes=8 * 1024 * 1024),
        "worker_affective_steering_active": bool(affective_steering_active),
        "worker_affective_steering_alpha": float(affective_steering_alpha),
    }


def worker_identity_errors(
    receipt: Any,
    *,
    expected: Mapping[str, Any] | None = None,
) -> list[str]:
    if not isinstance(receipt, Mapping):
        return ["worker_identity_receipt_not_mapping"]
    errors: list[str] = []
    boot_id = receipt.get("worker_boot_id")
    if not (
        isinstance(boot_id, str)
        and len(boot_id) == 32
        and all(character in "0123456789abcdef" for character in boot_id)
    ):
        errors.append("invalid_worker_boot_id")
    if type(receipt.get("worker_pid")) is not int or receipt["worker_pid"] <= 0:
        errors.append("invalid_worker_pid")
    if not str(receipt.get("worker_model_path") or "").strip():
        errors.append("missing_worker_model_path")
    if (
        type(receipt.get("worker_model_parameter_count")) is not int
        or receipt["worker_model_parameter_count"] <= 0
    ):
        errors.append("invalid_worker_model_parameter_count")
    if not _sha256(receipt.get("worker_source_sha256")):
        errors.append("invalid_worker_source_sha256")
    if type(receipt.get("worker_affective_steering_active")) is not bool:
        errors.append("invalid_worker_affective_steering_active")
    steering_alpha = receipt.get("worker_affective_steering_alpha")
    if (
        isinstance(steering_alpha, bool)
        or not isinstance(steering_alpha, (int, float))
        or not 0.0 <= float(steering_alpha) <= 1.0
    ):
        errors.append("invalid_worker_affective_steering_alpha")
    if expected is not None:
        for key in (
            "worker_boot_id",
            "worker_pid",
            "worker_model_path",
            "worker_model_parameter_count",
            "worker_source_sha256",
            "worker_affective_steering_active",
            "worker_affective_steering_alpha",
        ):
            if receipt.get(key) != expected.get(key):
                errors.append(f"{key}_mismatch")
    return errors


def collect_latent_runtime_identity(
    project_root: str | Path,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Bind one episode to the exact source and, when applicable, Aura.app."""

    from core.runtime.launch_provenance import (
        collect_runtime_launch_provenance,
        collect_source_identity,
    )

    provenance = collect_runtime_launch_provenance(project_root, env=env)
    required = provenance.get("required") is True
    source = provenance.get("actual") if required else collect_source_identity(project_root)
    source = dict(source) if isinstance(source, Mapping) else {}
    manifest = provenance.get("manifest")
    manifest = dict(manifest) if isinstance(manifest, Mapping) else {}
    issues = [str(item) for item in provenance.get("issues", []) if str(item)]

    app_executable_sha256 = ""
    launch_manifest_sha256 = ""
    if required:
        executable = str(provenance.get("app_executable") or "").strip()
        manifest_path = str(provenance.get("manifest_path") or "").strip()
        try:
            app_executable_sha256 = _stable_sha256(
                executable,
                max_bytes=256 * 1024 * 1024,
            )
            launch_manifest_sha256 = _stable_sha256(
                manifest_path,
                max_bytes=4 * 1024 * 1024,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            issues.append(f"app_identity_hash_failed:{type(exc).__name__}")

    commit_sha = str(source.get("commit_sha") or "").lower()
    workspace_sha256 = str(source.get("workspace_state_sha256") or "").lower()
    shell_assets_sha256 = str(
        source.get("shell_assets_sha256") or manifest.get("shell_assets_sha256") or ""
    ).lower()
    source_bound = bool(
        _git_oid(commit_sha)
        and _sha256(workspace_sha256)
        and _sha256(shell_assets_sha256)
        and provenance.get("source_verified") is True
    )
    installed_app_verified = bool(
        required
        and provenance.get("verified") is True
        and _sha256(app_executable_sha256)
        and _sha256(launch_manifest_sha256)
    )
    identity_bound = bool(source_bound and (not required or installed_app_verified))
    if not source_bound:
        issues.append("source_identity_unbound")
    if required and not installed_app_verified:
        issues.append("installed_app_identity_unbound")

    return {
        "schema": RUNTIME_IDENTITY_SCHEMA,
        "identity_bound": identity_bound,
        "launch_mode": str(provenance.get("launch_mode") or ""),
        "installed_app_required": required,
        "installed_app_verified": installed_app_verified,
        "source_verified": provenance.get("source_verified") is True,
        "source_root": str(source.get("source_root") or provenance.get("source_root") or ""),
        "source_commit": commit_sha,
        "source_branch": str(source.get("branch") or ""),
        "workspace_state_sha256": workspace_sha256,
        "source_dirty": source.get("source_dirty") is True,
        "source_change_count": int(source.get("source_change_count") or 0),
        "shell_assets_sha256": shell_assets_sha256,
        "bundle_identifier": str(
            manifest.get("bundle_identifier")
            or (provenance.get("expected") or {}).get("bundle_identifier")
            or ""
        ),
        "app_executable_sha256": app_executable_sha256,
        "launch_manifest_sha256": launch_manifest_sha256,
        "issues": sorted(set(issues)),
    }


__all__ = [
    "RUNTIME_IDENTITY_SCHEMA",
    "WORKER_IDENTITY_SCHEMA",
    "build_worker_identity",
    "canonical_model_path",
    "collect_latent_runtime_identity",
    "latent_request_payload_sha256",
    "model_parameter_count",
    "worker_identity_errors",
]
