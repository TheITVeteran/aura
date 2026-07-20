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
MAX_AFFECTIVE_STEERING_ALPHA = 50.0


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
    cognitive_context: Any = None,
    response_contract: Any = None,
) -> str:
    payload = {
        "prompt": prompt,
        "messages": messages,
        "domain": str(domain or "general"),
        "config": config,
        "budget": budget,
        "runtime_controls": runtime_controls,
    }
    # Additive so pre-ingress request digests stay reproducible: episodes
    # without typed cognitive context hash exactly as they always did.
    if cognitive_context is not None:
        payload["cognitive_context"] = cognitive_context
    if response_contract is not None:
        payload["response_contract"] = response_contract
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


def logical_model_parameter_count(
    model_path: str | Path,
    *,
    stored_element_count: int,
) -> tuple[int, str]:
    """Derive logical weights from architecture config for packed checkpoints."""

    config_path = Path(canonical_model_path(model_path)) / "config.json"
    try:
        config = json.loads(
            read_stable_bytes(config_path, max_bytes=2 * 1024 * 1024).decode(
                "utf-8"
            )
        )
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return stored_element_count, "stored_tensor_elements"
    if not isinstance(config, Mapping) or str(config.get("model_type") or "") != "qwen2":
        return stored_element_count, "stored_tensor_elements"

    names = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
    )
    values: dict[str, int] = {}
    for name in names:
        value = config.get(name)
        if type(value) is not int or value <= 0:
            return stored_element_count, "stored_tensor_elements"
        values[name] = value
    hidden = values["hidden_size"]
    attention_heads = values["num_attention_heads"]
    head_dim = config.get("head_dim")
    if head_dim is None:
        if hidden % attention_heads:
            return stored_element_count, "stored_tensor_elements"
        head_dim = hidden // attention_heads
    if type(head_dim) is not int or head_dim <= 0:
        return stored_element_count, "stored_tensor_elements"

    query_width = attention_heads * head_dim
    kv_width = values["num_key_value_heads"] * head_dim
    attention_weights = (
        hidden * query_width
        + 2 * hidden * kv_width
        + query_width * hidden
    )
    # Qwen2 q/k/v projections carry bias; o_proj and the gated MLP do not.
    attention_biases = query_width + 2 * kv_width
    mlp_weights = 3 * hidden * values["intermediate_size"]
    layer_norms = 2 * hidden
    per_layer = attention_weights + attention_biases + mlp_weights + layer_norms
    embeddings = values["vocab_size"] * hidden
    output_head = (
        0
        if config.get("tie_word_embeddings") is True
        else values["vocab_size"] * hidden
    )
    logical = (
        embeddings
        + values["num_hidden_layers"] * per_layer
        + hidden
        + output_head
    )
    if logical <= 0:
        return stored_element_count, "stored_tensor_elements"
    return logical, "architecture_config_logical"


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
    stored_element_count = model_parameter_count(model)
    logical_count, count_basis = logical_model_parameter_count(
        model_path,
        stored_element_count=stored_element_count,
    )
    return {
        "schema": WORKER_IDENTITY_SCHEMA,
        "worker_boot_id": boot_id,
        "worker_pid": os.getpid(),
        "worker_model_path": canonical_model_path(model_path),
        "worker_model_parameter_count": logical_count,
        "worker_model_stored_parameter_element_count": stored_element_count,
        "worker_model_parameter_count_basis": count_basis,
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
    if (
        type(receipt.get("worker_model_stored_parameter_element_count")) is not int
        or receipt["worker_model_stored_parameter_element_count"] <= 0
    ):
        errors.append("invalid_worker_model_stored_parameter_element_count")
    count_basis = receipt.get("worker_model_parameter_count_basis")
    if count_basis not in {
        "architecture_config_logical",
        "stored_tensor_elements",
    }:
        errors.append("invalid_worker_model_parameter_count_basis")
    logical_count = receipt.get("worker_model_parameter_count")
    stored_count = receipt.get("worker_model_stored_parameter_element_count")
    if (
        type(logical_count) is int
        and type(stored_count) is int
        and (
            (count_basis == "architecture_config_logical" and logical_count < stored_count)
            or (count_basis == "stored_tensor_elements" and logical_count != stored_count)
        )
    ):
        errors.append("worker_model_parameter_count_basis_contradiction")
    if not _sha256(receipt.get("worker_source_sha256")):
        errors.append("invalid_worker_source_sha256")
    if type(receipt.get("worker_affective_steering_active")) is not bool:
        errors.append("invalid_worker_affective_steering_active")
    steering_alpha = receipt.get("worker_affective_steering_alpha")
    if (
        isinstance(steering_alpha, bool)
        or not isinstance(steering_alpha, (int, float))
        or not 0.0 <= float(steering_alpha) <= MAX_AFFECTIVE_STEERING_ALPHA
    ):
        errors.append("invalid_worker_affective_steering_alpha")
    if expected is not None:
        for key in (
            "worker_boot_id",
            "worker_pid",
            "worker_model_path",
            "worker_model_parameter_count",
            "worker_model_stored_parameter_element_count",
            "worker_model_parameter_count_basis",
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
    "MAX_AFFECTIVE_STEERING_ALPHA",
    "RUNTIME_IDENTITY_SCHEMA",
    "WORKER_IDENTITY_SCHEMA",
    "build_worker_identity",
    "canonical_model_path",
    "collect_latent_runtime_identity",
    "latent_request_payload_sha256",
    "logical_model_parameter_count",
    "model_parameter_count",
    "worker_identity_errors",
]
