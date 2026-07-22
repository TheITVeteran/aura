"""Identity receipts for resident Recursive Latent Cortex episodes."""

from __future__ import annotations

import hashlib
import json
import math
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
    operation_authority: Any = None,
    action_policy_evidence: Any = None,
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
    if operation_authority is not None:
        payload["operation_authority"] = operation_authority
    if action_policy_evidence is not None:
        payload["action_policy_evidence"] = action_policy_evidence
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
        # CP126 866530f6. Model path plus a parameter count cannot tell two
        # runtimes apart when the difference is WHICH adapter is attached,
        # which tokenizer resolved the text, or how the weights are
        # quantized. Two workers with identical path and count could be
        # serving materially different functions, so identity comparisons
        # and control/treatment claims built on this receipt were weaker
        # than they read.
        **_serving_stack_identity(model, model_path),
    }


def _serving_stack_identity(model: Any, model_path: str | Path) -> dict[str, Any]:
    """Identity of everything that changes what the model computes.

    Ordered adapter identity, tokenizer identity, and the quantization/dtype
    layout — each best-effort and each reporting its own absence rather than
    silently contributing nothing. A field that could not be determined is
    recorded as an empty value with a reason in
    ``worker_stack_identity_gaps``, so a consumer can see that identity is
    partial instead of assuming it is complete.
    """
    gaps: list[str] = []

    adapters = _attached_adapter_identity(model, gaps)
    tokenizer = _tokenizer_identity(model_path, gaps)
    quantization = _quantization_identity(model_path, gaps)

    return {
        "worker_adapters": adapters,
        "worker_adapter_stack_sha256": _digest_of_json(adapters),
        "worker_tokenizer": tokenizer,
        "worker_quantization": quantization,
        "worker_stack_identity_gaps": gaps,
    }


def _attached_adapter_identity(model: Any, gaps: list[str]) -> list[dict[str, Any]]:
    """Ordered identity of adapter-class modules resident on the model.

    Order matters: the same adapters applied in a different order can
    compose to a different function, so this is a list, not a set.
    """
    adapters: list[dict[str, Any]] = []
    try:
        named_modules = getattr(model, "named_modules", None)
        if not callable(named_modules):
            gaps.append("adapters:model_exposes_no_named_modules")
            return adapters
        for name, module in named_modules():
            type_name = type(module).__name__
            if "LoRA" not in type_name and "Adapter" not in type_name:
                continue
            adapters.append(
                {
                    "name": str(name),
                    "type": type_name,
                    "rank": _int_or_zero(getattr(module, "r", None)),
                    "scale": _float_or_zero(getattr(module, "scale", None)),
                }
            )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        gaps.append(f"adapters:{type(exc).__name__}")
    return adapters


def _tokenizer_identity(model_path: str | Path, gaps: list[str]) -> dict[str, Any]:
    """Digest of the tokenizer artifacts that turn text into the token ids."""
    identity: dict[str, Any] = {}
    try:
        root = Path(str(model_path))
        found = False
        for filename in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ):
            candidate = root / filename
            if candidate.is_file():
                identity[filename] = _stable_sha256(candidate, max_bytes=32 * 1024 * 1024)
                found = True
        if not found:
            gaps.append("tokenizer:no_tokenizer_artifacts_found")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        gaps.append(f"tokenizer:{type(exc).__name__}")
    return identity


def _quantization_identity(model_path: str | Path, gaps: list[str]) -> dict[str, Any]:
    """Quantization layout and dtype, which change the computed function."""
    identity: dict[str, Any] = {}
    try:
        config_path = Path(str(model_path)) / "config.json"
        if not config_path.is_file():
            gaps.append("quantization:no_config")
            return identity
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if not isinstance(config, dict):
            gaps.append("quantization:config_not_an_object")
            return identity
        quantization = config.get("quantization")
        if isinstance(quantization, dict):
            identity["bits"] = _int_or_zero(quantization.get("bits"))
            identity["group_size"] = _int_or_zero(quantization.get("group_size"))
        else:
            identity["bits"] = 0
            identity["group_size"] = 0
        identity["dtype"] = str(
            config.get("torch_dtype") or config.get("dtype") or ""
        )
        identity["model_type"] = str(config.get("model_type") or "")
        identity["config_sha256"] = _stable_sha256(config_path, max_bytes=4 * 1024 * 1024)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        gaps.append(f"quantization:{type(exc).__name__}")
    return identity


def _digest_of_json(value: Any) -> str:
    try:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
                "utf-8"
            )
        ).hexdigest()
    except (TypeError, ValueError):
        return ""


def _int_or_zero(value: Any) -> int:
    try:
        if isinstance(value, bool):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_or_zero(value: Any) -> float:
    try:
        if isinstance(value, bool):
            return 0.0
        result = float(value)
        return result if math.isfinite(result) else 0.0
    except (TypeError, ValueError):
        return 0.0


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


def _typed_issue_list(value: Any) -> list[str]:
    """Issue strings from an untrusted field, never a crash.

    A bare string is ONE issue, not a sequence of characters — iterating it
    was how a malformed provenance response turned into a wall of
    single-letter issues. Anything else non-iterable becomes a typed issue
    describing the shape problem itself.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Mapping):
        return [f"{key}:{val}" for key, val in value.items()]
    try:
        return [str(item) for item in value if str(item).strip()]
    except TypeError:
        return [f"provenance_issues_malformed:{type(value).__name__}"]


def _mapping_or_empty(value: Any) -> Mapping:
    """A mapping to read, or an empty one — never an AttributeError.

    A truthy non-mapping (a list, a string) previously reached .get and
    raised.
    """
    return value if isinstance(value, Mapping) else {}


def _nonnegative_int(value: Any) -> int:
    """Coerce an untrusted count, defaulting to 0 rather than raising."""
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed >= 0 else 0


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

    # CP126 79271a67. This collector reports on identity DEGRADATION, so it
    # is precisely the code that must not itself raise when the thing it is
    # inspecting is malformed. Three unguarded assumptions lived here: that
    # `issues` is an iterable collection (a bare string iterates into
    # characters; an int raises), that a truthy `expected` is a mapping
    # before calling .get on it, and that `source_change_count` converts
    # with a bare int(). Any of those turned "identity could not be
    # verified" into an exception on the caller.
    provenance = collect_runtime_launch_provenance(project_root, env=env)
    provenance = dict(provenance) if isinstance(provenance, Mapping) else {}
    required = provenance.get("required") is True
    source = provenance.get("actual") if required else collect_source_identity(project_root)
    source = dict(source) if isinstance(source, Mapping) else {}
    manifest = provenance.get("manifest")
    manifest = dict(manifest) if isinstance(manifest, Mapping) else {}
    issues = _typed_issue_list(provenance.get("issues"))

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
        "source_change_count": _nonnegative_int(source.get("source_change_count")),
        "shell_assets_sha256": shell_assets_sha256,
        "bundle_identifier": str(
            manifest.get("bundle_identifier")
            or _mapping_or_empty(provenance.get("expected")).get("bundle_identifier")
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
