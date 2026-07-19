#!/usr/bin/env python
"""Run a crash-resumable resident-32B RLC 2x2 attribution campaign.

The parent freezes a public task/arm plan, then launches one isolated model
process per arm. Every task outcome is committed immediately to a hash-chained
journal. Wrong answers are valid outcomes; only infrastructure failures retry.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import signal
import stat
import subprocess
import sys
import time
import uuid
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.adapter_identity import (  # noqa: E402
    build_legacy_v1_manifest,
    inspect_mlx_tensor_metadata,
    validate_adapter_identity,
)
from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    CampaignJournal,
    CampaignPlan,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.campaign_trust import (  # noqa: E402
    CAMPAIGN_RUNNER,
    TASK_ISSUER,
    externally_custodied_roles,
    prepare_role_signature_request,
    validate_campaign_trust_policy,
    verify_role_attestation,
)
from core.brain.llm.latent_cortex.exact_paired_grade import (  # noqa: E402
    exact_campaign_power_plan,
)
from core.brain.llm.latent_cortex.frontier_tasks import (  # noqa: E402
    CURRENT_REGISTRY_VERSION,
    FRONTIER_DOMAINS,
    REGISTRY_VERSION,
    FrontierTask,
    PublicTaskRecord,
    build_public_task_manifest,
    build_task_manifest,
    generate_task_battery,
    parse_final_answer,
)
from core.brain.llm.latent_cortex.paired_campaign import (  # noqa: E402
    ADAPTER_EQUAL_COMPUTE,
    ADAPTER_RLC,
    BASE_EQUAL_COMPUTE,
    BASE_RLC,
    FULL_ARMS,
    PRIMARY_ARMS,
    build_campaign_plan,
    grade_campaign,
)
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (  # noqa: E402
    MANIFEST_SCHEMA_V2,
    model_behavior_bundle_identity,
    personality_bundle_identity,
    runtime_environment_identity,
    strict_json_loads,
    validate_v2_adapter_identity,
)
from core.brain.llm.latent_cortex.runtime_identity import (  # noqa: E402
    build_worker_identity,
    logical_model_parameter_count,
)
from core.brain.llm.latent_cortex.worker_origin import (  # noqa: E402
    WORKER_KEY_CUSTODY_PRODUCER_SOFTWARE,
    ZERO_SHA256,
)
from core.brain.llm.latent_cortex.worker_origin_legacy import (  # noqa: E402
    build_legacy_worker_authorization_payload,
    build_legacy_worker_result_origin,
    verify_legacy_worker_authorization,
    verify_legacy_worker_result_origin,
)
from core.runtime.detached_subprocess_broker import (  # noqa: E402
    broker_available,
    run_brokered_process,
)

PLAN_FILE = "plan.json"
JOURNAL_FILE = "campaign.jsonl"
MANIFEST_FILE = "campaign_manifest.json"
GRADE_FILE = "grade.json"
LOG_FILE = "runner.log"
SEALED_OUTPUT_MANIFEST_FILE = "sealed_output_manifest.json"
ANSWER_REVEAL_REQUEST_FILE = "answer_reveal_request.json"
ANSWER_REVEAL_FILE = "answer_reveal.json"
FINAL_RUN_REQUEST_FILE = "final_run_request.json"
FINAL_RUN_ENVELOPE_FILE = "final_run_envelope.json"
WORKER_ORIGIN_DIR = "worker_origins"
WORKER_AUTHORIZATION_MANIFEST_FILE = "worker_authorization_manifest.json"
WORKER_LIFECYCLE_MANIFEST_FILE = "worker_lifecycle_manifest.json"
WORKER_KEY_ERASURE_MANIFEST_FILE = "worker_key_erasure_manifest.json"
OBJECTIVE_SOURCE = REPO_ROOT / "core/learning/recurrence_native_objective.py"
V2_MANIFEST_FILE = "recurrence_adapter_manifest.json"
CONTAMINATION_AUDIT_SCHEMA = "aura.latent_cortex.contamination_audit.v2"
TASK_ISSUER_PAYLOAD_SCHEMA = "aura.latent_cortex.task_issuer_prelaunch.v1"
CAMPAIGN_RUNNER_PAYLOAD_SCHEMA = "aura.latent_cortex.runner_prelaunch.v1"
SEALED_OUTPUT_MANIFEST_SCHEMA = "aura.latent_cortex.sealed_output_manifest.v3"
ANSWER_REVEAL_PAYLOAD_SCHEMA = "aura.latent_cortex.answer_reveal_payload.v1"
FINAL_RUN_PAYLOAD_SCHEMA = "aura.latent_cortex.final_run_payload.v3"
WORKER_AUTHORIZATION_MANIFEST_SCHEMA = (
    "aura.latent_cortex.worker_authorization_manifest.v1"
)
WORKER_LIFECYCLE_MANIFEST_SCHEMA = (
    "aura.latent_cortex.worker_lifecycle_manifest.v1"
)


class CampaignProducerError(RuntimeError):
    pass


@contextlib.contextmanager
def _deadline_alarm(seconds: float, stage: str):
    """Hard wall deadline for one worker stage on POSIX main threads."""

    def expired(_signum: int, _frame: Any) -> None:
        error = TimeoutError(f"{stage} exceeded {seconds:.3f}s hard deadline")
        raise error

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _csv_ints(parser: argparse.ArgumentParser, raw: str, role: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError:
        parser.error(f"{role} must be comma-separated integers")
    if (
        not values
        or len(set(values)) != len(values)
        or any(value < 0 for value in values)
    ):
        parser.error(f"{role} must contain unique non-negative integers")
    return values


def _csv_domains(parser: argparse.ArgumentParser, raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if (
        not values
        or len(set(values)) != len(values)
        or any(value not in FRONTIER_DOMAINS for value in values)
    ):
        parser.error("--domains must contain unique registered frontier domains")
    return values


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_stable_bytes(path: Path, *, max_bytes: int) -> bytes:
    if path.is_symlink():
        raise CampaignProducerError(f"symlink artifact rejected: {path}")
    before = path.stat()
    if not path.is_file() or before.st_size <= 0 or before.st_size > max_bytes:
        raise CampaignProducerError(f"artifact size is invalid: {path}")
    payload = path.read_bytes()
    after = path.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(payload) != before.st_size:
        raise CampaignProducerError(f"artifact changed while reading: {path}")
    return payload


def _fresh_checkpoint_file_fingerprint(model_path: Path) -> dict[str, Any]:
    """Full-hash every weight shard without the process fingerprint cache."""

    files = (
        sorted(model_path.glob("*.safetensors"))
        or sorted(model_path.glob("*.npz"))
        or sorted(model_path.glob("*.gguf"))
    )
    if not files:
        raise CampaignProducerError("model checkpoint has no weight files")
    combined = hashlib.sha256()
    for path in files:
        digest = hashlib.sha256()
        if path.is_symlink():
            raise CampaignProducerError(f"model weight symlink rejected: {path}")
        before = path.stat()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise CampaignProducerError(f"model weight changed while hashing: {path}")
        combined.update(f"{path.name}:{digest.hexdigest()};".encode())
    return {
        "fingerprint": combined.hexdigest(),
        "method": "sha256",
        "files": len(files),
    }


def _atomic_create_or_verify(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise CampaignProducerError(f"symlink output rejected: {path}")
    if path.exists():
        if _read_stable_bytes(path, max_bytes=max(1, len(payload))) != payload:
            raise CampaignProducerError(f"existing artifact differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(temporary, flags, 0o600)
    try:
        try:
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise CampaignProducerError(f"short write: {temporary}")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            if _read_stable_bytes(path, max_bytes=max(1, len(payload))) != payload:
                raise CampaignProducerError(
                    f"concurrent artifact differs: {path}"
                ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _secure_worker_origin_dir(campaign_dir: Path) -> Path:
    path = campaign_dir / WORKER_ORIGIN_DIR
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    observed = path.lstat()
    if (
        not stat.S_ISDIR(observed.st_mode)
        or path.is_symlink()
        or observed.st_uid != os.geteuid()
        or stat.S_IMODE(observed.st_mode) & 0o077
    ):
        raise CampaignProducerError("worker origin directory is not private")
    return path


def _worker_slot_stem(arm: str, attempt_slot: int) -> str:
    if arm not in FULL_ARMS:
        raise CampaignProducerError("worker authorization arm is invalid")
    if (
        isinstance(attempt_slot, bool)
        or not isinstance(attempt_slot, int)
        or attempt_slot <= 0
    ):
        raise CampaignProducerError("worker authorization attempt slot is invalid")
    return f"{arm}.attempt-{attempt_slot:02d}"


def _worker_origin_paths(
    campaign_dir: Path,
    arm: str,
    attempt_slot: int,
) -> dict[str, Path]:
    root = _secure_worker_origin_dir(campaign_dir)
    stem = _worker_slot_stem(arm, attempt_slot)
    return {
        "private_key": root / f".{stem}.private-key.raw",
        "request": root / f"{stem}.request.json",
        "attestation": root / f"{stem}.attestation.json",
        "launch": root / f"{stem}.launch.json",
        "exit": root / f"{stem}.exit.json",
        "erasure_intent": root / f"{stem}.erasure-intent.json",
        "erasure": root / f"{stem}.erasure.json",
    }


def _load_worker_private_key(path: Path) -> Any:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    if not path.exists():
        raise CampaignProducerError("worker private key is missing")
    observed = path.lstat()
    if (
        not stat.S_ISREG(observed.st_mode)
        or path.is_symlink()
        or observed.st_uid != os.geteuid()
        or stat.S_IMODE(observed.st_mode) & 0o077
        or observed.st_size != 32
    ):
        raise CampaignProducerError("worker private key storage is invalid")
    raw = _read_stable_bytes(path, max_bytes=32)
    try:
        return Ed25519PrivateKey.from_private_bytes(raw)
    except ValueError as exc:
        raise CampaignProducerError("worker private key is invalid") from exc


def _load_or_create_worker_private_key(path: Path) -> Any:
    if not path.exists():
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )

        private_key = Ed25519PrivateKey.generate()
        raw = private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        _atomic_create_or_verify(path, raw)
    return _load_worker_private_key(path)


def _worker_public_key_raw(private_key: Any) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _worker_boot_id(private_key: Any) -> str:
    return _sha256_bytes(_worker_public_key_raw(private_key) + b":worker-boot")[:32]


def _worker_authorization_payload(
    args: argparse.Namespace,
    plan: CampaignPlan,
    policy: Any,
    *,
    arm: str,
    attempt_slot: int,
    worker_public_key_raw: bytes,
    worker_boot_id: str,
) -> dict[str, Any]:
    metadata = plan.to_dict()["metadata"]
    execution = metadata.get("execution_config")
    implementation = (
        execution.get("implementation_sha256")
        if isinstance(execution, dict)
        else None
    )
    source_key = "tools/run_latent_cortex_paired_campaign.py"
    source_sha256 = (
        implementation.get(source_key)
        if isinstance(implementation, dict)
        else None
    )
    actual_source_sha256 = _sha256_bytes(Path(__file__).resolve().read_bytes())
    if source_sha256 != actual_source_sha256:
        raise CampaignProducerError(
            "worker source differs from the frozen plan implementation"
        )
    command = _worker_args(
        args,
        arm,
        worker_attempt_slot=attempt_slot,
        worker_boot_id=worker_boot_id,
    )
    return build_legacy_worker_authorization_payload(
        campaign_name=plan.campaign_name,
        policy_sha256=policy.policy_sha256,
        protocol_sha256=_campaign_protocol_sha256(),
        plan_sha256=plan.plan_sha256,
        arm=arm,
        worker_attempt_slot=attempt_slot,
        worker_boot_id=worker_boot_id,
        worker_key_custody=WORKER_KEY_CUSTODY_PRODUCER_SOFTWARE,
        worker_source_sha256=actual_source_sha256,
        worker_command=command,
        model_identity_sha256=_sha256_bytes(
            canonical_json_bytes(metadata["model_identity"])
        ),
        adapter_identity_sha256=_sha256_bytes(
            canonical_json_bytes(metadata["adapter_identity"])
        ),
        worker_public_key_raw=worker_public_key_raw,
    )


def _runtime_bundle_identity(
    model_path: Path,
    *,
    weight_identity: dict[str, Any],
) -> dict[str, Any]:
    behavior_files: list[dict[str, Any]] = []
    for path in sorted(model_path.iterdir(), key=lambda item: item.name):
        if (
            not path.is_file()
            or path.name == "README.md"
            or path.suffix == ".safetensors"
        ):
            continue
        if path.is_symlink():
            raise CampaignProducerError(f"model bundle symlink rejected: {path}")
        payload = _read_stable_bytes(path, max_bytes=512 * 1024 * 1024)
        behavior_files.append(
            {
                "path": path.name,
                "size_bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        )
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    if not required.issubset(record["path"] for record in behavior_files):
        raise CampaignProducerError("model runtime bundle is incomplete")
    try:
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignProducerError("model config is unreadable") from exc
    if not isinstance(config, dict):
        raise CampaignProducerError("model config is invalid")
    logical_count, count_basis = logical_model_parameter_count(
        model_path,
        stored_element_count=0,
    )
    dependencies: dict[str, str] = {}
    for distribution in ("mlx", "mlx-lm", "numpy"):
        try:
            dependencies[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise CampaignProducerError(
                f"required runtime distribution is missing: {distribution}"
            ) from exc
    body = {
        "schema": "aura.latent_cortex.model_runtime_bundle.v1",
        "weight_fingerprint": weight_identity["fingerprint"],
        "weight_file_count": weight_identity["files"],
        "behavior_files": behavior_files,
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "logical_parameter_count": logical_count,
        "logical_parameter_count_basis": count_basis,
        "dependencies": dependencies,
        "python": platform.python_version(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
    }
    return {**body, "bundle_sha256": _sha256_bytes(canonical_json_bytes(body))}


def _contained_adapter_artifact(adapter_dir: Path, relative: Any) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or relative.startswith("/")
        or "\\" in relative
        or "\x00" in relative
    ):
        raise CampaignProducerError("adapter artifact path is invalid")
    root = adapter_dir.resolve(strict=True)
    cursor = root
    for part in Path(relative).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise CampaignProducerError("adapter artifact symlink is rejected")
    try:
        resolved = (root / relative).resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise CampaignProducerError("adapter artifact is missing") from exc
    if not resolved.is_relative_to(root) or resolved == root:
        raise CampaignProducerError("adapter artifact escapes bundle root")
    return resolved


def _v2_manifest_bytes(adapter_dir: Path) -> bytes | None:
    path = adapter_dir / V2_MANIFEST_FILE
    if not path.exists():
        return None
    return _read_stable_bytes(path, max_bytes=16 * 1024 * 1024)


def _v2_artifacts(adapter_dir: Path, manifest: dict[str, Any]) -> dict[str, bytes]:
    artifacts: dict[str, bytes] = {}
    bindings: list[tuple[str, Mapping[str, Any]]] = []
    for role in (
        "adapter",
        "adapter_alias",
        "loader_config",
        "training_receipt",
        "training_config",
        "dataset_manifest",
        "execution_spec",
    ):
        value = manifest.get(role)
        if not isinstance(value, dict):
            raise CampaignProducerError(f"v2 {role} binding is invalid")
        bindings.append((role, value))
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise CampaignProducerError("v2 source bindings are invalid")
    for role, value in sources.items():
        if not isinstance(role, str) or not isinstance(value, dict):
            raise CampaignProducerError("v2 source binding is invalid")
        bindings.append(
            (
                f"source_{role}",
                {
                    "path": value.get("snapshot_path"),
                    "size_bytes": value.get("size_bytes"),
                },
            )
        )
    for role, binding in bindings:
        relative = binding.get("path")
        size = binding.get("size_bytes")
        if type(size) is not int or size <= 0:
            raise CampaignProducerError(f"v2 {role} size is invalid")
        path = _contained_adapter_artifact(adapter_dir, relative)
        if relative in artifacts:
            raise CampaignProducerError("v2 artifact path is duplicated")
        artifacts[str(relative)] = _read_stable_bytes(path, max_bytes=size)
    completion = _contained_adapter_artifact(adapter_dir, "training_completion.json")
    artifacts["training_completion.json"] = _read_stable_bytes(
        completion, max_bytes=1024 * 1024
    )
    return artifacts


def _v2_training_config(
    adapter_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    binding = manifest.get("training_config")
    if not isinstance(binding, dict) or type(binding.get("size_bytes")) is not int:
        raise CampaignProducerError("v2 training config binding is invalid")
    path = _contained_adapter_artifact(adapter_dir, binding.get("path"))
    payload = _read_stable_bytes(path, max_bytes=int(binding["size_bytes"]))
    return strict_json_loads(payload, role="campaign_training_config")


def _resolve_campaign_personality(
    args: argparse.Namespace,
    *,
    model_path: Path,
    adapter_dir: Path,
    v2_manifest: dict[str, Any] | None,
) -> str | None:
    requested = str(
        getattr(args, "personality_adapter", "trained") or "trained"
    ).strip()
    lowered = requested.lower()
    if lowered == "trained":
        if v2_manifest is None:
            return None
        configured = str(
            _v2_training_config(adapter_dir, v2_manifest).get(
                "personality_adapter_path", ""
            )
        ).strip()
        requested = configured or "none"
        lowered = requested.lower()
    if lowered == "none":
        return None
    if lowered == "auto":
        from core.brain.llm.model_registry import resolve_personality_adapter

        resolved = resolve_personality_adapter(str(model_path), backend="mlx")
        return (
            str(Path(resolved).expanduser().resolve(strict=True)) if resolved else None
        )
    resolved = Path(requested).expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise CampaignProducerError("personality adapter must be a directory")
    return str(resolved)


def _effective_stack_sha256(
    *,
    weight_fingerprint: str,
    runtime_bundle_sha256: str,
    personality_identity: Mapping[str, Any],
) -> str:
    return _sha256_bytes(
        canonical_json_bytes(
            {
                "schema": "aura.latent_cortex.effective_model_stack.v1",
                "weight_fingerprint": weight_fingerprint,
                "runtime_bundle_sha256": runtime_bundle_sha256,
                "personality_adapter": dict(personality_identity),
            }
        )
    )


def _validate_v2_adapter_dir(
    adapter_dir: Path,
    manifest_bytes: bytes,
    *,
    adapter_id: str,
    base_checkpoint: Mapping[str, Any],
    model_behavior_bundle: Mapping[str, Any],
    personality_identity: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = strict_json_loads(manifest_bytes, role="campaign_v2_manifest")
    artifacts = _v2_artifacts(adapter_dir, manifest)
    adapter_binding = manifest.get("adapter")
    if not isinstance(adapter_binding, dict):
        raise CampaignProducerError("v2 adapter binding is invalid")
    adapter_path = _contained_adapter_artifact(adapter_dir, adapter_binding.get("path"))
    receipt = validate_v2_adapter_identity(
        manifest_bytes,
        adapter_id=adapter_id,
        actual_base_checkpoint=base_checkpoint,
        actual_model_behavior_bundle=model_behavior_bundle,
        actual_personality_adapter=personality_identity,
        actual_runtime_environment=runtime_environment,
        artifacts=artifacts,
        tensor_metadata=inspect_mlx_tensor_metadata(adapter_path),
    )
    return manifest, receipt


def _identity_material(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_path = Path(args.model).expanduser().resolve(strict=True)
    adapter_dir = Path(args.adapter).expanduser().resolve(strict=True)
    weight_identity = _fresh_checkpoint_file_fingerprint(model_path)
    model_behavior_identity = model_behavior_bundle_identity(model_path)
    runtime_environment = runtime_environment_identity()
    runtime_bundle = _runtime_bundle_identity(
        model_path,
        weight_identity=weight_identity,
    )
    v2_manifest_bytes = _v2_manifest_bytes(adapter_dir)
    v2_manifest = (
        strict_json_loads(v2_manifest_bytes, role="campaign_v2_manifest")
        if v2_manifest_bytes is not None
        else None
    )
    personality_path = _resolve_campaign_personality(
        args,
        model_path=model_path,
        adapter_dir=adapter_dir,
        v2_manifest=v2_manifest,
    )
    personality_identity = personality_bundle_identity(personality_path)
    model_identity = {
        "model_path": str(model_path),
        **weight_identity,
        "runtime_bundle": runtime_bundle,
        "model_behavior_bundle": model_behavior_identity,
        "runtime_environment": runtime_environment,
        "personality_adapter_path": personality_path or "",
        "personality_adapter": personality_identity,
        "effective_stack_sha256": _effective_stack_sha256(
            weight_fingerprint=weight_identity["fingerprint"],
            runtime_bundle_sha256=runtime_bundle["bundle_sha256"],
            personality_identity=personality_identity,
        ),
    }
    if v2_manifest_bytes is not None:
        manifest, receipt = _validate_v2_adapter_dir(
            adapter_dir,
            v2_manifest_bytes,
            adapter_id=args.adapter_id,
            base_checkpoint=weight_identity,
            model_behavior_bundle=model_behavior_identity,
            personality_identity=personality_identity,
            runtime_environment=runtime_environment,
        )
        execution_binding = manifest["execution_spec"]
        execution_payload = _read_stable_bytes(
            _contained_adapter_artifact(adapter_dir, execution_binding["path"]),
            max_bytes=int(execution_binding["size_bytes"]),
        )
        adapter_identity = {
            "adapter_dir": str(adapter_dir),
            "format": MANIFEST_SCHEMA_V2,
            "manifest": manifest,
            "identity_receipt": receipt,
            "execution_spec": strict_json_loads(
                execution_payload, role="campaign_execution_spec"
            ),
        }
        return model_identity, adapter_identity
    if personality_path is not None:
        raise CampaignProducerError(
            "legacy recurrence adapter cannot be composed with an unbound personality adapter"
        )
    manifest = build_legacy_v1_manifest(
        adapter_dir,
        adapter_id=args.adapter_id,
        actual_base_checkpoint_fingerprint=model_identity["fingerprint"],
        objective_source_path=OBJECTIVE_SOURCE,
    )
    identity = validate_adapter_identity(
        manifest,
        actual_base_checkpoint_fingerprint=model_identity["fingerprint"],
        adapter_bytes=(adapter_dir / manifest.adapter.path).read_bytes(),
        training_receipt_bytes=(
            adapter_dir / manifest.training_receipt.path
        ).read_bytes(),
        tensor_metadata=inspect_mlx_tensor_metadata(
            adapter_dir / manifest.adapter.path
        ),
    )
    adapter_identity = {
        "adapter_dir": str(adapter_dir),
        "format": manifest.schema,
        "manifest": manifest.to_dict(),
        "identity_receipt": identity.to_dict(),
        "execution_spec": None,
    }
    return model_identity, adapter_identity


def _model_load_boundary_identity(
    model_path: Path,
    personality_path: str | None,
) -> dict[str, Any]:
    weight_identity = _fresh_checkpoint_file_fingerprint(model_path)
    runtime_bundle = _runtime_bundle_identity(
        model_path,
        weight_identity=weight_identity,
    )
    personality_identity = personality_bundle_identity(personality_path)
    model_behavior_identity = model_behavior_bundle_identity(model_path)
    runtime_environment = runtime_environment_identity()
    return {
        "weight_fingerprint": weight_identity["fingerprint"],
        "weight_method": weight_identity["method"],
        "weight_file_count": weight_identity["files"],
        "runtime_bundle_sha256": runtime_bundle["bundle_sha256"],
        "model_behavior_bundle": model_behavior_identity,
        "runtime_environment": runtime_environment,
        "personality_adapter": personality_identity,
        "effective_stack_sha256": _effective_stack_sha256(
            weight_fingerprint=weight_identity["fingerprint"],
            runtime_bundle_sha256=runtime_bundle["bundle_sha256"],
            personality_identity=personality_identity,
        ),
    }


def _adapter_load_boundary_identity(
    adapter_dir: Path,
    manifest: dict[str, Any],
    *,
    adapter_id: str,
    base_checkpoint: Mapping[str, Any],
    model_behavior_bundle: Mapping[str, Any],
    personality_identity: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
) -> dict[str, Any]:
    if manifest.get("schema") == MANIFEST_SCHEMA_V2:
        manifest_bytes = _read_stable_bytes(
            adapter_dir / V2_MANIFEST_FILE,
            max_bytes=16 * 1024 * 1024,
        )
        _parsed, receipt = _validate_v2_adapter_dir(
            adapter_dir,
            manifest_bytes,
            adapter_id=adapter_id,
            base_checkpoint=base_checkpoint,
            model_behavior_bundle=model_behavior_bundle,
            personality_identity=personality_identity,
            runtime_environment=runtime_environment,
        )
        if _parsed != manifest:
            raise CampaignProducerError("v2 adapter manifest differs from frozen plan")
        return receipt
    adapter_binding = manifest["adapter"]
    receipt_binding = manifest["training_receipt"]
    adapter_path = adapter_dir / adapter_binding["path"]
    receipt_path = adapter_dir / receipt_binding["path"]
    receipt = validate_adapter_identity(
        manifest,
        actual_base_checkpoint_fingerprint=str(base_checkpoint["fingerprint"]),
        adapter_bytes=_read_stable_bytes(
            adapter_path,
            max_bytes=max(1, int(adapter_binding["size_bytes"])),
        ),
        training_receipt_bytes=_read_stable_bytes(
            receipt_path,
            max_bytes=max(1, int(receipt_binding["size_bytes"])),
        ),
        tensor_metadata=inspect_mlx_tensor_metadata(adapter_path),
    )
    return receipt.to_dict()


def _tasks(args: argparse.Namespace) -> tuple[FrontierTask, ...]:
    return generate_task_battery(
        args.seed_values,
        domains=args.domain_values,
        difficulty=args.difficulty,
        registry_version=getattr(args, "task_registry_version", REGISTRY_VERSION),
    )


def _public_tasks_from_plan(plan: CampaignPlan) -> tuple[PublicTaskRecord, ...]:
    """Load only candidate-visible task records from a hash-validated plan."""

    metadata = plan.to_dict().get("metadata")
    manifest = metadata.get("task_manifest") if isinstance(metadata, dict) else None
    raw_tasks = manifest.get("tasks") if isinstance(manifest, dict) else None
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise CampaignProducerError("persisted plan has no public task manifest")
    try:
        tasks = tuple(PublicTaskRecord.from_dict(raw) for raw in raw_tasks)
        rebuilt = build_public_task_manifest(tasks).to_dict()
    except (TypeError, ValueError) as exc:
        raise CampaignProducerError("persisted public task manifest is invalid") from exc
    if rebuilt != manifest:
        raise CampaignProducerError("persisted public task manifest hash mismatch")
    task_ids = {task.task_id for task in tasks}
    if any(
        plan.cell_definition(cell_id).get("task_id") not in task_ids
        for cell_id in plan.cell_ids
    ):
        raise CampaignProducerError("persisted plan cell references an unknown task")
    return tasks


def _manifest_for_tasks(
    tasks: tuple[FrontierTask, ...] | tuple[PublicTaskRecord, ...],
):
    if all(isinstance(task, FrontierTask) for task in tasks):
        return build_task_manifest(tasks)
    if all(isinstance(task, PublicTaskRecord) for task in tasks):
        return build_public_task_manifest(tasks)
    raise CampaignProducerError("campaign task types are inconsistent")


def _load_contamination_trust_root(path_value: str) -> tuple[Any, bytes, str]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    trust_path = Path(path_value).expanduser().resolve(strict=True)
    trust_bytes = _read_stable_bytes(trust_path, max_bytes=64 * 1024)
    try:
        public_key = serialization.load_pem_public_key(trust_bytes)
    except ValueError as exc:
        raise CampaignProducerError(
            "contamination audit trust root is invalid"
        ) from exc
    if not isinstance(public_key, Ed25519PublicKey):
        raise CampaignProducerError("contamination audit trust root is not Ed25519")
    public_der = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return public_key, public_der, _sha256_bytes(public_der)


def _adapter_dataset_manifest_sha256(
    adapter_identity: Mapping[str, Any],
) -> str | None:
    manifest = adapter_identity.get("manifest")
    if not isinstance(manifest, Mapping):
        return None
    binding = manifest.get("dataset_manifest")
    if not isinstance(binding, Mapping):
        return None
    digest = binding.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        return None
    return digest


def _contamination_audit(
    args: argparse.Namespace,
    tasks: tuple[FrontierTask, ...] | tuple[PublicTaskRecord, ...],
    *,
    expected_training_corpus_sha256: str | None = None,
) -> dict[str, Any]:
    raw_path = str(getattr(args, "contamination_audit", "") or "").strip()
    if not raw_path:
        return {}
    path = Path(raw_path).expanduser().resolve(strict=True)
    if path.is_symlink() or path.stat().st_size > 8 * 1024 * 1024:
        raise CampaignProducerError("contamination audit artifact is unsafe")
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignProducerError("contamination audit artifact is invalid") from exc
    trust_root_path = str(getattr(args, "contamination_trust_root", "") or "").strip()
    if not trust_root_path:
        raise CampaignProducerError("contamination audit trust root is required")
    required = {
        "schema",
        "task_manifest_sha256",
        "status",
        "overlap_count",
        "auditor_independence",
        "corpora",
        "methods",
        "signature",
    }
    if not isinstance(audit, dict) or set(audit) != required:
        raise CampaignProducerError("contamination audit schema is invalid")
    manifest = _manifest_for_tasks(tasks)
    body = dict(audit)
    signature = body.pop("signature")
    methods = audit["methods"]
    corpora = audit["corpora"]
    required_methods = {"exact_prompt", "normalized_prompt", "token_fivegram"}
    if (
        audit["schema"] != CONTAMINATION_AUDIT_SCHEMA
        or audit["task_manifest_sha256"] != manifest.manifest_sha256
        or audit["status"] != "passed_zero_overlap"
        or audit["overlap_count"] != 0
        or audit["auditor_independence"] != "external"
        or not isinstance(methods, list)
        or not required_methods.issubset(methods)
        or not isinstance(corpora, list)
        or not corpora
        or any(
            not isinstance(record, dict)
            or set(record) != {"name", "snapshot_sha256"}
            or not isinstance(record["name"], str)
            or len(str(record["snapshot_sha256"])) != 64
            for record in corpora
        )
    ):
        raise CampaignProducerError("contamination audit verification failed")
    corpus_hashes = {
        record["snapshot_sha256"]
        for record in corpora
        if isinstance(record, dict)
    }
    if (
        expected_training_corpus_sha256 is not None
        and expected_training_corpus_sha256 not in corpus_hashes
    ):
        raise CampaignProducerError(
            "contamination audit does not cover the adapter training corpus"
        )
    if (
        not isinstance(signature, dict)
        or set(signature) != {"algorithm", "key_id", "signature_b64"}
        or signature.get("algorithm") != "ed25519"
    ):
        raise CampaignProducerError("contamination audit signature is invalid")
    try:
        signature_bytes = base64.b64decode(
            str(signature["signature_b64"]), validate=True
        )
    except (KeyError, ValueError) as exc:
        raise CampaignProducerError("contamination audit signature is invalid") from exc
    from cryptography.exceptions import InvalidSignature

    public_key, public_der, trust_root_sha256 = _load_contamination_trust_root(
        trust_root_path
    )
    if signature.get("key_id") != trust_root_sha256:
        raise CampaignProducerError(
            "contamination audit signer does not match trust root"
        )
    signed_payload = canonical_json_bytes(body)
    try:
        public_key.verify(signature_bytes, signed_payload)
    except InvalidSignature as exc:
        raise CampaignProducerError(
            "contamination audit signature verification failed"
        ) from exc
    return {
        **body,
        "signature": {
            "algorithm": "ed25519",
            "key_id": trust_root_sha256,
            "signature_b64": signature["signature_b64"],
            "signed_payload_sha256": _sha256_bytes(signed_payload),
            "public_key_der_b64": base64.b64encode(public_der).decode("ascii"),
            "trust_root_sha256": trust_root_sha256,
            "verified": True,
        },
    }


def _arms(args: argparse.Namespace) -> tuple[str, ...]:
    return FULL_ARMS if args.profile == "full" else PRIMARY_ARMS


def _implementation_sha256() -> dict[str, str]:
    latent_cortex_root = REPO_ROOT / "core/brain/llm/latent_cortex"
    implementation_paths = (
        Path(__file__).resolve(),
        *sorted(latent_cortex_root.glob("*.py")),
    )
    return {
        str(path.relative_to(REPO_ROOT)): _sha256_bytes(path.read_bytes())
        for path in implementation_paths
    }


def _campaign_protocol_sha256() -> str:
    return _sha256_bytes(canonical_json_bytes(_implementation_sha256()))


def _prelaunch_role_implementation_sha256(role: str) -> str:
    paths = {
        TASK_ISSUER: REPO_ROOT
        / "core/brain/llm/latent_cortex/frontier_tasks.py",
        CAMPAIGN_RUNNER: Path(__file__).resolve(),
        "contamination_auditor": REPO_ROOT / "tools/produce_contamination_audit.py",
    }
    path = paths.get(role)
    if path is None:
        raise CampaignProducerError("unsupported prelaunch trust role")
    return _sha256_bytes(_read_stable_bytes(path, max_bytes=16 * 1024 * 1024))


def _read_json_artifact(path_value: str, *, role: str) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve(strict=True)
    payload = _read_stable_bytes(path, max_bytes=16 * 1024 * 1024)
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignProducerError(f"{role} is not valid JSON") from exc
    if not isinstance(document, dict):
        raise CampaignProducerError(f"{role} must be a JSON object")
    return document


def _load_campaign_trust_policy(
    args: argparse.Namespace,
    *,
    require_current: bool,
) -> Any | None:
    policy_path = str(getattr(args, "campaign_trust_policy", "") or "").strip()
    root_path = str(getattr(args, "campaign_trust_root", "") or "").strip()
    if not policy_path and not root_path:
        return None
    if not policy_path or not root_path:
        raise CampaignProducerError(
            "campaign trust policy and independent root are both required"
        )
    policy = _read_json_artifact(policy_path, role="campaign trust policy")
    root_bytes = _read_stable_bytes(
        Path(root_path).expanduser().resolve(strict=True),
        max_bytes=64 * 1024,
    )
    return validate_campaign_trust_policy(
        policy,
        trusted_root_public_key_pem=root_bytes,
        expected_campaign_name=args.campaign_name,
        expected_protocol_sha256=_campaign_protocol_sha256(),
        now_unix=int(time.time()) if require_current else None,
    )


def _prelaunch_payloads(
    args: argparse.Namespace,
    *,
    unsigned_plan: CampaignPlan,
    policy: Any,
) -> dict[str, dict[str, Any]]:
    metadata = unsigned_plan.to_dict()["metadata"]
    task_manifest = metadata["task_manifest"]
    task_commitment = metadata["task_commitment"]
    execution_config = metadata["execution_config"]
    generation_config = {
        "difficulty": execution_config["difficulty"],
        "domains": execution_config["domains"],
        "task_registry_version": execution_config["task_registry_version"],
    }
    if "generation_seeds" in execution_config:
        generation_config["generation_seeds"] = execution_config[
            "generation_seeds"
        ]
    else:
        generation_config["generation_seed_count"] = execution_config[
            "generation_seed_count"
        ]
        generation_config["generation_seed_min_entropy_bits"] = execution_config[
            "generation_seed_min_entropy_bits"
        ]
        generation_config["generation_seed_policy"] = execution_config[
            "generation_seed_policy"
        ]
        generation_config["generation_seed_disclosure"] = execution_config[
            "generation_seed_disclosure"
        ]
    return {
        TASK_ISSUER: {
            "schema": TASK_ISSUER_PAYLOAD_SCHEMA,
            "campaign_name": args.campaign_name,
            "policy_sha256": policy.policy_sha256,
            "unsigned_plan_sha256": unsigned_plan.plan_sha256,
            "task_manifest_sha256": task_manifest["manifest_sha256"],
            "task_commitment_sha256": task_commitment["commitment_sha256"],
            "generation_config_sha256": _sha256_bytes(
                canonical_json_bytes(generation_config)
            ),
        },
        CAMPAIGN_RUNNER: {
            "schema": CAMPAIGN_RUNNER_PAYLOAD_SCHEMA,
            "campaign_name": args.campaign_name,
            "policy_sha256": policy.policy_sha256,
            "protocol_sha256": _campaign_protocol_sha256(),
            "unsigned_plan_sha256": unsigned_plan.plan_sha256,
            "model_identity_sha256": _sha256_bytes(
                canonical_json_bytes(metadata["model_identity"])
            ),
            "adapter_identity_sha256": _sha256_bytes(
                canonical_json_bytes(metadata["adapter_identity"])
            ),
            "execution_config_sha256": _sha256_bytes(
                canonical_json_bytes(execution_config)
            ),
            "contamination_audit_sha256": _sha256_bytes(
                canonical_json_bytes(metadata["contamination_audit"])
            ),
            "arms": metadata["arms"],
            "cell_count": len(unsigned_plan.cell_ids),
        },
    }


def _policy_auditor_matches(
    policy: Any,
    contamination_audit: Mapping[str, Any],
) -> bool:
    signature = contamination_audit.get("signature")
    if not isinstance(signature, Mapping):
        return False
    public_der_b64 = signature.get("public_key_der_b64")
    if not isinstance(public_der_b64, str):
        return False
    try:
        public_der = base64.b64decode(public_der_b64, validate=True)
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        public_key = serialization.load_der_public_key(public_der)
        if not isinstance(public_key, Ed25519PublicKey):
            return False
        public_raw = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    except (TypeError, ValueError):
        return False
    return (
        base64.b64encode(public_raw).decode("ascii")
        == policy.role_pin("contamination_auditor")["public_key_b64"]
    )


def _verified_campaign_trust(
    args: argparse.Namespace,
    *,
    unsigned_plan: CampaignPlan,
    contamination_audit: Mapping[str, Any],
) -> dict[str, Any] | None:
    policy = _load_campaign_trust_policy(args, require_current=True)
    if policy is None:
        return None
    if not _policy_auditor_matches(policy, contamination_audit):
        raise CampaignProducerError(
            "contamination auditor does not match the pre-pinned campaign role"
        )
    for role in (TASK_ISSUER, CAMPAIGN_RUNNER, "contamination_auditor"):
        if (
            policy.role_pin(role)["implementation_sha256"]
            != _prelaunch_role_implementation_sha256(role)
        ):
            raise CampaignProducerError(
                f"{role} implementation does not match the pre-pinned source"
            )
    payloads = _prelaunch_payloads(args, unsigned_plan=unsigned_plan, policy=policy)
    issuer_path = str(
        getattr(args, "task_issuer_attestation", "") or ""
    ).strip()
    runner_path = str(getattr(args, "runner_attestation", "") or "").strip()
    if not issuer_path or not runner_path:
        raise CampaignProducerError(
            "task issuer and campaign runner prelaunch attestations are required"
        )
    admitted_at = int(time.time())
    issuer_attestation = _read_json_artifact(
        issuer_path, role="task issuer attestation"
    )
    runner_attestation = _read_json_artifact(
        runner_path, role="campaign runner attestation"
    )
    verify_role_attestation(
        policy,
        issuer_attestation,
        role=TASK_ISSUER,
        expected_payload=payloads[TASK_ISSUER],
        not_after_unix=admitted_at,
    )
    verify_role_attestation(
        policy,
        runner_attestation,
        role=CAMPAIGN_RUNNER,
        expected_payload=payloads[CAMPAIGN_RUNNER],
        not_after_unix=admitted_at,
    )
    return {
        "schema": "aura.latent_cortex.campaign_prelaunch_trust.v1",
        "policy": policy.document,
        "policy_sha256": policy.policy_sha256,
        "root_key_id": policy.root_key_id,
        "protocol_sha256": _campaign_protocol_sha256(),
        "unsigned_plan_sha256": unsigned_plan.plan_sha256,
        "task_issuer_attestation": issuer_attestation,
        "task_issuer_payload_sha256": _sha256_bytes(
            canonical_json_bytes(payloads[TASK_ISSUER])
        ),
        "runner_attestation": runner_attestation,
        "runner_payload_sha256": _sha256_bytes(
            canonical_json_bytes(payloads[CAMPAIGN_RUNNER])
        ),
        "prelaunch_verified": True,
        "externally_custodied": externally_custodied_roles(policy),
    }


def _build_rlc_config(
    args: argparse.Namespace,
    execution_spec: Mapping[str, Any] | None = None,
) -> Any:
    from core.brain.llm.latent_cortex.types import (
        BranchConfig,
        CortexConfig,
        FastWeightsConfig,
        LatentOptConfig,
        RecurrenceConfig,
        WorkspaceConfig,
    )

    if execution_spec is not None:
        from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec

        spec = RLCExecutionSpec.from_dict(execution_spec)
        return CortexConfig(
            workspace=WorkspaceConfig(
                n_slots=spec.n_slots,
                seed=spec.slot_seed,
                roles=spec.slot_roles,
                anchor_scale=spec.anchor_scale,
            ),
            recurrence=RecurrenceConfig(
                max_steps=spec.recurrent_steps,
                min_steps=spec.recurrent_steps,
                alpha=spec.alpha,
                alpha_schedule=spec.alpha_schedule,
                rms_clip_ratio=spec.rms_clip_ratio,
                fixed_depth=not spec.adaptive_halting,
            ),
            branches=BranchConfig(
                n_branches=len(spec.branch_roles),
                exchange_interval=spec.exchange_interval,
                exchange_gamma=spec.exchange_gamma,
                comm_slot=spec.comm_slot,
                collapse_cos_threshold=spec.collapse_cos_threshold,
                jitter_scale=spec.jitter_scale,
                roles=spec.branch_roles,
            ),
            prelude_frac=spec.prelude_frac,
            coda_frac=spec.coda_frac,
            decode_max_tokens=args.decode_max_tokens,
            decode_bridge_policy=(
                "assistant_answer_v3"
                if spec.decode_bridge_policy == "assistant_answer"
                else "none"
            ),
            decode_repetition_penalty=1.25,
            decode_repetition_window=72,
            allow_vanilla_fallback=False,
            escape={"enabled": False},
        )
    if args.rlc_profile == "resident_full_stack":
        return CortexConfig(
            workspace=WorkspaceConfig(n_slots=4, seed=0),
            recurrence=RecurrenceConfig(max_steps=2, min_steps=2),
            branches=BranchConfig(n_branches=2, exchange_interval=1),
            latent_opt=LatentOptConfig(enabled=True, steps=1, lr=0.03),
            fast_weights=FastWeightsConfig(
                enabled=True,
                rank=2,
                opt_steps=1,
                lr=0.005,
                max_wrapped_layers=2,
                export_candidates=False,
            ),
            decode_max_tokens=args.decode_max_tokens,
            decode_min_tokens=min(96, max(0, args.decode_max_tokens - 1)),
            verifier_probe_max_tokens=24,
            verifier_accept_non_regression=True,
            decode_bridge_policy="assistant_answer_v3",
            decode_repetition_penalty=1.25,
            decode_repetition_window=72,
            allow_vanilla_fallback=False,
        )
    return CortexConfig(
        workspace=WorkspaceConfig(n_slots=args.n_slots, seed=0),
        recurrence=RecurrenceConfig(max_steps=args.rlc_steps, min_steps=2),
        branches=BranchConfig(n_branches=args.branches),
        decode_max_tokens=args.decode_max_tokens,
        decode_repetition_penalty=1.25,
        decode_repetition_window=72,
        allow_vanilla_fallback=False,
    )


def _execution_config(
    args: argparse.Namespace,
    adapter_identity: Mapping[str, Any],
) -> dict[str, Any]:
    execution_spec = adapter_identity.get("execution_spec")
    if execution_spec is not None and not isinstance(execution_spec, Mapping):
        raise CampaignProducerError("adapter execution spec is invalid")
    effective = _build_rlc_config(args, execution_spec)
    return {
        "profile": args.profile,
        "difficulty": args.difficulty,
        "task_registry_version": getattr(
            args,
            "task_registry_version",
            REGISTRY_VERSION,
        ),
        "generation_seed_count": int(
            getattr(args, "seed_count", 0) or len(args.seed_values)
        ),
        "generation_seed_min_entropy_bits": int(
            getattr(args, "seed_entropy_bits", 0)
            or min(value.bit_length() for value in args.seed_values)
        ),
        "generation_seed_policy": "external_issuer_uniform_63bit",
        "generation_seed_disclosure": "post_seal_answer_reveal",
        "domains": list(args.domain_values),
        "n_slots": effective.workspace.n_slots,
        "branches": effective.branches.n_branches,
        "rlc_steps": effective.recurrence.max_steps,
        "requested_rlc_shape": {
            "n_slots": args.n_slots,
            "branches": args.branches,
            "rlc_steps": args.rlc_steps,
        },
        "effective_rlc_config": asdict(effective),
        "adapter_execution_spec": (
            dict(execution_spec) if isinstance(execution_spec, Mapping) else None
        ),
        "rlc_profile": args.rlc_profile,
        "decode_max_tokens": args.decode_max_tokens,
        "episode_timeout_s": args.episode_timeout,
        "load_timeout_s": args.load_timeout,
        "warmup_timeout_s": args.warmup_timeout,
        "arm_timeout_s": args.arm_timeout,
        "campaign_timeout_s": args.campaign_timeout,
        "equal_compute_max_samples": args.equal_compute_max_samples,
        "adapter_process_isolation": True,
        "worker_task_material": "public_manifest_only",
        "answer_reveal_protocol": "sealed_outputs_then_issuer_reveal_v1",
        "worker_origin_protocol": "preauthorized_ephemeral_chain_v2",
        "worker_origin_attempt_slots": args.max_infra_attempts,
        "vanilla_fallback_allowed": False,
        "exact_statistical_power": _statistical_power_plan(args),
        "implementation_sha256": _implementation_sha256(),
    }


def _statistical_power_plan(args: argparse.Namespace) -> dict[str, Any]:
    arms = _arms(args)
    comparison_count = 4
    if BASE_EQUAL_COMPUTE in arms:
        comparison_count += 1
    if ADAPTER_EQUAL_COMPUTE in arms:
        comparison_count += 1
    planned = int(
        getattr(args, "seed_count", 0) or len(args.seed_values)
    )
    return exact_campaign_power_plan(
        domain_count=len(args.domain_values),
        comparison_count=comparison_count,
        arm_count=len(arms),
        planned_observations_per_domain=planned,
    )


def _expected_plan(
    args: argparse.Namespace,
) -> tuple[CampaignPlan, tuple[FrontierTask, ...]]:
    tasks = _tasks(args)
    model_identity, adapter_identity = _identity_material(args)
    training_corpus_sha256 = _adapter_dataset_manifest_sha256(adapter_identity)
    contamination_audit = _contamination_audit(
        args,
        tasks,
        expected_training_corpus_sha256=training_corpus_sha256,
    )
    execution_config = _execution_config(args, adapter_identity)
    unsigned_plan = build_campaign_plan(
        args.campaign_name,
        tasks,
        model_identity=model_identity,
        adapter_identity=adapter_identity,
        execution_config=execution_config,
        contamination_audit=contamination_audit,
        arms=_arms(args),
        claim_eligible=False,
    )
    campaign_trust = _verified_campaign_trust(
        args,
        unsigned_plan=unsigned_plan,
        contamination_audit=contamination_audit,
    )
    claim_eligible = _claim_eligible(
        args,
        model_identity,
        adapter_identity,
        contamination_audit,
        campaign_trust,
    )
    plan = build_campaign_plan(
        args.campaign_name,
        tasks,
        model_identity=model_identity,
        adapter_identity=adapter_identity,
        execution_config=execution_config,
        contamination_audit=contamination_audit,
        campaign_trust=campaign_trust,
        arms=_arms(args),
        claim_eligible=claim_eligible,
    )
    return plan, tasks


def _expected_worker_plan(
    args: argparse.Namespace,
    persisted: CampaignPlan,
) -> tuple[CampaignPlan, tuple[PublicTaskRecord, ...]]:
    """Rebuild the worker contract without invoking answer-bearing generators."""

    tasks = _public_tasks_from_plan(persisted)
    model_identity, adapter_identity = _identity_material(args)
    contamination_audit = _contamination_audit(
        args,
        tasks,
        expected_training_corpus_sha256=_adapter_dataset_manifest_sha256(
            adapter_identity
        ),
    )
    execution_config = _execution_config(args, adapter_identity)
    unsigned_plan = build_campaign_plan(
        args.campaign_name,
        tasks,
        model_identity=model_identity,
        adapter_identity=adapter_identity,
        execution_config=execution_config,
        contamination_audit=contamination_audit,
        arms=_arms(args),
        claim_eligible=False,
    )
    campaign_trust = _verified_campaign_trust(
        args,
        unsigned_plan=unsigned_plan,
        contamination_audit=contamination_audit,
    )
    claim_eligible = _claim_eligible(
        args,
        model_identity,
        adapter_identity,
        contamination_audit,
        campaign_trust,
    )
    plan = build_campaign_plan(
        args.campaign_name,
        tasks,
        model_identity=model_identity,
        adapter_identity=adapter_identity,
        execution_config=execution_config,
        contamination_audit=contamination_audit,
        campaign_trust=campaign_trust,
        arms=_arms(args),
        claim_eligible=claim_eligible,
    )
    return plan, tasks


def _claim_eligible(
    args: argparse.Namespace,
    model_identity: dict[str, Any],
    adapter_identity: Mapping[str, Any],
    contamination_audit: dict[str, Any],
    campaign_trust: Mapping[str, Any] | None,
) -> bool:
    per_domain = int(getattr(args, "seed_count", 0) or len(args.seed_values))
    power = _statistical_power_plan(args)
    seed_entropy_bits = int(
        getattr(args, "seed_entropy_bits", 0)
        or min(value.bit_length() for value in args.seed_values)
    )
    runtime_bundle = model_identity.get("runtime_bundle")
    return bool(
        args.confirmatory
        and per_domain >= power["minimum_observations"]
        and power["powered_for_zero_loss_noninferiority"] is True
        and seed_entropy_bits >= 60
        and args.profile == "full"
        and set(args.domain_values) == set(FRONTIER_DOMAINS)
        and isinstance(runtime_bundle, dict)
        and runtime_bundle.get("model_type") == "qwen2"
        and runtime_bundle.get("logical_parameter_count_basis")
        == "architecture_config_logical"
        and int(runtime_bundle.get("logical_parameter_count") or 0) >= 30_000_000_000
        and _adapter_dataset_manifest_sha256(adapter_identity) is not None
        and contamination_audit.get("status") == "passed_zero_overlap"
        and isinstance(contamination_audit.get("signature"), dict)
        and contamination_audit["signature"].get("verified") is True
        and isinstance(campaign_trust, Mapping)
        and campaign_trust.get("prelaunch_verified") is True
        and campaign_trust.get("externally_custodied") is True
    )


def _persist_plan(campaign_dir: Path, plan: CampaignPlan) -> None:
    _atomic_create_or_verify(
        campaign_dir / PLAN_FILE,
        canonical_json_bytes(plan.to_dict()) + b"\n",
    )


def _load_persisted_plan(campaign_dir: Path) -> CampaignPlan:
    return CampaignPlan.from_dict(
        _read_canonical_json_artifact(
            campaign_dir / PLAN_FILE,
            role="campaign plan",
        )
    )


def _resolve_projection(model: Any, projection: str) -> tuple[Any, str, Any]:
    current = model
    parts = projection.split(".")
    if len(parts) < 4 or parts[:2] != ["model", "layers"]:
        raise CampaignProducerError(f"adapter projection path is invalid: {projection}")
    for segment in parts[:-1]:
        if segment.isdigit():
            try:
                current = current[int(segment)]
            except (IndexError, KeyError, TypeError) as exc:
                raise CampaignProducerError(
                    f"adapter projection index is invalid: {projection}"
                ) from exc
        else:
            try:
                current = getattr(current, segment)
            except AttributeError as exc:
                raise CampaignProducerError(
                    f"adapter projection owner is missing: {projection}"
                ) from exc
    leaf = parts[-1]
    try:
        original = getattr(current, leaf)
    except AttributeError as exc:
        raise CampaignProducerError(
            f"adapter projection is missing: {projection}"
        ) from exc
    return current, leaf, original


def _load_adapter(model: Any, adapter_dir: Path, manifest: dict[str, Any]) -> int:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.lora import LoRALinear

    from core.brain.llm.latent_cortex.fast_weights import _linear_dims
    from core.brain.llm.latent_cortex.recurrence_adapter import ScopedLoRALinear

    rank = int(manifest["lora"]["rank"])
    targets = tuple(manifest["lora"]["targets"])
    is_v2 = manifest.get("schema") == MANIFEST_SCHEMA_V2
    expected = int(
        manifest["lora"]["wrapped_projections" if is_v2 else "wrapped_projection_count"]
    )
    tensor_records = {record["key"]: record for record in manifest["tensors"]}
    objective_name = (
        "aura.recurrence_native_objective.v2"
        if is_v2
        else str(manifest["training_receipt"]["objective"].get("name") or "")
    )
    wrapper_type = (
        ScopedLoRALinear
        if objective_name == "aura.recurrence_native_objective.v2"
        else LoRALinear
    )
    projections = sorted(
        {key.removesuffix(".lora_a").removesuffix(".lora_b") for key in tensor_records}
    )
    if is_v2 and projections != sorted(manifest["lora"]["projection_paths"]):
        raise CampaignProducerError("v2 adapter projection inventory differs")
    if len(projections) != expected:
        raise CampaignProducerError(
            f"adapter topology mismatch: planned {len(projections)}, expected {expected}"
        )
    originals: list[tuple[Any, str, Any]] = []
    try:
        for projection in projections:
            target = projection.rsplit(".", 1)[-1]
            if target not in targets:
                raise CampaignProducerError(
                    f"adapter projection target is not declared: {projection}"
                )
            parent, leaf, original = _resolve_projection(model, projection)
            out_features, in_features = _linear_dims(original)
            a_shape = tuple(tensor_records[f"{projection}.lora_a"]["shape"])
            b_shape = tuple(tensor_records[f"{projection}.lora_b"]["shape"])
            if a_shape != (in_features, rank) or b_shape != (rank, out_features):
                raise CampaignProducerError(
                    f"adapter tensor dimensions do not match projection: {projection}"
                )
            originals.append((parent, leaf, original))
            setattr(parent, leaf, wrapper_type.from_base(original, r=rank))

        weights_path = adapter_dir / manifest["adapter"]["path"]
        expected_adapter_sha256 = manifest["adapter"]["sha256"]
        expected_adapter_size = int(manifest["adapter"]["size_bytes"])
        before_load = _read_stable_bytes(
            weights_path,
            max_bytes=max(1, expected_adapter_size),
        )
        if (
            len(before_load) != expected_adapter_size
            or _sha256_bytes(before_load) != expected_adapter_sha256
        ):
            raise CampaignProducerError("adapter bytes differ from frozen manifest")
        weights = mx.load(str(weights_path))
        expected_keys = set(tensor_records)
        if set(weights) != expected_keys:
            raise CampaignProducerError("adapter file keys differ from frozen manifest")
        parameter_map = dict(tree_flatten(model.parameters()))
        missing_parameters = sorted(expected_keys - set(parameter_map))
        if missing_parameters:
            raise CampaignProducerError(
                f"adapter parameters are not runtime-addressable: {missing_parameters[0]}"
            )
        model.load_weights(list(weights.items()), strict=False)
        loaded = dict(tree_flatten(model.parameters()))
        mx.eval(*(loaded[key] for key in sorted(expected_keys)))
        if any(
            not bool(mx.array_equal(loaded[key], weights[key])) for key in expected_keys
        ):
            raise CampaignProducerError("adapter weight readback mismatch")
        after_load = _read_stable_bytes(
            weights_path,
            max_bytes=max(1, expected_adapter_size),
        )
        if after_load != before_load:
            raise CampaignProducerError("adapter bytes changed across load boundary")
    except BaseException:  # noqa: BLE001 - adapter rollback on any exit; original re-raised
        for parent, leaf, original in reversed(originals):
            setattr(parent, leaf, original)
        raise
    mx.eval(model.parameters())
    return len(originals)


def _render_prompt(tokenizer: Any, task: PublicTaskRecord) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": task.prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )


def _vanilla_once(
    model: Any,
    tokenizer: Any,
    task: PublicTaskRecord,
    *,
    max_tokens: int,
    sample_seed: int | None = None,
) -> tuple[str, int]:
    import mlx.core as mx
    from mlx_lm import generate

    kwargs: dict[str, Any] = {}
    if sample_seed is not None:
        from mlx_lm.sample_utils import make_sampler

        mx.random.seed(sample_seed)
        kwargs["sampler"] = make_sampler(temp=0.7, top_p=0.95)
    rendered = _render_prompt(tokenizer, task)
    text = generate(
        model,
        tokenizer,
        prompt=rendered,
        max_tokens=max_tokens,
        verbose=False,
        **kwargs,
    )
    prompt_tokens = len(tokenizer.encode(rendered))
    output_tokens = max(1, len(tokenizer.encode(text)))
    return text, (prompt_tokens + output_tokens) * len(model.model.layers)


def _majority_output(outputs: list[str]) -> str:
    keys: list[str] = []
    for output in outputs:
        try:
            key = canonical_json_bytes(parse_final_answer(output)).decode("ascii")
        except (TypeError, ValueError):
            key = f"invalid:{_sha256_bytes(output.encode('utf-8', errors='replace'))}"
        keys.append(key)
    counts = Counter(keys)
    winner = min(counts, key=lambda key: (-counts[key], keys.index(key), key))
    return outputs[keys.index(winner)]


def _equal_compute(
    model: Any,
    tokenizer: Any,
    task: PublicTaskRecord,
    *,
    target_layer_apps: int,
    max_tokens: int,
    max_samples: int,
) -> tuple[str, int, int]:
    outputs: list[str] = []
    spent = 0
    seed_base = int(task.task_payload_sha256[:16], 16)
    for sample_index in range(max_samples):
        text, cost = _vanilla_once(
            model,
            tokenizer,
            task,
            max_tokens=max_tokens,
            sample_seed=(seed_base + sample_index) % (2**31 - 1),
        )
        outputs.append(text)
        spent += cost
        if spent >= target_layer_apps:
            break
    return _majority_output(outputs), spent, len(outputs)


def _make_rlc_engine(
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    execution_spec: Mapping[str, Any] | None,
) -> Any:
    from core.brain.llm.latent_cortex.engine import LatentCortexEngine

    config = _build_rlc_config(args, execution_spec)

    return LatentCortexEngine(
        model,
        tokenizer,
        config,
        model_path=str(Path(args.model).expanduser().resolve()),
    )


def _run_rlc(
    engine: Any,
    task: PublicTaskRecord,
    args: argparse.Namespace,
) -> tuple[str, int, dict[str, Any]]:
    from core.brain.llm.latent_cortex.task_verifiers import EpisodeTaskVerifier
    from core.brain.llm.latent_cortex.types import ComputeBudget

    verifier = EpisodeTaskVerifier(task.prompt)
    budget = ComputeBudget(wall_clock_s=args.episode_timeout)
    result = engine.reason(
        messages=[{"role": "user", "content": task.prompt}],
        budget=budget,
        verifier=verifier,
        domain=task.domain,
        decode_max_tokens=args.decode_max_tokens,
    )
    receipt = result.receipt.to_dict()
    receipt["verifier_guidance"] = verifier.to_receipt()
    if not result.ok:
        raise CampaignProducerError(f"latent episode failed: {result.reason}")
    return result.text, budget.spent_layer_apps, receipt


def _prior_rlc_costs(journal: CampaignJournal) -> dict[tuple[str, str], int]:
    costs: dict[tuple[str, str], int] = {}
    for record in journal.result_records():
        definition = record["definition"]
        arm = definition["arm"]
        if arm in {BASE_RLC, ADAPTER_RLC}:
            costs[(definition["task_id"], arm)] = int(record["result"]["layer_apps"])
    return costs


def _worker_origin_context(
    args: argparse.Namespace,
    plan: CampaignPlan,
) -> dict[str, Any] | None:
    claim_required = plan.to_dict()["metadata"].get("claim_eligible") is True
    supplied = bool(
        args.worker_attempt_slot
        and args.worker_boot_id
        and args.worker_private_key
        and args.worker_authorization
    )
    if not claim_required:
        if supplied:
            raise CampaignProducerError(
                "preflight worker received claim-only origin credentials"
            )
        return None
    if not supplied:
        raise CampaignProducerError("claim worker origin credentials are required")
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    paths = _worker_origin_paths(
        campaign_dir,
        args.worker_arm,
        args.worker_attempt_slot,
    )
    if (
        Path(args.worker_private_key) != paths["private_key"]
        or Path(args.worker_authorization) != paths["attestation"]
    ):
        raise CampaignProducerError("worker origin credential path substitution")
    private_key = _load_worker_private_key(paths["private_key"])
    if args.worker_boot_id != _worker_boot_id(private_key):
        raise CampaignProducerError("worker boot identity differs from private key")
    policy = _load_campaign_trust_policy(args, require_current=True)
    if policy is None:
        raise CampaignProducerError("claim worker has no trusted policy")
    payload = _worker_authorization_payload(
        args,
        plan,
        policy,
        arm=args.worker_arm,
        attempt_slot=args.worker_attempt_slot,
        worker_public_key_raw=_worker_public_key_raw(private_key),
        worker_boot_id=_worker_boot_id(private_key),
    )
    attestation = _read_canonical_json_artifact(
        paths["attestation"], role="worker authorization attestation"
    )
    request = _read_canonical_json_artifact(
        paths["request"], role="worker authorization request"
    )
    signed = verify_legacy_worker_authorization(
        policy,
        attestation,
        expected_payload=payload,
        not_after_unix=int(time.time()),
    )
    if signed != request.get("signed_payload"):
        raise CampaignProducerError(
            "worker authorization differs from its issued request"
        )
    authorization_manifest = _read_canonical_json_artifact(
        campaign_dir / WORKER_AUTHORIZATION_MANIFEST_FILE,
        role="worker authorization manifest",
    )
    authorization_manifest = _validate_worker_authorization_manifest(
        args,
        plan,
        policy,
        authorization_manifest,
    )
    return {
        "policy": policy,
        "payload": payload,
        "attestation": attestation,
        "private_key": private_key,
        "paths": paths,
        "authorization_manifest": authorization_manifest,
    }


def _verify_existing_arm_origin_chain(
    args: argparse.Namespace,
    plan: CampaignPlan,
    *,
    arm: str,
    policy: Any,
    authorization_manifest: Mapping[str, Any],
    records: tuple[dict[str, Any], ...],
) -> tuple[int, str, int]:
    ordered = [
        record
        for record in records
        if record["definition"].get("arm") == arm
    ]
    ordered.sort(
        key=lambda record: int(
            record["definition"]["execution_ordinal_within_arm"]
        )
    )
    sequence = 0
    previous = ZERO_SHA256
    latest_slot = 0
    entries = authorization_manifest.get("entries")
    if not isinstance(entries, list):
        raise CampaignProducerError("worker authorization entries are invalid")
    for record in ordered:
        result = record.get("result")
        origin = result.get("worker_origin") if isinstance(result, dict) else None
        signed_payload = (
            origin.get("signed_payload") if isinstance(origin, dict) else None
        )
        attempt_slot = (
            signed_payload.get("worker_attempt_slot")
            if isinstance(signed_payload, dict)
            else None
        )
        if (
            isinstance(attempt_slot, bool)
            or not isinstance(attempt_slot, int)
            or attempt_slot < latest_slot
            or attempt_slot <= 0
            or attempt_slot > args.max_infra_attempts
        ):
            raise CampaignProducerError("worker result attempt-slot order is invalid")
        entry = next(
            (
                candidate
                for candidate in entries
                if isinstance(candidate, dict)
                and candidate.get("arm") == arm
                and candidate.get("attempt_slot") == attempt_slot
            ),
            None,
        )
        authorization_payload = (
            entry.get("authorization_payload")
            if isinstance(entry, dict)
            else None
        )
        public_b64 = (
            authorization_payload.get("worker_public_key_b64")
            if isinstance(authorization_payload, dict)
            else None
        )
        try:
            public_raw = base64.b64decode(public_b64, validate=True)
        except (TypeError, ValueError) as exc:
            raise CampaignProducerError(
                "worker chain authorization key is invalid"
            ) from exc
        authorization = _worker_authorization_payload(
            args,
            plan,
            policy,
            arm=arm,
            attempt_slot=attempt_slot,
            worker_public_key_raw=public_raw,
            worker_boot_id=str(entry.get("worker_boot_id") or ""),
        )
        if authorization != authorization_payload:
            raise CampaignProducerError(
                "worker chain authorization differs from reconstruction"
            )
        campaign_dir = Path(args.campaign_dir).expanduser().resolve()
        paths = _worker_origin_paths(campaign_dir, arm, attempt_slot)
        attestation = _read_canonical_json_artifact(
            paths["attestation"], role="prior worker authorization"
        )
        sequence += 1
        verify_legacy_worker_result_origin(
            policy,
            authorization_attestation=attestation,
            expected_authorization_payload=authorization,
            result=result,
            expected_cell_id=record["cell_id"],
            expected_attempt_id=record["attempt_id"],
            expected_sequence=sequence,
            expected_previous_origin_sha256=previous,
        )
        previous = origin["origin_sha256"]
        latest_slot = attempt_slot
    return sequence, previous, latest_slot


def _execute_worker(
    args: argparse.Namespace,
    plan: CampaignPlan,
    tasks: tuple[PublicTaskRecord, ...],
) -> int:
    arm = args.worker_arm
    if arm not in _arms(args):
        raise CampaignProducerError(f"worker arm is outside frozen plan: {arm}")
    model_dir = Path(args.model).expanduser().resolve(strict=True)
    model_path = str(model_dir)
    adapter_dir = Path(args.adapter).expanduser().resolve(strict=True)
    metadata = plan.to_dict()["metadata"]
    manifest = metadata["adapter_identity"]["manifest"]
    task_by_id = {task.task_id: task for task in tasks}
    origin_context = _worker_origin_context(args, plan)

    from mlx_lm import load

    from core.runtime.model_lane_control import standalone_model_lane

    with standalone_model_lane(
        owner_id=f"rlc-paired:{arm}:{os.getpid()}",
        model_path=model_path,
        purpose="benchmark",
        preemptible=False,
        metadata={"tool": "run_latent_cortex_paired_campaign", "arm": arm},
    ):
        load_started = time.monotonic()
        with _deadline_alarm(args.load_timeout, "model_load"):
            planned_model = metadata["model_identity"]
            planned_runtime = planned_model["runtime_bundle"]
            personality_path = (
                str(planned_model.get("personality_adapter_path") or "") or None
            )
            planned_load_boundary = {
                "weight_fingerprint": planned_model["fingerprint"],
                "weight_method": planned_model["method"],
                "weight_file_count": planned_model["files"],
                "runtime_bundle_sha256": planned_runtime["bundle_sha256"],
                "model_behavior_bundle": planned_model["model_behavior_bundle"],
                "runtime_environment": planned_model["runtime_environment"],
                "personality_adapter": planned_model["personality_adapter"],
                "effective_stack_sha256": planned_model["effective_stack_sha256"],
            }
            pre_load_boundary = _model_load_boundary_identity(
                model_dir, personality_path
            )
            if pre_load_boundary != planned_load_boundary:
                raise CampaignProducerError(
                    "model bytes differ from frozen plan before load"
                )
            actual_adapter_identity: dict[str, Any] | None = None
            if arm.startswith("adapter_"):
                actual_adapter_identity = _adapter_load_boundary_identity(
                    adapter_dir,
                    manifest,
                    adapter_id=args.adapter_id,
                    base_checkpoint={
                        "fingerprint": planned_model["fingerprint"],
                        "method": planned_model["method"],
                        "files": planned_model["files"],
                    },
                    model_behavior_bundle=planned_model["model_behavior_bundle"],
                    personality_identity=planned_model["personality_adapter"],
                    runtime_environment=planned_model["runtime_environment"],
                )
                if (
                    actual_adapter_identity
                    != metadata["adapter_identity"]["identity_receipt"]
                ):
                    raise CampaignProducerError(
                        "adapter bytes differ from frozen plan before load"
                    )
            load_kwargs = {"adapter_path": personality_path} if personality_path else {}
            model, tokenizer = load(model_path, **load_kwargs)
            wrapped = 0
            if arm.startswith("adapter_"):
                wrapped = _load_adapter(model, adapter_dir, manifest)
                post_adapter_identity = _adapter_load_boundary_identity(
                    adapter_dir,
                    manifest,
                    adapter_id=args.adapter_id,
                    base_checkpoint={
                        "fingerprint": planned_model["fingerprint"],
                        "method": planned_model["method"],
                        "files": planned_model["files"],
                    },
                    model_behavior_bundle=planned_model["model_behavior_bundle"],
                    personality_identity=planned_model["personality_adapter"],
                    runtime_environment=planned_model["runtime_environment"],
                )
                if post_adapter_identity != actual_adapter_identity:
                    raise CampaignProducerError(
                        "adapter identity changed across load boundary"
                    )
            post_load_boundary = _model_load_boundary_identity(
                model_dir, personality_path
            )
            if post_load_boundary != pre_load_boundary:
                raise CampaignProducerError(
                    "model identity changed across load boundary"
                )
            worker_identity = build_worker_identity(
                model,
                model_path=model_path,
                worker_boot_id=(
                    origin_context["payload"]["worker_boot_id"]
                    if origin_context is not None
                    else uuid.uuid4().hex
                ),
                worker_source_path=Path(__file__).resolve(),
            )
            worker_identity.update(
                {
                    "worker_weight_fingerprint": post_load_boundary[
                        "weight_fingerprint"
                    ],
                    "worker_weight_fingerprint_method": post_load_boundary[
                        "weight_method"
                    ],
                    "worker_weight_file_count": post_load_boundary["weight_file_count"],
                    "worker_runtime_bundle_sha256": post_load_boundary[
                        "runtime_bundle_sha256"
                    ],
                    "worker_personality_adapter": post_load_boundary[
                        "personality_adapter"
                    ],
                    "worker_effective_stack_sha256": post_load_boundary[
                        "effective_stack_sha256"
                    ],
                    "worker_load_boundary_verified": True,
                }
            )
            if (
                worker_identity["worker_model_parameter_count"]
                != planned_runtime["logical_parameter_count"]
                or worker_identity["worker_model_parameter_count_basis"]
                != planned_runtime["logical_parameter_count_basis"]
            ):
                raise CampaignProducerError(
                    "loaded model identity differs from frozen plan"
                )
        load_elapsed = time.monotonic() - load_started
        if load_elapsed > args.load_timeout:
            raise CampaignProducerError(
                f"model load exceeded budget: {load_elapsed:.3f}s > {args.load_timeout:.3f}s"
            )
        warm_started = time.monotonic()
        from mlx_lm import generate

        warm_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Reply with OK."}],
            add_generation_prompt=True,
            tokenize=False,
        )
        with _deadline_alarm(args.warmup_timeout, "model_warmup"):
            generate(model, tokenizer, prompt=warm_prompt, max_tokens=2, verbose=False)
        warm_elapsed = time.monotonic() - warm_started
        if warm_elapsed > args.warmup_timeout:
            raise CampaignProducerError(
                f"warmup exceeded budget: {warm_elapsed:.3f}s > {args.warmup_timeout:.3f}s"
            )

        raw_execution_spec = metadata["execution_config"].get("adapter_execution_spec")
        execution_spec = (
            raw_execution_spec if isinstance(raw_execution_spec, Mapping) else None
        )
        rlc_engine = (
            _make_rlc_engine(model, tokenizer, args, execution_spec)
            if arm.endswith("_rlc")
            else None
        )
        campaign_dir = Path(args.campaign_dir).expanduser().resolve()
        with CampaignJournal(campaign_dir / JOURNAL_FILE, plan) as journal:
            costs = _prior_rlc_costs(journal)
            origin_sequence = 0
            previous_origin_sha256 = ZERO_SHA256
            if origin_context is not None:
                (
                    origin_sequence,
                    previous_origin_sha256,
                    latest_attempt_slot,
                ) = _verify_existing_arm_origin_chain(
                    args,
                    plan,
                    arm=arm,
                    policy=origin_context["policy"],
                    authorization_manifest=origin_context[
                        "authorization_manifest"
                    ],
                    records=journal.result_records(),
                )
                if args.worker_attempt_slot < latest_attempt_slot:
                    raise CampaignProducerError(
                        "worker attempt slot regresses the signed result chain"
                    )
            sealed = set(journal.resume().sealed_cell_ids)
            pending = [
                cell_id
                for cell_id in journal.resume().runnable_cell_ids
                if plan.cell_definition(cell_id)["arm"] == arm
                and cell_id not in sealed
            ]
            pending.sort(
                key=lambda cell_id: int(
                    plan.cell_definition(cell_id)["execution_ordinal_within_arm"]
                )
            )
            for cell_id in pending:
                definition = plan.cell_definition(cell_id)
                task = task_by_id[definition["task_id"]]
                attempt_id = journal.start_cell(cell_id)
                started = time.monotonic()
                try:
                    with _deadline_alarm(args.episode_timeout, "campaign_cell"):
                        receipt: dict[str, Any] = {}
                        samples = 1
                        if arm.endswith("_rlc"):
                            text, layer_apps, receipt = _run_rlc(rlc_engine, task, args)
                        elif arm.endswith("_equal_compute"):
                            source_arm = (
                                BASE_RLC if arm == BASE_EQUAL_COMPUTE else ADAPTER_RLC
                            )
                            target = costs.get((task.task_id, source_arm))
                            if target is None:
                                raise CampaignProducerError(
                                    "equal-compute prerequisite missing"
                                )
                            text, layer_apps, samples = _equal_compute(
                                model,
                                tokenizer,
                                task,
                                target_layer_apps=target,
                                max_tokens=args.decode_max_tokens,
                                max_samples=args.equal_compute_max_samples,
                            )
                        else:
                            text, layer_apps = _vanilla_once(
                                model,
                                tokenizer,
                                task,
                                max_tokens=args.decode_max_tokens,
                            )
                    if receipt:
                        receipt.update(worker_identity)
                    elapsed = time.monotonic() - started
                    if elapsed > args.episode_timeout:
                        raise CampaignProducerError(
                            f"cell exceeded budget: {elapsed:.3f}s > {args.episode_timeout:.3f}s"
                        )
                    result = {
                        "arm": arm,
                        "text": text,
                        "output_sha256": _sha256_bytes(text.encode("utf-8")),
                        "layer_apps": int(layer_apps),
                        "latency_s": round(elapsed, 6),
                        "samples": samples,
                        "worker_pid": os.getpid(),
                        "model_load_s": round(load_elapsed, 6),
                        "warmup_s": round(warm_elapsed, 6),
                        "adapter_wrapped_projections": wrapped,
                        "adapter_identity_sha256": metadata["adapter_identity"][
                            "identity_receipt"
                        ]["composite_identity_sha256"]
                        if arm.startswith("adapter_")
                        else None,
                        "runtime_adapter_identity": actual_adapter_identity,
                        "runtime_model_identity": worker_identity,
                        "episode_receipt": receipt,
                    }
                    if origin_context is not None:
                        origin_sequence += 1
                        result["worker_origin"] = build_legacy_worker_result_origin(
                            authorization_attestation=origin_context["attestation"],
                            authorization_payload=origin_context["payload"],
                            private_key=origin_context["private_key"],
                            result_body=result,
                            cell_id=cell_id,
                            attempt_id=attempt_id,
                            worker_boot_id=worker_identity["worker_boot_id"],
                            sequence=origin_sequence,
                            previous_origin_sha256=previous_origin_sha256,
                        )
                        previous_origin_sha256 = result["worker_origin"][
                            "origin_sha256"
                        ]
                    journal.record_arm_result(cell_id, attempt_id, result)
                    print(
                        f"[{arm}] sealed {task.task_id} "
                        f"latency={elapsed:.2f}s layer_apps={layer_apps}",
                        flush=True,
                    )
                except BaseException as exc:  # noqa: BLE001 - infrastructure failure becomes a journaled cell failure, then re-raises
                    try:
                        journal.fail_cell(
                            cell_id,
                            attempt_id,
                            reason=f"infrastructure_failure:{type(exc).__name__}",
                            details={"message": str(exc)[:2000]},
                        )
                    except BaseException as journal_exc:  # noqa: BLE001 - crash path: the ORIGINAL infrastructure error must win
                        print(
                            "journal fail_cell during crash handling failed: "
                            f"{type(journal_exc).__name__}",
                            file=sys.stderr,
                        )
                    raise
    return 0


def _worker_args(
    args: argparse.Namespace,
    arm: str,
    *,
    worker_attempt_slot: int | None = None,
    worker_boot_id: str | None = None,
) -> list[str]:
    if str(getattr(args, "worker_arm", "") or ""):
        seed_count = int(args.seed_count)
        seed_entropy_bits = int(args.seed_entropy_bits)
    else:
        seed_values = tuple(
            getattr(
                args,
                "seed_values",
                tuple(int(value) for value in str(args.seeds).split(",") if value),
            )
        )
        seed_count = len(seed_values)
        seed_entropy_bits = min(value.bit_length() for value in seed_values)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--campaign-dir",
        str(Path(args.campaign_dir).expanduser().resolve()),
        "--campaign-name",
        args.campaign_name,
        "--model",
        args.model,
        "--adapter",
        args.adapter,
        "--adapter-id",
        args.adapter_id,
        "--personality-adapter",
        str(getattr(args, "personality_adapter", "trained")),
        "--seed-count",
        str(seed_count),
        "--seed-entropy-bits",
        str(seed_entropy_bits),
        "--domains",
        args.domains,
        "--difficulty",
        str(args.difficulty),
        "--task-registry-version",
        str(getattr(args, "task_registry_version", REGISTRY_VERSION)),
        "--profile",
        args.profile,
        "--n-slots",
        str(args.n_slots),
        "--branches",
        str(args.branches),
        "--rlc-steps",
        str(args.rlc_steps),
        "--rlc-profile",
        args.rlc_profile,
        "--decode-max-tokens",
        str(args.decode_max_tokens),
        "--episode-timeout",
        str(args.episode_timeout),
        "--load-timeout",
        str(args.load_timeout),
        "--warmup-timeout",
        str(args.warmup_timeout),
        "--arm-timeout",
        str(args.arm_timeout),
        "--campaign-timeout",
        str(args.campaign_timeout),
        "--equal-compute-max-samples",
        str(args.equal_compute_max_samples),
        "--max-infra-attempts",
        str(args.max_infra_attempts),
        "--worker-arm",
        arm,
    ]
    if worker_attempt_slot is not None:
        campaign_dir = Path(args.campaign_dir).expanduser().resolve()
        paths = _worker_origin_paths(campaign_dir, arm, worker_attempt_slot)
        if worker_boot_id is None:
            private_key = _load_or_create_worker_private_key(paths["private_key"])
            worker_boot_id = _worker_boot_id(private_key)
        command.extend(
            [
                "--worker-attempt-slot",
                str(worker_attempt_slot),
                "--worker-boot-id",
                worker_boot_id,
                "--worker-private-key",
                str(paths["private_key"]),
                "--worker-authorization",
                str(paths["attestation"]),
            ]
        )
    if args.confirmatory:
        command.append("--confirmatory")
    if args.contamination_audit:
        command.extend(["--contamination-audit", args.contamination_audit])
        command.extend(["--contamination-trust-root", args.contamination_trust_root])
    for option, attribute in (
        ("--campaign-trust-policy", "campaign_trust_policy"),
        ("--campaign-trust-root", "campaign_trust_root"),
        ("--task-issuer-attestation", "task_issuer_attestation"),
        ("--runner-attestation", "runner_attestation"),
    ):
        value = str(getattr(args, attribute, "") or "").strip()
        if value:
            command.extend([option, value])
    return command


def _arm_outputs_sealed(campaign_dir: Path, plan: CampaignPlan, arm: str) -> bool:
    with CampaignJournal(campaign_dir / JOURNAL_FILE, plan) as journal:
        sealed = set(journal.resume().sealed_cell_ids)
    expected = {
        cell_id
        for cell_id in plan.cell_ids
        if plan.cell_definition(cell_id)["arm"] == arm
    }
    return expected.issubset(sealed)


def _verify_worker_origin_chains(
    args: argparse.Namespace,
    plan: CampaignPlan,
) -> dict[str, Any] | None:
    if plan.to_dict()["metadata"].get("claim_eligible") is not True:
        return None
    policy = _load_campaign_trust_policy(args, require_current=True)
    if policy is None:
        raise CampaignProducerError("worker result chains have no trusted policy")
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    authorization_manifest = _read_canonical_json_artifact(
        campaign_dir / WORKER_AUTHORIZATION_MANIFEST_FILE,
        role="worker authorization manifest",
    )
    authorization_manifest = _validate_worker_authorization_manifest(
        args,
        plan,
        policy,
        authorization_manifest,
    )
    with CampaignJournal(campaign_dir / JOURNAL_FILE, plan) as journal:
        records = journal.result_records()
    if len(records) != len(plan.cell_ids):
        raise CampaignProducerError("worker result chain set is incomplete")
    chains: list[dict[str, Any]] = []
    for arm in _arms(args):
        sequence, chain_head, latest_slot = _verify_existing_arm_origin_chain(
            args,
            plan,
            arm=arm,
            policy=policy,
            authorization_manifest=authorization_manifest,
            records=records,
        )
        expected_count = sum(
            plan.cell_definition(cell_id)["arm"] == arm
            for cell_id in plan.cell_ids
        )
        if sequence != expected_count or latest_slot <= 0:
            raise CampaignProducerError(f"worker result chain is incomplete: {arm}")
        chains.append(
            {
                "arm": arm,
                "result_count": sequence,
                "latest_attempt_slot": latest_slot,
                "chain_head_sha256": chain_head,
            }
        )
    material = {
        "schema": "aura.latent_cortex.worker_origin_chains.v1",
        "policy_sha256": policy.policy_sha256,
        "plan_sha256": plan.plan_sha256,
        "chains": chains,
    }
    return {
        **material,
        "chains_sha256": _sha256_bytes(canonical_json_bytes(material)),
    }


def _worker_lifecycle_entry(
    campaign_dir: Path,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    arm = authorization["arm"]
    attempt_slot = authorization["attempt_slot"]
    payload = authorization["authorization_payload"]
    paths = _worker_origin_paths(campaign_dir, arm, attempt_slot)
    launch = _read_canonical_json_artifact(
        paths["launch"], role="worker launch receipt"
    )
    if (
        set(launch)
        != {
            "schema",
            "arm",
            "attempt_slot",
            "worker_boot_id",
            "worker_key_id",
            "worker_command_sha256",
            "authorization_request_sha256",
            "authorization_attestation_sha256",
            "launched_at_unix_ns",
        }
        or launch.get("schema") != "aura.latent_cortex.worker_launch.v1"
        or launch.get("arm") != arm
        or launch.get("attempt_slot") != attempt_slot
        or launch.get("worker_boot_id") != payload["worker_boot_id"]
        or launch.get("worker_key_id") != payload["worker_key_id"]
        or launch.get("worker_command_sha256")
        != payload["worker_command_sha256"]
        or launch.get("authorization_request_sha256")
        != authorization["request_sha256"]
        or launch.get("authorization_attestation_sha256")
        != authorization["attestation_sha256"]
        or type(launch.get("launched_at_unix_ns")) is not int
        or launch["launched_at_unix_ns"] <= 0
    ):
        raise CampaignProducerError("worker launch receipt differs")
    exit_receipt = _read_canonical_json_artifact(
        paths["exit"], role="worker exit receipt"
    )
    exit_material = dict(exit_receipt)
    exit_sha256 = exit_material.pop("receipt_sha256", None)
    if (
        set(exit_receipt)
        != {
            "schema",
            "launch_sha256",
            "outcome",
            "returncode",
            "error_type",
            "exited_at_unix_ns",
            "receipt_sha256",
        }
        or exit_receipt.get("schema") != "aura.latent_cortex.worker_exit.v2"
        or exit_receipt.get("launch_sha256")
        != _sha256_bytes(canonical_json_bytes(launch))
        or type(exit_receipt.get("exited_at_unix_ns")) is not int
        or exit_receipt["exited_at_unix_ns"] < launch["launched_at_unix_ns"]
        or exit_sha256 != _sha256_bytes(canonical_json_bytes(exit_material))
    ):
        raise CampaignProducerError("worker exit receipt differs")
    outcome = exit_receipt.get("outcome")
    if outcome == "process_exit":
        if (
            type(exit_receipt.get("returncode")) is not int
            or exit_receipt.get("error_type") is not None
        ):
            raise CampaignProducerError("worker process exit receipt differs")
    elif outcome == "launcher_failure":
        if (
            exit_receipt.get("returncode") is not None
            or not isinstance(exit_receipt.get("error_type"), str)
            or not exit_receipt["error_type"]
        ):
            raise CampaignProducerError("worker launcher failure receipt differs")
    else:
        raise CampaignProducerError("worker exit outcome differs")
    material = {
        "arm": arm,
        "attempt_slot": attempt_slot,
        "launch": launch,
        "exit": exit_receipt,
    }
    return {
        **material,
        "entry_sha256": _sha256_bytes(canonical_json_bytes(material)),
    }


def _build_worker_lifecycle_manifest(
    args: argparse.Namespace,
    plan: CampaignPlan,
    *,
    authorization_manifest: Mapping[str, Any],
) -> dict[str, Any] | None:
    if plan.to_dict()["metadata"].get("claim_eligible") is not True:
        return None
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    authorizations = authorization_manifest.get("entries")
    if not isinstance(authorizations, list):
        raise CampaignProducerError("worker lifecycle authorizations are invalid")
    entries: list[dict[str, Any]] = []
    consumed_positions: set[tuple[str, int]] = set()
    for authorization in authorizations:
        arm = authorization["arm"]
        attempt_slot = authorization["attempt_slot"]
        paths = _worker_origin_paths(campaign_dir, arm, attempt_slot)
        launch_exists = paths["launch"].exists()
        exit_exists = paths["exit"].exists()
        if not launch_exists:
            if paths["launch"].is_symlink() or exit_exists or paths["exit"].is_symlink():
                raise CampaignProducerError(
                    "worker lifecycle has an exit without a launch"
                )
            continue
        if not exit_exists:
            raise CampaignProducerError(
                "worker lifecycle launch has no terminal receipt"
            )
        entries.append(_worker_lifecycle_entry(campaign_dir, authorization))
        consumed_positions.add((arm, attempt_slot))
    with CampaignJournal(campaign_dir / JOURNAL_FILE, plan) as journal:
        records = journal.result_records()
    used_positions = {
        (
            record["definition"]["arm"],
            record["result"]["worker_origin"]["signed_payload"][
                "worker_attempt_slot"
            ],
        )
        for record in records
    }
    if not used_positions.issubset(consumed_positions):
        raise CampaignProducerError(
            "worker result has no complete lifecycle transaction"
        )
    if set(_arms(args)) != {arm for arm, _slot in used_positions}:
        raise CampaignProducerError("worker lifecycle arm coverage is incomplete")
    material = {
        "schema": WORKER_LIFECYCLE_MANIFEST_SCHEMA,
        "policy_sha256": authorization_manifest["policy_sha256"],
        "plan_sha256": plan.plan_sha256,
        "worker_authorization_manifest_sha256": authorization_manifest[
            "manifest_sha256"
        ],
        "entry_count": len(entries),
        "entries": entries,
    }
    manifest = {
        **material,
        "manifest_sha256": _sha256_bytes(canonical_json_bytes(material)),
    }
    _atomic_create_or_verify(
        campaign_dir / WORKER_LIFECYCLE_MANIFEST_FILE,
        canonical_json_bytes(manifest) + b"\n",
    )
    return manifest


def _seal_output_manifest(
    campaign_dir: Path,
    plan: CampaignPlan,
    *,
    worker_origin_chains: Mapping[str, Any] | None = None,
    worker_lifecycle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    with CampaignJournal(campaign_dir / JOURNAL_FILE, plan) as journal:
        records = journal.result_records()
    if len(records) != len(plan.cell_ids):
        raise CampaignProducerError("cannot seal an incomplete output set")
    by_cell = {record["cell_id"]: record for record in records}
    if set(by_cell) != set(plan.cell_ids):
        raise CampaignProducerError("sealed output set differs from the frozen plan")
    cells = [
        {
            "cell_id": cell_id,
            "attempt_id": by_cell[cell_id]["attempt_id"],
            "arm_result_event_sha256": by_cell[cell_id][
                "arm_result_event_sha256"
            ],
            "result_sha256": _sha256_bytes(
                canonical_json_bytes(by_cell[cell_id]["result"])
            ),
        }
        for cell_id in plan.cell_ids
    ]
    material = {
        "schema": SEALED_OUTPUT_MANIFEST_SCHEMA,
        "plan_sha256": plan.plan_sha256,
        "cell_count": len(cells),
        "cells": cells,
    }
    if worker_origin_chains is not None:
        material["worker_origin_chains"] = dict(worker_origin_chains)
    if worker_lifecycle is not None:
        material["worker_lifecycle_manifest_sha256"] = worker_lifecycle[
            "manifest_sha256"
        ]
    manifest = {
        **material,
        "manifest_sha256": _sha256_bytes(canonical_json_bytes(material)),
    }
    _atomic_create_or_verify(
        campaign_dir / SEALED_OUTPUT_MANIFEST_FILE,
        canonical_json_bytes(manifest) + b"\n",
    )
    return manifest


def _validate_worker_key_erasure_intent(
    campaign_dir: Path,
    plan: CampaignPlan,
    *,
    authorization_manifest: Mapping[str, Any],
    sealed_outputs: Mapping[str, Any],
    entry: Mapping[str, Any],
    intent: Mapping[str, Any],
) -> dict[str, Any]:
    material = dict(intent)
    intent_sha256 = material.pop("intent_sha256", None)
    if (
        set(intent)
        != {
            "schema",
            "policy_sha256",
            "plan_sha256",
            "worker_authorization_manifest_sha256",
            "sealed_output_manifest_sha256",
            "arm",
            "attempt_slot",
            "worker_boot_id",
            "worker_key_id",
            "method",
            "intent_at_unix_ns",
            "intent_sha256",
        }
        or intent.get("schema")
        != "aura.latent_cortex.worker_key_erasure_intent.v1"
        or intent.get("policy_sha256")
        != authorization_manifest["policy_sha256"]
        or intent.get("plan_sha256") != plan.plan_sha256
        or intent.get("worker_authorization_manifest_sha256")
        != authorization_manifest["manifest_sha256"]
        or intent.get("sealed_output_manifest_sha256")
        != sealed_outputs["manifest_sha256"]
        or intent.get("arm") != entry["arm"]
        or intent.get("attempt_slot") != entry["attempt_slot"]
        or intent.get("worker_boot_id") != entry["worker_boot_id"]
        or intent.get("worker_key_id") != entry["worker_key_id"]
        or intent.get("method")
        != "write_ahead_intent_then_unlink_and_parent_directory_fsync"
        or type(intent.get("intent_at_unix_ns")) is not int
        or intent["intent_at_unix_ns"] <= 0
        or intent_sha256 != _sha256_bytes(canonical_json_bytes(material))
    ):
        raise CampaignProducerError("worker key erasure intent differs")
    paths = _worker_origin_paths(
        campaign_dir,
        entry["arm"],
        entry["attempt_slot"],
    )
    if paths["erasure_intent"].is_symlink():
        raise CampaignProducerError("worker key erasure intent is a symlink")
    return dict(intent)


def _load_or_create_worker_key_erasure_intent(
    campaign_dir: Path,
    plan: CampaignPlan,
    *,
    authorization_manifest: Mapping[str, Any],
    sealed_outputs: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    paths = _worker_origin_paths(
        campaign_dir,
        entry["arm"],
        entry["attempt_slot"],
    )
    if paths["erasure_intent"].exists():
        intent = _read_canonical_json_artifact(
            paths["erasure_intent"], role="worker key erasure intent"
        )
        return _validate_worker_key_erasure_intent(
            campaign_dir,
            plan,
            authorization_manifest=authorization_manifest,
            sealed_outputs=sealed_outputs,
            entry=entry,
            intent=intent,
        )
    if paths["erasure_intent"].is_symlink():
        raise CampaignProducerError("worker key erasure intent is a symlink")
    material = {
        "schema": "aura.latent_cortex.worker_key_erasure_intent.v1",
        "policy_sha256": authorization_manifest["policy_sha256"],
        "plan_sha256": plan.plan_sha256,
        "worker_authorization_manifest_sha256": authorization_manifest[
            "manifest_sha256"
        ],
        "sealed_output_manifest_sha256": sealed_outputs["manifest_sha256"],
        "arm": entry["arm"],
        "attempt_slot": entry["attempt_slot"],
        "worker_boot_id": entry["worker_boot_id"],
        "worker_key_id": entry["worker_key_id"],
        "method": "write_ahead_intent_then_unlink_and_parent_directory_fsync",
        "intent_at_unix_ns": time.time_ns(),
    }
    intent = {
        **material,
        "intent_sha256": _sha256_bytes(canonical_json_bytes(material)),
    }
    _atomic_create_or_verify(
        paths["erasure_intent"], canonical_json_bytes(intent) + b"\n"
    )
    return intent


def _validate_worker_key_erasure_manifest(
    campaign_dir: Path,
    plan: CampaignPlan,
    *,
    authorization_manifest: Mapping[str, Any],
    sealed_outputs: Mapping[str, Any],
    aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    entries = authorization_manifest.get("entries")
    if not isinstance(entries, list):
        raise CampaignProducerError("worker erasure authorization set is invalid")
    material = dict(aggregate)
    manifest_sha256 = material.pop("manifest_sha256", None)
    receipts = aggregate.get("receipts")
    if (
        set(aggregate)
        != {
            "schema",
            "policy_sha256",
            "plan_sha256",
            "worker_authorization_manifest_sha256",
            "sealed_output_manifest_sha256",
            "receipt_count",
            "receipts",
            "all_private_paths_absent",
            "copy_exclusion_claimed",
            "manifest_sha256",
        }
        or aggregate.get("schema")
        != "aura.latent_cortex.worker_key_erasure_manifest.v2"
        or aggregate.get("policy_sha256")
        != authorization_manifest["policy_sha256"]
        or aggregate.get("plan_sha256") != plan.plan_sha256
        or aggregate.get("worker_authorization_manifest_sha256")
        != authorization_manifest["manifest_sha256"]
        or aggregate.get("sealed_output_manifest_sha256")
        != sealed_outputs["manifest_sha256"]
        or aggregate.get("receipt_count") != len(entries)
        or not isinstance(receipts, list)
        or len(receipts) != len(entries)
        or aggregate.get("all_private_paths_absent") is not True
        or aggregate.get("copy_exclusion_claimed") is not False
        or manifest_sha256 != _sha256_bytes(canonical_json_bytes(material))
    ):
        raise CampaignProducerError("worker key erasure manifest is invalid")
    for receipt, entry in zip(receipts, entries, strict=True):
        arm = entry["arm"]
        attempt_slot = entry["attempt_slot"]
        paths = _worker_origin_paths(campaign_dir, arm, attempt_slot)
        if paths["private_key"].exists() or paths["private_key"].is_symlink():
            raise CampaignProducerError(
                "worker private key reappeared after erasure"
            )
        intent = _read_canonical_json_artifact(
            paths["erasure_intent"], role="worker key erasure intent"
        )
        intent = _validate_worker_key_erasure_intent(
            campaign_dir,
            plan,
            authorization_manifest=authorization_manifest,
            sealed_outputs=sealed_outputs,
            entry=entry,
            intent=intent,
        )
        disk_receipt = _read_canonical_json_artifact(
            paths["erasure"], role="worker key erasure receipt"
        )
        receipt_material = dict(disk_receipt)
        receipt_sha256 = receipt_material.pop("receipt_sha256", None)
        if (
            receipt != disk_receipt
            or set(disk_receipt)
            != {
                "schema",
                "intent_sha256",
                "policy_sha256",
                "plan_sha256",
                "sealed_output_manifest_sha256",
                "arm",
                "attempt_slot",
                "worker_boot_id",
                "worker_key_id",
                "absence_observed_at_unix_ns",
                "method",
                "absence_verified",
                "copy_exclusion_claimed",
                "receipt_sha256",
            }
            or disk_receipt.get("schema")
            != "aura.latent_cortex.worker_key_erasure.v2"
            or disk_receipt.get("intent_sha256") != intent["intent_sha256"]
            or disk_receipt.get("policy_sha256")
            != authorization_manifest["policy_sha256"]
            or disk_receipt.get("plan_sha256") != plan.plan_sha256
            or disk_receipt.get("sealed_output_manifest_sha256")
            != sealed_outputs["manifest_sha256"]
            or disk_receipt.get("arm") != arm
            or disk_receipt.get("attempt_slot") != attempt_slot
            or disk_receipt.get("worker_boot_id") != entry["worker_boot_id"]
            or disk_receipt.get("worker_key_id") != entry["worker_key_id"]
            or type(disk_receipt.get("absence_observed_at_unix_ns")) is not int
            or disk_receipt["absence_observed_at_unix_ns"]
            < intent["intent_at_unix_ns"]
            or disk_receipt.get("method")
            != "write_ahead_intent_then_unlink_and_parent_directory_fsync"
            or disk_receipt.get("absence_verified") is not True
            or disk_receipt.get("copy_exclusion_claimed") is not False
            or receipt_sha256
            != _sha256_bytes(canonical_json_bytes(receipt_material))
        ):
            raise CampaignProducerError("worker key erasure receipt differs")
    return dict(aggregate)


def _erase_worker_private_keys(
    args: argparse.Namespace,
    plan: CampaignPlan,
    *,
    authorization_manifest: Mapping[str, Any],
    sealed_outputs: Mapping[str, Any],
) -> dict[str, Any] | None:
    if plan.to_dict()["metadata"].get("claim_eligible") is not True:
        return None
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    aggregate_path = campaign_dir / WORKER_KEY_ERASURE_MANIFEST_FILE
    entries = authorization_manifest.get("entries")
    if not isinstance(entries, list):
        raise CampaignProducerError("worker erasure authorization set is invalid")
    if aggregate_path.exists():
        aggregate = _read_canonical_json_artifact(
            aggregate_path, role="worker key erasure manifest"
        )
        return _validate_worker_key_erasure_manifest(
            campaign_dir,
            plan,
            authorization_manifest=authorization_manifest,
            sealed_outputs=sealed_outputs,
            aggregate=aggregate,
        )
    receipts: list[dict[str, Any]] = []
    origin_dir = _secure_worker_origin_dir(campaign_dir)
    for entry in entries:
        arm = entry["arm"]
        attempt_slot = entry["attempt_slot"]
        paths = _worker_origin_paths(campaign_dir, arm, attempt_slot)
        intent = _load_or_create_worker_key_erasure_intent(
            campaign_dir,
            plan,
            authorization_manifest=authorization_manifest,
            sealed_outputs=sealed_outputs,
            entry=entry,
        )
        if paths["erasure"].exists():
            if paths["private_key"].exists():
                raise CampaignProducerError(
                    "worker key and erasure receipt coexist"
                )
            receipt = _read_canonical_json_artifact(
                paths["erasure"], role="worker key erasure receipt"
            )
            receipts.append(receipt)
            continue
        if paths["private_key"].exists():
            private_key = _load_worker_private_key(paths["private_key"])
            public_raw = _worker_public_key_raw(private_key)
            if _sha256_bytes(public_raw) != entry.get("worker_key_id"):
                raise CampaignProducerError(
                    "worker private key differs from authorization before erasure"
                )
            paths["private_key"].unlink()
            directory_fd = os.open(
                origin_dir, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        elif paths["private_key"].is_symlink():
            raise CampaignProducerError("worker private key is a symlink")
        material = {
            "schema": "aura.latent_cortex.worker_key_erasure.v2",
            "intent_sha256": intent["intent_sha256"],
            "policy_sha256": authorization_manifest["policy_sha256"],
            "plan_sha256": plan.plan_sha256,
            "sealed_output_manifest_sha256": sealed_outputs["manifest_sha256"],
            "arm": arm,
            "attempt_slot": attempt_slot,
            "worker_boot_id": entry["worker_boot_id"],
            "worker_key_id": entry["worker_key_id"],
            "absence_observed_at_unix_ns": time.time_ns(),
            "method": "write_ahead_intent_then_unlink_and_parent_directory_fsync",
            "absence_verified": not paths["private_key"].exists(),
            "copy_exclusion_claimed": False,
        }
        receipt = {
            **material,
            "receipt_sha256": _sha256_bytes(canonical_json_bytes(material)),
        }
        _atomic_create_or_verify(
            paths["erasure"], canonical_json_bytes(receipt) + b"\n"
        )
        receipts.append(receipt)
    material = {
        "schema": "aura.latent_cortex.worker_key_erasure_manifest.v2",
        "policy_sha256": authorization_manifest["policy_sha256"],
        "plan_sha256": plan.plan_sha256,
        "worker_authorization_manifest_sha256": authorization_manifest[
            "manifest_sha256"
        ],
        "sealed_output_manifest_sha256": sealed_outputs["manifest_sha256"],
        "receipt_count": len(receipts),
        "receipts": receipts,
        "all_private_paths_absent": all(
            not _worker_origin_paths(
                campaign_dir,
                entry["arm"],
                entry["attempt_slot"],
            )["private_key"].exists()
            for entry in entries
        ),
        "copy_exclusion_claimed": False,
    }
    aggregate = {
        **material,
        "manifest_sha256": _sha256_bytes(canonical_json_bytes(material)),
    }
    _atomic_create_or_verify(
        aggregate_path, canonical_json_bytes(aggregate) + b"\n"
    )
    return _validate_worker_key_erasure_manifest(
        campaign_dir,
        plan,
        authorization_manifest=authorization_manifest,
        sealed_outputs=sealed_outputs,
        aggregate=aggregate,
    )


def _answer_reveal_payload(
    plan: CampaignPlan,
    tasks: tuple[FrontierTask, ...],
    sealed_outputs: Mapping[str, Any],
) -> dict[str, Any]:
    by_id = {task.task_id: task for task in tasks}
    task_manifest = plan.to_dict()["metadata"]["task_manifest"]
    answers: list[dict[str, Any]] = []
    for public in task_manifest["tasks"]:
        task = by_id.get(public["task_id"])
        if task is None:
            raise CampaignProducerError("answer reveal task is absent from issuer set")
        payload = task.reveal_for_verifier()
        if _sha256_bytes(canonical_json_bytes(payload)) != public[
            "answer_commitment_sha256"
        ]:
            raise CampaignProducerError("answer reveal differs from prelaunch commitment")
        answers.append(
            {
                "task_id": task.task_id,
                "answer_commitment_sha256": public[
                    "answer_commitment_sha256"
                ],
                "answer_payload": payload,
            }
        )
    return {
        "schema": ANSWER_REVEAL_PAYLOAD_SCHEMA,
        "campaign_name": plan.campaign_name,
        "plan_sha256": plan.plan_sha256,
        "sealed_output_manifest_sha256": sealed_outputs["manifest_sha256"],
        "task_commitment_sha256": plan.to_dict()["metadata"]["task_commitment"][
            "commitment_sha256"
        ],
        "answers": answers,
    }


def _load_or_prepare_role_request(
    path: Path,
    *,
    policy: Any,
    role: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if path.exists():
        request = _read_json_artifact(str(path), role=f"{role} signature request")
        signed_payload = request.get("signed_payload")
        signed_at = (
            signed_payload.get("signed_at_unix")
            if isinstance(signed_payload, dict)
            else None
        )
        if type(signed_at) is not int:
            raise CampaignProducerError(f"{role} request timestamp is invalid")
        expected = prepare_role_signature_request(
            policy,
            role=role,
            payload=payload,
            signed_at_unix=signed_at,
        )
        if request != expected:
            raise CampaignProducerError(
                f"{role} request differs from the current payload"
            )
        return request
    request = prepare_role_signature_request(
        policy,
        role=role,
        payload=payload,
        signed_at_unix=int(time.time()),
    )
    _atomic_create_or_verify(path, canonical_json_bytes(request) + b"\n")
    return request


def _read_canonical_json_artifact(path: Path, *, role: str) -> dict[str, Any]:
    payload = _read_stable_bytes(path, max_bytes=64 * 1024 * 1024)
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignProducerError(f"{role} is not valid JSON") from exc
    if (
        not isinstance(document, dict)
        or payload != canonical_json_bytes(document) + b"\n"
    ):
        raise CampaignProducerError(f"{role} is not canonical JSON")
    return document


def _validate_worker_authorization_manifest(
    args: argparse.Namespace,
    plan: CampaignPlan,
    policy: Any,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    material = dict(manifest)
    manifest_sha256 = material.pop("manifest_sha256", None)
    if (
        set(manifest)
        != {
            "schema",
            "claim_required",
            "campaign_name",
            "policy_sha256",
            "protocol_sha256",
            "plan_sha256",
            "attempt_slots_per_arm",
            "entries",
            "manifest_sha256",
        }
        or manifest.get("schema") != WORKER_AUTHORIZATION_MANIFEST_SCHEMA
        or manifest.get("claim_required") is not True
        or manifest.get("campaign_name") != plan.campaign_name
        or manifest.get("policy_sha256") != policy.policy_sha256
        or manifest.get("protocol_sha256") != _campaign_protocol_sha256()
        or manifest.get("plan_sha256") != plan.plan_sha256
        or manifest.get("attempt_slots_per_arm") != args.max_infra_attempts
        or manifest_sha256 != _sha256_bytes(canonical_json_bytes(material))
        or not isinstance(manifest.get("entries"), list)
    ):
        raise CampaignProducerError("worker authorization manifest is invalid")
    expected_positions = [
        (arm, attempt_slot)
        for arm in _arms(args)
        for attempt_slot in range(1, args.max_infra_attempts + 1)
    ]
    entries = manifest["entries"]
    if len(entries) != len(expected_positions):
        raise CampaignProducerError("worker authorization manifest is incomplete")
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    for entry, (arm, attempt_slot) in zip(entries, expected_positions, strict=True):
        if (
            not isinstance(entry, dict)
            or set(entry)
            != {
                "arm",
                "attempt_slot",
                "worker_boot_id",
                "worker_key_id",
                "authorization_payload",
                "request_sha256",
                "attestation_sha256",
            }
            or entry.get("arm") != arm
            or entry.get("attempt_slot") != attempt_slot
        ):
            raise CampaignProducerError(
                "worker authorization manifest entry is invalid"
            )
        payload = entry.get("authorization_payload")
        public_b64 = (
            payload.get("worker_public_key_b64")
            if isinstance(payload, dict)
            else None
        )
        try:
            public_raw = base64.b64decode(public_b64, validate=True)
        except (TypeError, ValueError) as exc:
            raise CampaignProducerError(
                "worker authorization public key is invalid"
            ) from exc
        expected_payload = _worker_authorization_payload(
            args,
            plan,
            policy,
            arm=arm,
            attempt_slot=attempt_slot,
            worker_public_key_raw=public_raw,
            worker_boot_id=str(entry.get("worker_boot_id") or ""),
        )
        if (
            payload != expected_payload
            or entry.get("worker_key_id") != payload.get("worker_key_id")
        ):
            raise CampaignProducerError(
                "worker authorization payload differs from reconstruction"
            )
        paths = _worker_origin_paths(campaign_dir, arm, attempt_slot)
        request = _read_canonical_json_artifact(
            paths["request"], role="worker authorization request"
        )
        attestation = _read_canonical_json_artifact(
            paths["attestation"], role="worker authorization attestation"
        )
        try:
            signed = verify_legacy_worker_authorization(
                policy,
                attestation,
                expected_payload=expected_payload,
                not_after_unix=int(time.time()),
            )
        except ValueError as exc:
            raise CampaignProducerError(
                "worker authorization attestation is invalid"
            ) from exc
        if (
            request.get("request_sha256") != entry.get("request_sha256")
            or signed != request.get("signed_payload")
            or _sha256_bytes(canonical_json_bytes(attestation))
            != entry.get("attestation_sha256")
        ):
            raise CampaignProducerError(
                "worker authorization evidence differs from its manifest"
            )
    return dict(manifest)


def _admit_worker_authorizations(
    args: argparse.Namespace,
    plan: CampaignPlan,
) -> dict[str, Any] | None:
    metadata = plan.to_dict()["metadata"]
    if metadata.get("claim_eligible") is not True:
        return {
            "schema": WORKER_AUTHORIZATION_MANIFEST_SCHEMA,
            "claim_required": False,
        }
    policy = _load_campaign_trust_policy(args, require_current=True)
    trust = metadata.get("campaign_trust")
    if (
        policy is None
        or not isinstance(trust, dict)
        or trust.get("policy_sha256") != policy.policy_sha256
    ):
        raise CampaignProducerError("worker authorization has no trusted policy")
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    manifest_path = campaign_dir / WORKER_AUTHORIZATION_MANIFEST_FILE
    if manifest_path.exists():
        manifest = _read_canonical_json_artifact(
            manifest_path, role="worker authorization manifest"
        )
        return _validate_worker_authorization_manifest(
            args,
            plan,
            policy,
            manifest,
        )
    entries: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for arm in _arms(args):
        for attempt_slot in range(1, args.max_infra_attempts + 1):
            paths = _worker_origin_paths(campaign_dir, arm, attempt_slot)
            private_key = _load_or_create_worker_private_key(paths["private_key"])
            payload = _worker_authorization_payload(
                args,
                plan,
                policy,
                arm=arm,
                attempt_slot=attempt_slot,
                worker_public_key_raw=_worker_public_key_raw(private_key),
                worker_boot_id=_worker_boot_id(private_key),
            )
            request = _load_or_prepare_role_request(
                paths["request"],
                policy=policy,
                role=CAMPAIGN_RUNNER,
                payload=payload,
            )
            if not paths["attestation"].exists():
                missing.append(
                    {
                        "arm": arm,
                        "attempt_slot": attempt_slot,
                        "request_path": str(paths["request"]),
                        "request_sha256": request["request_sha256"],
                        "attestation_path": str(paths["attestation"]),
                    }
                )
                continue
            attestation = _read_canonical_json_artifact(
                paths["attestation"],
                role=f"{arm} attempt {attempt_slot} worker authorization",
            )
            try:
                signed = verify_legacy_worker_authorization(
                    policy,
                    attestation,
                    expected_payload=payload,
                    not_after_unix=int(time.time()),
                )
            except ValueError as exc:
                raise CampaignProducerError(
                    "worker authorization attestation is invalid"
                ) from exc
            if signed != request["signed_payload"]:
                raise CampaignProducerError(
                    "worker authorization does not sign the issued request"
                )
            entries.append(
                {
                    "arm": arm,
                    "attempt_slot": attempt_slot,
                    "worker_boot_id": payload["worker_boot_id"],
                    "worker_key_id": payload["worker_key_id"],
                    "authorization_payload": payload,
                    "request_sha256": request["request_sha256"],
                    "attestation_sha256": _sha256_bytes(
                        canonical_json_bytes(attestation)
                    ),
                }
            )
    if missing:
        print(
            json.dumps(
                {
                    "state": "worker_authorization_signatures_required",
                    "missing_count": len(missing),
                    "requests": missing,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return None
    material = {
        "schema": WORKER_AUTHORIZATION_MANIFEST_SCHEMA,
        "claim_required": True,
        "campaign_name": plan.campaign_name,
        "policy_sha256": policy.policy_sha256,
        "protocol_sha256": _campaign_protocol_sha256(),
        "plan_sha256": plan.plan_sha256,
        "attempt_slots_per_arm": args.max_infra_attempts,
        "entries": entries,
    }
    manifest = {
        **material,
        "manifest_sha256": _sha256_bytes(canonical_json_bytes(material)),
    }
    _atomic_create_or_verify(
        manifest_path,
        canonical_json_bytes(manifest) + b"\n",
    )
    return manifest


def _admit_answer_reveal(
    args: argparse.Namespace,
    plan: CampaignPlan,
    tasks: tuple[FrontierTask, ...],
    sealed_outputs: Mapping[str, Any],
) -> dict[str, Any] | None:
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    payload = _answer_reveal_payload(plan, tasks, sealed_outputs)
    metadata = plan.to_dict()["metadata"]
    attestation: dict[str, Any] | None = None
    request_sha256: str | None = None
    if metadata.get("claim_eligible") is True:
        policy = _load_campaign_trust_policy(args, require_current=True)
        trust = metadata.get("campaign_trust")
        if (
            policy is None
            or not isinstance(trust, dict)
            or trust.get("policy_sha256") != policy.policy_sha256
        ):
            raise CampaignProducerError("answer reveal has no trusted issuer policy")
        request = _load_or_prepare_role_request(
            campaign_dir / ANSWER_REVEAL_REQUEST_FILE,
            policy=policy,
            role=TASK_ISSUER,
            payload=payload,
        )
        request_sha256 = request["request_sha256"]
        attestation_path = str(
            getattr(args, "answer_reveal_attestation", "") or ""
        ).strip()
        if not attestation_path:
            print(
                json.dumps(
                    {
                        "state": "answer_reveal_signature_required",
                        "request_path": str(
                            campaign_dir / ANSWER_REVEAL_REQUEST_FILE
                        ),
                        "request_sha256": request_sha256,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
            return None
        attestation = _read_json_artifact(
            attestation_path, role="answer reveal attestation"
        )
        signed = verify_role_attestation(
            policy,
            attestation,
            role=TASK_ISSUER,
            expected_payload=payload,
            not_before_unix=request["signed_payload"]["signed_at_unix"],
        )
        if signed != request["signed_payload"]:
            raise CampaignProducerError(
                "answer reveal attestation does not sign the issued request"
            )
    reveal_material = {
        "payload": payload,
        "request_sha256": request_sha256,
        "task_issuer_attestation": attestation,
    }
    reveal = {
        "schema": "aura.latent_cortex.answer_reveal.v1",
        **reveal_material,
        "reveal_sha256": _sha256_bytes(canonical_json_bytes(reveal_material)),
    }
    _atomic_create_or_verify(
        campaign_dir / ANSWER_REVEAL_FILE,
        canonical_json_bytes(reveal) + b"\n",
    )
    return reveal


def _score_sealed_outputs(
    campaign_dir: Path,
    plan: CampaignPlan,
    tasks: tuple[FrontierTask, ...],
) -> None:
    task_by_id = {task.task_id: task for task in tasks}
    with CampaignJournal(campaign_dir / JOURNAL_FILE, plan) as journal:
        for record in journal.result_records():
            state = record["state"]
            if state == "COMMITTED":
                continue
            task = task_by_id.get(record["definition"]["task_id"])
            text = record["result"].get("text")
            if task is None or not isinstance(text, str):
                raise CampaignProducerError("sealed output cannot be scored")
            score = task.score(text).to_dict()
            verification = {
                "correct": score["correct"],
                "score_receipt": score,
                "answer_commitment_sha256": (
                    task.public.answer_commitment_sha256
                ),
            }
            if state == "ARM_RESULT":
                journal.record_verified(
                    record["cell_id"], record["attempt_id"], verification
                )
            elif state == "VERIFIED" and record["verification"] != verification:
                raise CampaignProducerError(
                    "persisted verification differs from post-seal scoring"
                )
            elif state != "VERIFIED":
                raise CampaignProducerError("sealed output state is invalid")
            journal.commit_cell(
                record["cell_id"],
                record["attempt_id"],
                {
                    "result_sha256": _sha256_bytes(
                        canonical_json_bytes(record["result"])
                    ),
                    "verification_sha256": _sha256_bytes(
                        canonical_json_bytes(verification)
                    ),
                },
            )


def _admit_final_run_envelope(
    args: argparse.Namespace,
    plan: CampaignPlan,
    *,
    sealed_outputs: Mapping[str, Any],
    answer_reveal: Mapping[str, Any],
    campaign_manifest: Mapping[str, Any],
    grade: Mapping[str, Any],
    worker_authorizations: Mapping[str, Any] | None = None,
    worker_lifecycle: Mapping[str, Any] | None = None,
    worker_key_erasure: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if plan.to_dict()["metadata"].get("claim_eligible") is not True:
        return {
            "schema": "aura.latent_cortex.final_run_envelope.v3",
            "claim_required": False,
        }
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    policy = _load_campaign_trust_policy(args, require_current=True)
    trust = plan.to_dict()["metadata"].get("campaign_trust")
    if (
        policy is None
        or not isinstance(trust, dict)
        or trust.get("policy_sha256") != policy.policy_sha256
    ):
        raise CampaignProducerError("final run has no trusted runner policy")
    if (
        worker_authorizations is None
        or worker_lifecycle is None
        or worker_key_erasure is None
    ):
        raise CampaignProducerError(
            "final run is missing worker authorization, lifecycle, or key "
            "erasure evidence"
        )
    if worker_authorizations != _read_canonical_json_artifact(
        campaign_dir / WORKER_AUTHORIZATION_MANIFEST_FILE,
        role="worker authorization manifest",
    ) or worker_lifecycle != _read_canonical_json_artifact(
        campaign_dir / WORKER_LIFECYCLE_MANIFEST_FILE,
        role="worker lifecycle manifest",
    ) or worker_key_erasure != _read_canonical_json_artifact(
        campaign_dir / WORKER_KEY_ERASURE_MANIFEST_FILE,
        role="worker key erasure manifest",
    ):
        raise CampaignProducerError(
            "final run worker evidence differs from canonical disk artifacts"
        )
    payload = {
        "schema": FINAL_RUN_PAYLOAD_SCHEMA,
        "campaign_name": plan.campaign_name,
        "policy_sha256": policy.policy_sha256,
        "protocol_sha256": _campaign_protocol_sha256(),
        "plan_sha256": plan.plan_sha256,
        "sealed_output_manifest_sha256": sealed_outputs["manifest_sha256"],
        "answer_reveal_sha256": answer_reveal["reveal_sha256"],
        "campaign_manifest_sha256": campaign_manifest["manifest_sha256"],
        "journal_head_sha256": campaign_manifest["journal_head_sha256"],
        "published_grade_sha256": grade["grade_sha256"],
        "worker_authorization_manifest_sha256": worker_authorizations[
            "manifest_sha256"
        ],
        "worker_lifecycle_manifest_sha256": worker_lifecycle[
            "manifest_sha256"
        ],
        "worker_key_erasure_manifest_sha256": worker_key_erasure[
            "manifest_sha256"
        ],
    }
    request = _load_or_prepare_role_request(
        campaign_dir / FINAL_RUN_REQUEST_FILE,
        policy=policy,
        role=CAMPAIGN_RUNNER,
        payload=payload,
    )
    attestation_path = str(
        getattr(args, "final_run_attestation", "") or ""
    ).strip()
    if not attestation_path:
        print(
            json.dumps(
                {
                    "state": "final_run_signature_required",
                    "request_path": str(campaign_dir / FINAL_RUN_REQUEST_FILE),
                    "request_sha256": request["request_sha256"],
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return None
    attestation = _read_json_artifact(
        attestation_path, role="final run attestation"
    )
    signed = verify_role_attestation(
        policy,
        attestation,
        role=CAMPAIGN_RUNNER,
        expected_payload=payload,
        not_before_unix=request["signed_payload"]["signed_at_unix"],
    )
    if signed != request["signed_payload"]:
        raise CampaignProducerError(
            "final run attestation does not sign the issued request"
        )
    material = {
        "payload": payload,
        "request_sha256": request["request_sha256"],
        "campaign_runner_attestation": attestation,
    }
    envelope = {
        "schema": "aura.latent_cortex.final_run_envelope.v3",
        **material,
        "envelope_sha256": _sha256_bytes(canonical_json_bytes(material)),
    }
    _atomic_create_or_verify(
        campaign_dir / FINAL_RUN_ENVELOPE_FILE,
        canonical_json_bytes(envelope) + b"\n",
    )
    return envelope


def _next_worker_attempt_slot(
    campaign_dir: Path,
    arm: str,
    *,
    maximum: int,
) -> int | None:
    for attempt_slot in range(1, maximum + 1):
        paths = _worker_origin_paths(campaign_dir, arm, attempt_slot)
        if not paths["launch"].exists():
            return attempt_slot
    return None


def _record_worker_launch(
    args: argparse.Namespace,
    arm: str,
    attempt_slot: int,
    command: list[str],
) -> tuple[dict[str, Any], dict[str, Path]]:
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    paths = _worker_origin_paths(campaign_dir, arm, attempt_slot)
    private_key = _load_worker_private_key(paths["private_key"])
    request = _read_canonical_json_artifact(
        paths["request"], role="worker authorization request"
    )
    attestation = _read_canonical_json_artifact(
        paths["attestation"], role="worker authorization attestation"
    )
    launch = {
        "schema": "aura.latent_cortex.worker_launch.v1",
        "arm": arm,
        "attempt_slot": attempt_slot,
        "worker_boot_id": _worker_boot_id(private_key),
        "worker_key_id": _sha256_bytes(_worker_public_key_raw(private_key)),
        "worker_command_sha256": _sha256_bytes(canonical_json_bytes(command)),
        "authorization_request_sha256": request["request_sha256"],
        "authorization_attestation_sha256": _sha256_bytes(
            canonical_json_bytes(attestation)
        ),
        "launched_at_unix_ns": time.time_ns(),
    }
    if paths["launch"].exists():
        raise CampaignProducerError("worker attempt slot was already consumed")
    _atomic_create_or_verify(
        paths["launch"], canonical_json_bytes(launch) + b"\n"
    )
    return launch, paths


def _record_worker_exit(
    paths: Mapping[str, Path],
    launch: Mapping[str, Any],
    *,
    returncode: int | None,
    outcome: str = "process_exit",
    error_type: str | None = None,
) -> None:
    if outcome not in {"process_exit", "launcher_failure"}:
        raise CampaignProducerError("worker exit outcome is invalid")
    if outcome == "process_exit":
        if type(returncode) is not int or error_type is not None:
            raise CampaignProducerError("worker process exit receipt is invalid")
    elif returncode is not None or not error_type:
        raise CampaignProducerError("worker launcher failure receipt is invalid")
    material = {
        "schema": "aura.latent_cortex.worker_exit.v2",
        "launch_sha256": _sha256_bytes(canonical_json_bytes(launch)),
        "outcome": outcome,
        "returncode": returncode,
        "error_type": error_type,
        "exited_at_unix_ns": time.time_ns(),
    }
    receipt = {
        **material,
        "receipt_sha256": _sha256_bytes(canonical_json_bytes(material)),
    }
    _atomic_create_or_verify(
        paths["exit"], canonical_json_bytes(receipt) + b"\n"
    )


def _run_child(
    args: argparse.Namespace,
    arm: str,
    timeout_s: float,
    *,
    worker_attempt_slot: int | None = None,
) -> int:
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    log_path = campaign_dir / LOG_FILE
    command = _worker_args(
        args,
        arm,
        worker_attempt_slot=worker_attempt_slot,
    )
    launch: dict[str, Any] | None = None
    paths: dict[str, Path] | None = None
    if worker_attempt_slot is not None:
        launch, paths = _record_worker_launch(
            args,
            arm,
            worker_attempt_slot,
            command,
        )
    try:
        if broker_available():
            returncode = run_brokered_process(
                command,
                cwd=REPO_ROOT,
                stdout_path=log_path,
                timeout_s=timeout_s,
            ).returncode
        else:
            with log_path.open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    returncode = process.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=15)
                    returncode = 124
    except BaseException as exc:
        if launch is not None and paths is not None:
            _record_worker_exit(
                paths,
                launch,
                returncode=None,
                outcome="launcher_failure",
                error_type=type(exc).__name__,
            )
        raise
    if launch is not None and paths is not None:
        _record_worker_exit(paths, launch, returncode=returncode)
    return returncode


def _detached_broker_policy(args: argparse.Namespace) -> list[dict[str, Any]]:
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    if args.confirmatory:
        return [
            {
                "command": _worker_args(
                    args,
                    arm,
                    worker_attempt_slot=attempt_slot,
                ),
                "cwd": str(REPO_ROOT),
                "stdout_path": str(campaign_dir / LOG_FILE),
                "timeout_s_max": float(args.arm_timeout),
                "max_invocations": 1,
            }
            for arm in _arms(args)
            for attempt_slot in range(1, args.max_infra_attempts + 1)
        ]
    return [
        {
            "command": _worker_args(args, arm),
            "cwd": str(REPO_ROOT),
            "stdout_path": str(campaign_dir / LOG_FILE),
            "timeout_s_max": float(args.arm_timeout),
            "max_invocations": int(args.max_infra_attempts),
        }
        for arm in _arms(args)
    ]


def _orchestrate(
    args: argparse.Namespace,
    plan: CampaignPlan,
    tasks: tuple[FrontierTask, ...],
) -> int:
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    deadline = time.monotonic() + args.campaign_timeout
    metadata = plan.to_dict()["metadata"]
    worker_authorizations = _admit_worker_authorizations(args, plan)
    if worker_authorizations is None:
        return 7
    worker_origin_required = worker_authorizations.get("claim_required") is True
    arm_execution_order = tuple(metadata["arm_execution_order"])
    if set(arm_execution_order) != set(_arms(args)):
        raise CampaignProducerError("frozen arm execution order is invalid")
    for arm in arm_execution_order:
        if arm not in _arms(args):
            continue
        attempts = 0
        while not _arm_outputs_sealed(campaign_dir, plan, arm):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print(
                    "campaign deadline exceeded; resumable evidence preserved",
                    flush=True,
                )
                return 3
            attempts += 1
            if attempts > args.max_infra_attempts:
                print(f"arm {arm} exhausted infrastructure attempts", flush=True)
                return 4
            worker_attempt_slot = None
            if worker_origin_required:
                worker_attempt_slot = _next_worker_attempt_slot(
                    campaign_dir,
                    arm,
                    maximum=args.max_infra_attempts,
                )
                if worker_attempt_slot is None:
                    print(
                        f"arm {arm} exhausted pre-authorized worker slots",
                        flush=True,
                    )
                    return 4
            code = _run_child(
                args,
                arm,
                min(args.arm_timeout, remaining),
                worker_attempt_slot=worker_attempt_slot,
            )
            print(f"arm {arm} process exit={code} attempt={attempts}", flush=True)
            if code != 0 and attempts >= args.max_infra_attempts:
                return code or 4

    worker_origin_chains = _verify_worker_origin_chains(args, plan)
    worker_lifecycle = _build_worker_lifecycle_manifest(
        args,
        plan,
        authorization_manifest=worker_authorizations,
    )
    sealed_outputs = _seal_output_manifest(
        campaign_dir,
        plan,
        worker_origin_chains=worker_origin_chains,
        worker_lifecycle=worker_lifecycle,
    )
    worker_key_erasure = _erase_worker_private_keys(
        args,
        plan,
        authorization_manifest=worker_authorizations,
        sealed_outputs=sealed_outputs,
    )
    answer_reveal = _admit_answer_reveal(args, plan, tasks, sealed_outputs)
    if answer_reveal is None:
        return 5
    _score_sealed_outputs(campaign_dir, plan, tasks)
    with CampaignJournal(campaign_dir / JOURNAL_FILE, plan) as journal:
        records = journal.committed_records()
        manifest = journal.finalize(campaign_dir / MANIFEST_FILE)
    grade = grade_campaign(
        records,
        plan=plan,
        issuer_tasks=tasks,
        trusted_contamination_root_sha256=(
            _load_contamination_trust_root(args.contamination_trust_root)[2]
            if args.contamination_audit
            else None
        ),
        trusted_campaign_policy_sha256=(
            metadata["campaign_trust"]["policy_sha256"]
            if isinstance(metadata.get("campaign_trust"), dict)
            else None
        ),
    )
    final_material = dict(grade)
    final_material.pop("grade_sha256", None)
    final_material["campaign_manifest_sha256"] = manifest["manifest_sha256"]
    final_material["sealed_output_manifest_sha256"] = sealed_outputs[
        "manifest_sha256"
    ]
    final_material["answer_reveal_sha256"] = answer_reveal["reveal_sha256"]
    if worker_authorizations.get("claim_required") is True:
        if worker_key_erasure is None:
            raise CampaignProducerError("worker key erasure evidence is missing")
        final_material["worker_authorization_manifest_sha256"] = (
            worker_authorizations["manifest_sha256"]
        )
        if worker_lifecycle is None:
            raise CampaignProducerError("worker lifecycle evidence is missing")
        final_material["worker_lifecycle_manifest_sha256"] = worker_lifecycle[
            "manifest_sha256"
        ]
        final_material["worker_key_erasure_manifest_sha256"] = (
            worker_key_erasure["manifest_sha256"]
        )
    final = {
        **final_material,
        "grade_sha256": _sha256_bytes(canonical_json_bytes(final_material)),
    }
    _atomic_create_or_verify(
        campaign_dir / GRADE_FILE, canonical_json_bytes(final) + b"\n"
    )
    final_run_envelope = _admit_final_run_envelope(
        args,
        plan,
        sealed_outputs=sealed_outputs,
        answer_reveal=answer_reveal,
        campaign_manifest=manifest,
        grade=final,
        worker_authorizations=worker_authorizations,
        worker_lifecycle=worker_lifecycle,
        worker_key_erasure=worker_key_erasure,
    )
    if final_run_envelope is None:
        return 6
    print(
        json.dumps(
            {
                "verdict": final["verdict"],
                "claim_tier": final["claim_tier"],
                "grade_sha256": final["grade_sha256"],
                "campaign_manifest_sha256": final["campaign_manifest_sha256"],
                "observed_task_count": final["observed_task_count"],
                "observed_cell_count": final["observed_cell_count"],
                "reasons": final["reasons"],
                "frontier_claim_eligible": False,
                "final_run_envelope_sha256": final_run_envelope.get(
                    "envelope_sha256"
                ),
                "grade_path": str(campaign_dir / GRADE_FILE),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if final["verdict"] in {"gain_preverified", "gain_proven"} else 2


def _prepare_trust_requests(args: argparse.Namespace) -> dict[str, Any]:
    tasks = _tasks(args)
    model_identity, adapter_identity = _identity_material(args)
    contamination_audit = _contamination_audit(
        args,
        tasks,
        expected_training_corpus_sha256=_adapter_dataset_manifest_sha256(
            adapter_identity
        ),
    )
    execution_config = _execution_config(args, adapter_identity)
    unsigned_plan = build_campaign_plan(
        args.campaign_name,
        tasks,
        model_identity=model_identity,
        adapter_identity=adapter_identity,
        execution_config=execution_config,
        contamination_audit=contamination_audit,
        arms=_arms(args),
        claim_eligible=False,
    )
    policy = _load_campaign_trust_policy(args, require_current=True)
    if policy is None:
        raise CampaignProducerError("campaign trust policy is required")
    if not _policy_auditor_matches(policy, contamination_audit):
        raise CampaignProducerError(
            "contamination auditor does not match the pre-pinned campaign role"
        )
    payloads = _prelaunch_payloads(
        args,
        unsigned_plan=unsigned_plan,
        policy=policy,
    )
    return {
        "schema": "aura.latent_cortex.campaign_trust_requests.v1",
        "campaign_name": args.campaign_name,
        "policy_sha256": policy.policy_sha256,
        "protocol_sha256": _campaign_protocol_sha256(),
        "unsigned_plan_sha256": unsigned_plan.plan_sha256,
        "externally_custodied": externally_custodied_roles(policy),
        "requests": payloads,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--campaign-name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--adapter-id", default="resident-32b-r1")
    parser.add_argument(
        "--personality-adapter",
        default="trained",
        help="trained, none, auto, or an explicit MLX personality-adapter directory",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="issuer-only generation seeds; required outside isolated worker mode",
    )
    parser.add_argument("--seed-count", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--seed-entropy-bits", type=int, default=0, help=argparse.SUPPRESS
    )
    parser.add_argument("--domains", default=",".join(FRONTIER_DOMAINS))
    parser.add_argument("--difficulty", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument(
        "--task-registry-version",
        choices=(REGISTRY_VERSION, CURRENT_REGISTRY_VERSION),
        default=REGISTRY_VERSION,
        help="versioned task lineage; legacy remains the replay-safe default",
    )
    parser.add_argument("--profile", choices=("primary", "full"), default="primary")
    parser.add_argument("--confirmatory", action="store_true")
    parser.add_argument("--contamination-audit", default="")
    parser.add_argument("--contamination-trust-root", default="")
    parser.add_argument("--campaign-trust-policy", default="")
    parser.add_argument("--campaign-trust-root", default="")
    parser.add_argument("--task-issuer-attestation", default="")
    parser.add_argument("--runner-attestation", default="")
    parser.add_argument("--answer-reveal-attestation", default="")
    parser.add_argument("--final-run-attestation", default="")
    parser.add_argument(
        "--prepare-trust",
        action="store_true",
        help="emit exact prelaunch role payloads without persisting or running a plan",
    )
    parser.add_argument("--n-slots", type=_positive_int, default=16)
    parser.add_argument("--branches", type=_positive_int, default=2)
    parser.add_argument("--rlc-steps", type=_positive_int, default=8)
    parser.add_argument(
        "--rlc-profile",
        choices=("recurrence_attribution", "resident_full_stack"),
        default="recurrence_attribution",
    )
    parser.add_argument("--decode-max-tokens", type=_positive_int, default=512)
    parser.add_argument("--episode-timeout", type=_positive_float, default=180.0)
    parser.add_argument("--load-timeout", type=_positive_float, default=600.0)
    parser.add_argument("--warmup-timeout", type=_positive_float, default=120.0)
    parser.add_argument("--arm-timeout", type=_positive_float, default=10800.0)
    parser.add_argument("--campaign-timeout", type=_positive_float, default=43200.0)
    parser.add_argument("--equal-compute-max-samples", type=_positive_int, default=8)
    parser.add_argument("--max-infra-attempts", type=_positive_int, default=3)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--worker-arm", choices=FULL_ARMS, default="", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--worker-attempt-slot", type=_positive_int, default=0, help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-boot-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-private-key", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-authorization", default="", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.worker_arm:
        args.seeds = ""
        args.seed_values = ()
    else:
        args.seed_values = _csv_ints(parser, args.seeds, "--seeds")
    args.domain_values = _csv_domains(parser, args.domains)
    worker_origin_values = (
        args.worker_attempt_slot,
        args.worker_boot_id,
        args.worker_private_key,
        args.worker_authorization,
    )
    if args.worker_arm:
        if any(worker_origin_values) and not all(worker_origin_values):
            parser.error("worker origin arguments must be supplied together")
    elif any(worker_origin_values):
        parser.error("worker origin arguments are reserved for isolated workers")
    if args.worker_arm:
        if args.seed_count <= 0 or not 1 <= args.seed_entropy_bits <= 63:
            parser.error(
                "worker process requires public seed count and entropy bounds"
            )
    elif args.seed_count != 0 or args.seed_entropy_bits != 0:
        parser.error(
            "--seed-count/--seed-entropy-bits are reserved for isolated workers"
        )
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    args.campaign_dir = str(campaign_dir)
    if args.worker_private_key:
        args.worker_private_key = str(
            Path(args.worker_private_key).expanduser().resolve(strict=True)
        )
        args.worker_authorization = str(
            Path(args.worker_authorization).expanduser().resolve(strict=True)
        )
    if args.contamination_audit:
        args.contamination_audit = str(
            Path(args.contamination_audit).expanduser().resolve(strict=True)
        )
        if not args.contamination_trust_root:
            parser.error(
                "--contamination-trust-root is required with --contamination-audit"
            )
        args.contamination_trust_root = str(
            Path(args.contamination_trust_root).expanduser().resolve(strict=True)
        )
    elif args.contamination_trust_root:
        parser.error("--contamination-trust-root requires --contamination-audit")
    trust_paths = (
        args.campaign_trust_policy,
        args.campaign_trust_root,
        args.task_issuer_attestation,
        args.runner_attestation,
    )
    if any(trust_paths) and not (
        args.campaign_trust_policy and args.campaign_trust_root
    ):
        parser.error(
            "--campaign-trust-policy and --campaign-trust-root are required together"
        )
    if args.confirmatory and not args.prepare_trust:
        if not args.contamination_audit:
            parser.error("--confirmatory requires --contamination-audit")
        if not all(trust_paths):
            parser.error(
                "--confirmatory requires the campaign trust policy, independent "
                "root, task issuer attestation, and runner attestation"
            )
    for attribute in (
        "campaign_trust_policy",
        "campaign_trust_root",
        "task_issuer_attestation",
        "runner_attestation",
        "answer_reveal_attestation",
        "final_run_attestation",
    ):
        value = str(getattr(args, attribute, "") or "").strip()
        if value:
            setattr(args, attribute, str(Path(value).expanduser().resolve(strict=True)))
    if args.prepare_trust:
        if not args.contamination_audit:
            parser.error("--prepare-trust requires --contamination-audit")
        if not args.campaign_trust_policy or not args.campaign_trust_root:
            parser.error(
                "--prepare-trust requires campaign trust policy and independent root"
            )
        if args.worker_arm:
            parser.error("--prepare-trust cannot be combined with --worker-arm")
        print(json.dumps(_prepare_trust_requests(args), indent=2, sort_keys=True))
        return 0
    campaign_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_arm:
        persisted = _load_persisted_plan(campaign_dir)
        expected, public_tasks = _expected_worker_plan(args, persisted)
        if persisted.to_dict() != expected.to_dict():
            raise CampaignProducerError(
                "persisted plan does not match requested worker campaign"
            )
        return _execute_worker(args, persisted, public_tasks)
    expected, tasks = _expected_plan(args)
    _persist_plan(campaign_dir, expected)
    persisted = _load_persisted_plan(campaign_dir)
    if persisted.to_dict() != expected.to_dict():
        raise CampaignProducerError("persisted plan does not match requested campaign")
    if args.plan_only:
        document = expected.to_dict()
        print(
            json.dumps(
                {
                    "plan_sha256": expected.plan_sha256,
                    "plan_path": str(campaign_dir / PLAN_FILE),
                    "cell_count": len(expected.cell_ids),
                    "task_count": document["metadata"]["task_manifest"]["task_count"],
                    "arms": document["metadata"]["arms"],
                    "claim_eligible": document["metadata"]["claim_eligible"],
                    "external_frontier_claim_eligible": False,
                    "detached_broker_policy": _detached_broker_policy(args),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    return _orchestrate(args, persisted, tasks)


if __name__ == "__main__":
    raise SystemExit(main())
