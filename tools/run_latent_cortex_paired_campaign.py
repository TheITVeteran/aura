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
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex import (  # noqa: E402
    recurrent_grpo_adapter_identity,  # noqa: E402
    resident_recurrent_sft_adapter_identity,
)
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
from core.brain.llm.latent_cortex.detached_campaign_evidence import (  # noqa: E402
    DetachedCampaignEvidenceError,
    VerifiedDetachedBrokerEvidence,
    verify_detached_broker_evidence,
)
from core.brain.llm.latent_cortex.exact_paired_grade import (  # noqa: E402
    exact_campaign_power_plan,
    exact_group_sequential_power_plan,
)
from core.brain.llm.latent_cortex.exact_paired_statistics import Rational  # noqa: E402
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
    WORKER_ORIGIN_PROTOCOL,
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
from core.brain.llm.latent_cortex.resident_adapter_loader import (  # noqa: E402
    ResidentAdapterLoadError,
    load_resident_adapter,
    resolve_resident_adapter_projection,
)
from core.brain.llm.latent_cortex.runtime_identity import (  # noqa: E402
    build_worker_identity,
    logical_model_parameter_count,
)
from core.brain.llm.latent_cortex.sequential_campaign_evidence import (  # noqa: E402
    SequentialCampaignEvidenceError,
    sequential_task_look_assignments,
)
from core.brain.llm.latent_cortex.worker_attempt_import import (  # noqa: E402
    PAIRED_CAMPAIGN_CELL_TYPE,
    import_verified_worker_stage,
    verify_terminal_worker_stage,
)
from core.runtime.detached_subprocess_broker import (  # noqa: E402
    BrokeredProcessResult,
    DetachedBrokerError,
    broker_available,
    run_brokered_process,
)
from core.runtime.detached_worker_origin_channel import (  # noqa: E402
    DetachedWorkerOriginChannelClient,
)
from tools.resident_recurrent_sft_bootstrap_identity import (  # noqa: E402
    absent_personality_identity,
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
WORKER_ATTEMPT_DIR = "worker_attempts"
WORKER_EXECUTION_MANIFEST_FILE = "worker_execution_manifest.json"
OBJECTIVE_SOURCE = REPO_ROOT / "core/learning/recurrence_native_objective.py"
V2_MANIFEST_FILE = "recurrence_adapter_manifest.json"
CONTAMINATION_AUDIT_SCHEMA = "aura.latent_cortex.contamination_audit.v2"
TASK_ISSUER_PAYLOAD_SCHEMA = "aura.latent_cortex.task_issuer_prelaunch.v1"
CAMPAIGN_RUNNER_PAYLOAD_SCHEMA = "aura.latent_cortex.runner_prelaunch.v1"
SEALED_OUTPUT_MANIFEST_SCHEMA = "aura.latent_cortex.sealed_output_manifest.v4"
ANSWER_REVEAL_PAYLOAD_SCHEMA = "aura.latent_cortex.answer_reveal_payload.v1"
FINAL_RUN_PAYLOAD_SCHEMA = "aura.latent_cortex.final_run_payload.v4"
WORKER_EXECUTION_MANIFEST_SCHEMA = "aura.latent_cortex.worker_execution_manifest.v1"
DETACHED_RUN_DIR_ENV = "AURA_DETACHED_RUN_DIR"
DETACHED_PLAN_PATH_ENV = "AURA_DETACHED_PLAN_PATH"
DETACHED_ATTEMPTS_PATH_ENV = "AURA_DETACHED_ATTEMPTS_PATH"
DETACHED_PLAN_SHA256_ENV = "AURA_DETACHED_PLAN_SHA256"
DETACHED_SUPERVISOR_ATTEMPT_ENV = "AURA_DETACHED_SUPERVISOR_ATTEMPT"
RLC_MECHANISM_PROFILES = (
    "recurrence_attribution",
    "resident_full_stack",
    "resident_full_stack_no_latent_opt",
    "resident_full_stack_no_fast_weights",
    "resident_full_stack_no_branch_exchange",
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
    if not values or len(set(values)) != len(values) or any(value < 0 for value in values):
        parser.error(f"{role} must contain unique non-negative integers")
    return values


def _csv_rationals(
    parser: argparse.ArgumentParser,
    raw: str,
    role: str,
) -> tuple[Rational, ...]:
    values: list[Rational] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split("/")
        if len(parts) != 2:
            parser.error(f"{role} values must use numerator/denominator syntax")
        try:
            values.append(Rational(int(parts[0]), int(parts[1])))
        except (TypeError, ValueError) as exc:
            parser.error(f"{role} contains an invalid rational: {token!r}")
            raise AssertionError from exc
    if not values:
        parser.error(f"{role} must contain at least one rational")
    return tuple(values)


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
                raise CampaignProducerError(f"concurrent artifact differs: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _worker_slot_stem(arm: str, attempt_slot: int) -> str:
    if arm not in FULL_ARMS:
        raise CampaignProducerError("worker authorization arm is invalid")
    if isinstance(attempt_slot, bool) or not isinstance(attempt_slot, int) or attempt_slot <= 0:
        raise CampaignProducerError("worker authorization attempt slot is invalid")
    return f"{arm}.attempt-{attempt_slot:02d}"


def _secure_worker_attempt_dir(
    campaign_dir: Path,
    arm: str,
    attempt_slot: int,
) -> Path:
    root = campaign_dir / WORKER_ATTEMPT_DIR
    for path in (root, root / _worker_slot_stem(arm, attempt_slot)):
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
            raise CampaignProducerError("worker attempt directory is not private")
    return root / _worker_slot_stem(arm, attempt_slot)


def _worker_attempt_paths(
    campaign_dir: Path,
    arm: str,
    attempt_slot: int,
) -> dict[str, Path]:
    root = _secure_worker_attempt_dir(campaign_dir, arm, attempt_slot)
    return {
        "root": root,
        "stage": root / "stage.jsonl",
        "origin_dir": root / "supervisor-origin",
        "broker_result": root / "broker-result.json",
        "verified_stage": root / "verified-stage.json",
        "import_intent": root / "import-intent.json",
        "import_receipt": root / "import-receipt.json",
    }


def _runtime_bundle_identity(
    model_path: Path,
    *,
    weight_identity: dict[str, Any],
) -> dict[str, Any]:
    behavior_files: list[dict[str, Any]] = []
    for path in sorted(model_path.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name == "README.md" or path.suffix == ".safetensors":
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
    artifacts["training_completion.json"] = _read_stable_bytes(completion, max_bytes=1024 * 1024)
    return artifacts


def _recurrent_grpo_artifacts(adapter_dir: Path, manifest: dict[str, Any]) -> dict[str, bytes]:
    artifacts: dict[str, bytes] = {}
    for _role, binding in recurrent_grpo_adapter_identity.declared_bindings(manifest):
        path = _contained_adapter_artifact(adapter_dir, binding["path"])
        artifacts[binding["path"]] = _read_stable_bytes(path, max_bytes=int(binding["size_bytes"]))
    completion = _contained_adapter_artifact(adapter_dir, "training_completion.json")
    artifacts["training_completion.json"] = _read_stable_bytes(completion, max_bytes=1024 * 1024)
    return artifacts


def _resident_recurrent_sft_artifacts(
    adapter_dir: Path, manifest: dict[str, Any]
) -> dict[str, bytes]:
    artifacts: dict[str, bytes] = {}
    for _role, binding in resident_recurrent_sft_adapter_identity.declared_bindings(manifest):
        path = _contained_adapter_artifact(adapter_dir, binding["path"])
        artifacts[binding["path"]] = _read_stable_bytes(path, max_bytes=int(binding["size_bytes"]))
    completion = _contained_adapter_artifact(adapter_dir, "training_completion.json")
    artifacts["training_completion.json"] = _read_stable_bytes(completion, max_bytes=1024 * 1024)
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
    adapter_manifest: dict[str, Any] | None,
) -> str | None:
    requested = str(getattr(args, "personality_adapter", "trained") or "trained").strip()
    lowered = requested.lower()
    if lowered == "trained":
        if adapter_manifest is None:
            return None
        if adapter_manifest.get("schema") == MANIFEST_SCHEMA_V2:
            configured = str(
                _v2_training_config(adapter_dir, adapter_manifest).get(
                    "personality_adapter_path", ""
                )
            ).strip()
        elif adapter_manifest.get("schema") == recurrent_grpo_adapter_identity.MANIFEST_SCHEMA:
            personality = adapter_manifest.get("personality_adapter")
            if not isinstance(personality, Mapping):
                raise CampaignProducerError("recurrent GRPO personality binding is invalid")
            if personality.get("present") is True:
                raise CampaignProducerError(
                    "recurrent GRPO bundle does not carry a loadable personality path"
                )
            configured = "none"
        elif adapter_manifest.get("schema") in (
            resident_recurrent_sft_adapter_identity.MANIFEST_SCHEMAS
        ):
            personality = adapter_manifest.get("personality_adapter")
            if not isinstance(personality, Mapping):
                raise CampaignProducerError("resident recurrent-SFT personality binding is invalid")
            if personality.get("present") is True:
                raise CampaignProducerError(
                    "resident recurrent-SFT package does not carry a loadable personality path"
                )
            configured = "none"
        else:
            raise CampaignProducerError("adapter manifest schema is unsupported")
        requested = configured or "none"
        lowered = requested.lower()
    if lowered == "none":
        return None
    if lowered == "auto":
        from core.brain.llm.model_registry import resolve_personality_adapter

        resolved = resolve_personality_adapter(str(model_path), backend="mlx")
        return str(Path(resolved).expanduser().resolve(strict=True)) if resolved else None
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


def _validate_recurrent_grpo_adapter_dir(
    adapter_dir: Path,
    manifest_bytes: bytes,
    *,
    adapter_id: str,
    base_checkpoint: Mapping[str, Any],
    model_behavior_bundle: Mapping[str, Any],
    personality_identity: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = strict_json_loads(manifest_bytes, role="campaign_recurrent_grpo_manifest")
    artifacts = _recurrent_grpo_artifacts(adapter_dir, manifest)
    adapter_binding = manifest.get("adapter")
    if not isinstance(adapter_binding, dict):
        raise CampaignProducerError("recurrent GRPO adapter binding is invalid")
    adapter_path = _contained_adapter_artifact(adapter_dir, adapter_binding.get("path"))
    receipt = recurrent_grpo_adapter_identity.validate_recurrent_grpo_adapter_identity(
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


def _validate_resident_recurrent_sft_adapter_dir(
    adapter_dir: Path,
    manifest_bytes: bytes,
    *,
    adapter_id: str,
    base_checkpoint: Mapping[str, Any],
    model_behavior_bundle: Mapping[str, Any],
    personality_identity: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = strict_json_loads(manifest_bytes, role="campaign_resident_recurrent_sft_manifest")
    artifacts = _resident_recurrent_sft_artifacts(adapter_dir, manifest)
    adapter_binding = manifest.get("bindings", {}).get("adapter")
    if not isinstance(adapter_binding, dict):
        raise CampaignProducerError("resident recurrent-SFT adapter binding is invalid")
    adapter_path = _contained_adapter_artifact(adapter_dir, adapter_binding.get("path"))
    expected_absent = personality_bundle_identity(None)
    if dict(personality_identity) != expected_absent:
        raise CampaignProducerError(
            "resident recurrent-SFT personality selection must remain absent"
        )
    sft_personality_identity = absent_personality_identity()
    receipt = (
        resident_recurrent_sft_adapter_identity.validate_resident_recurrent_sft_adapter_identity(
            manifest_bytes,
            adapter_id=adapter_id,
            actual_base_checkpoint=base_checkpoint,
            actual_model_behavior_bundle=model_behavior_bundle,
            actual_personality_adapter=sft_personality_identity,
            actual_runtime_environment=runtime_environment,
            artifacts=artifacts,
            tensor_metadata=inspect_mlx_tensor_metadata(adapter_path),
        )
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
    adapter_manifest_bytes = _v2_manifest_bytes(adapter_dir)
    adapter_manifest = (
        strict_json_loads(adapter_manifest_bytes, role="campaign_adapter_manifest")
        if adapter_manifest_bytes is not None
        else None
    )
    personality_path = _resolve_campaign_personality(
        args,
        model_path=model_path,
        adapter_dir=adapter_dir,
        adapter_manifest=adapter_manifest,
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
    if adapter_manifest_bytes is not None:
        schema = adapter_manifest.get("schema") if adapter_manifest is not None else None
        if schema == MANIFEST_SCHEMA_V2:
            manifest, receipt = _validate_v2_adapter_dir(
                adapter_dir,
                adapter_manifest_bytes,
                adapter_id=args.adapter_id,
                base_checkpoint=weight_identity,
                model_behavior_bundle=model_behavior_identity,
                personality_identity=personality_identity,
                runtime_environment=runtime_environment,
            )
        elif schema == recurrent_grpo_adapter_identity.MANIFEST_SCHEMA:
            manifest, receipt = _validate_recurrent_grpo_adapter_dir(
                adapter_dir,
                adapter_manifest_bytes,
                adapter_id=args.adapter_id,
                base_checkpoint=weight_identity,
                model_behavior_bundle=model_behavior_identity,
                personality_identity=personality_identity,
                runtime_environment=runtime_environment,
            )
        elif schema in resident_recurrent_sft_adapter_identity.MANIFEST_SCHEMAS:
            manifest, receipt = _validate_resident_recurrent_sft_adapter_dir(
                adapter_dir,
                adapter_manifest_bytes,
                adapter_id=args.adapter_id,
                base_checkpoint=weight_identity,
                model_behavior_bundle=model_behavior_identity,
                personality_identity=personality_identity,
                runtime_environment=runtime_environment,
            )
        else:
            raise CampaignProducerError("adapter manifest schema is unsupported")
        execution_binding = (
            manifest["bindings"]["execution_spec"]
            if schema in resident_recurrent_sft_adapter_identity.MANIFEST_SCHEMAS
            else manifest["execution_spec"]
        )
        execution_payload = _read_stable_bytes(
            _contained_adapter_artifact(adapter_dir, execution_binding["path"]),
            max_bytes=int(execution_binding["size_bytes"]),
        )
        adapter_identity = {
            "adapter_dir": str(adapter_dir),
            "format": schema,
            "manifest": manifest,
            "identity_receipt": receipt,
            "execution_spec": strict_json_loads(execution_payload, role="campaign_execution_spec"),
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
        training_receipt_bytes=(adapter_dir / manifest.training_receipt.path).read_bytes(),
        tensor_metadata=inspect_mlx_tensor_metadata(adapter_dir / manifest.adapter.path),
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
    if manifest.get("schema") == recurrent_grpo_adapter_identity.MANIFEST_SCHEMA:
        manifest_bytes = _read_stable_bytes(
            adapter_dir / V2_MANIFEST_FILE,
            max_bytes=16 * 1024 * 1024,
        )
        parsed, receipt = _validate_recurrent_grpo_adapter_dir(
            adapter_dir,
            manifest_bytes,
            adapter_id=adapter_id,
            base_checkpoint=base_checkpoint,
            model_behavior_bundle=model_behavior_bundle,
            personality_identity=personality_identity,
            runtime_environment=runtime_environment,
        )
        if parsed != manifest:
            raise CampaignProducerError("recurrent GRPO adapter manifest differs from frozen plan")
        return receipt
    if manifest.get("schema") in resident_recurrent_sft_adapter_identity.MANIFEST_SCHEMAS:
        manifest_bytes = _read_stable_bytes(
            adapter_dir / V2_MANIFEST_FILE,
            max_bytes=16 * 1024 * 1024,
        )
        parsed, receipt = _validate_resident_recurrent_sft_adapter_dir(
            adapter_dir,
            manifest_bytes,
            adapter_id=adapter_id,
            base_checkpoint=base_checkpoint,
            model_behavior_bundle=model_behavior_bundle,
            personality_identity=personality_identity,
            runtime_environment=runtime_environment,
        )
        if parsed != manifest:
            raise CampaignProducerError(
                "resident recurrent-SFT adapter manifest differs from frozen plan"
            )
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
        plan.cell_definition(cell_id).get("task_id") not in task_ids for cell_id in plan.cell_ids
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
        raise CampaignProducerError("contamination audit trust root is invalid") from exc
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
    if manifest.get("schema") in resident_recurrent_sft_adapter_identity.MANIFEST_SCHEMAS:
        receipt = adapter_identity.get("identity_receipt")
        digest = receipt.get("dataset_sha256") if isinstance(receipt, Mapping) else None
        return digest if isinstance(digest, str) and len(digest) == 64 else None
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
    corpus_hashes = {record["snapshot_sha256"] for record in corpora if isinstance(record, dict)}
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
        signature_bytes = base64.b64decode(str(signature["signature_b64"]), validate=True)
    except (KeyError, ValueError) as exc:
        raise CampaignProducerError("contamination audit signature is invalid") from exc
    from cryptography.exceptions import InvalidSignature

    public_key, public_der, trust_root_sha256 = _load_contamination_trust_root(trust_root_path)
    if signature.get("key_id") != trust_root_sha256:
        raise CampaignProducerError("contamination audit signer does not match trust root")
    signed_payload = canonical_json_bytes(body)
    try:
        public_key.verify(signature_bytes, signed_payload)
    except InvalidSignature as exc:
        raise CampaignProducerError("contamination audit signature verification failed") from exc
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
        TASK_ISSUER: REPO_ROOT / "core/brain/llm/latent_cortex/frontier_tasks.py",
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
        raise CampaignProducerError("campaign trust policy and independent root are both required")
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
        generation_config["generation_seeds"] = execution_config["generation_seeds"]
    else:
        generation_config["generation_seed_count"] = execution_config["generation_seed_count"]
        generation_config["generation_seed_min_entropy_bits"] = execution_config[
            "generation_seed_min_entropy_bits"
        ]
        generation_config["generation_seed_policy"] = execution_config["generation_seed_policy"]
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
            "generation_config_sha256": _sha256_bytes(canonical_json_bytes(generation_config)),
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
            "execution_config_sha256": _sha256_bytes(canonical_json_bytes(execution_config)),
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
        if policy.role_pin(role)["implementation_sha256"] != _prelaunch_role_implementation_sha256(
            role
        ):
            raise CampaignProducerError(
                f"{role} implementation does not match the pre-pinned source"
            )
    payloads = _prelaunch_payloads(args, unsigned_plan=unsigned_plan, policy=policy)
    issuer_path = str(getattr(args, "task_issuer_attestation", "") or "").strip()
    runner_path = str(getattr(args, "runner_attestation", "") or "").strip()
    if not issuer_path or not runner_path:
        raise CampaignProducerError(
            "task issuer and campaign runner prelaunch attestations are required"
        )
    admitted_at = int(time.time())
    issuer_attestation = _read_json_artifact(issuer_path, role="task issuer attestation")
    runner_attestation = _read_json_artifact(runner_path, role="campaign runner attestation")
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
        "task_issuer_payload_sha256": _sha256_bytes(canonical_json_bytes(payloads[TASK_ISSUER])),
        "runner_attestation": runner_attestation,
        "runner_payload_sha256": _sha256_bytes(canonical_json_bytes(payloads[CAMPAIGN_RUNNER])),
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

    contract_grace_tokens = min(max(0, int(args.decode_max_tokens)), 512)
    verifier_probe_tokens = min(max(16, int(args.decode_max_tokens)), 192)

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
            decode_contract="final_answer_v1",
            decode_contract_grace_tokens=contract_grace_tokens,
            verifier_probe_max_tokens=verifier_probe_tokens,
            decode_bridge_policy=(
                "assistant_answer_v3" if spec.decode_bridge_policy == "assistant_answer" else "none"
            ),
            decode_repetition_penalty=1.25,
            decode_repetition_window=72,
            allow_vanilla_fallback=False,
            escape={"enabled": False},
        )
    if args.rlc_profile in {
        "resident_full_stack",
        "resident_full_stack_no_latent_opt",
        "resident_full_stack_no_fast_weights",
        "resident_full_stack_no_branch_exchange",
    }:
        branch_exchange_enabled = args.rlc_profile != "resident_full_stack_no_branch_exchange"
        latent_opt_enabled = args.rlc_profile != "resident_full_stack_no_latent_opt"
        fast_weights_enabled = args.rlc_profile != "resident_full_stack_no_fast_weights"
        return CortexConfig(
            workspace=WorkspaceConfig(n_slots=4, seed=0),
            recurrence=RecurrenceConfig(max_steps=2, min_steps=2),
            branches=BranchConfig(
                n_branches=2,
                exchange_interval=1 if branch_exchange_enabled else 999999,
                exchange_gamma=0.35 if branch_exchange_enabled else 0.0,
            ),
            latent_opt=LatentOptConfig(enabled=latent_opt_enabled, steps=1, lr=0.03),
            fast_weights=FastWeightsConfig(
                enabled=fast_weights_enabled,
                rank=2,
                opt_steps=1,
                lr=0.005,
                max_wrapped_layers=2,
                export_candidates=False,
            ),
            decode_max_tokens=args.decode_max_tokens,
            decode_contract="final_answer_v1",
            decode_contract_grace_tokens=contract_grace_tokens,
            decode_min_tokens=min(96, max(0, args.decode_max_tokens - 1)),
            verifier_probe_max_tokens=verifier_probe_tokens,
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
        decode_contract="final_answer_v1",
        decode_contract_grace_tokens=contract_grace_tokens,
        verifier_probe_max_tokens=verifier_probe_tokens,
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
        "generation_seed_count": int(getattr(args, "seed_count", 0) or len(args.seed_values)),
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
        "response_contract_policy": {
            "schema": "public_response_contract_v1",
            "termination": "final_answer_v1",
            "contract_grace_tokens": effective.decode_contract_grace_tokens,
            "verifier_probe_max_tokens": effective.verifier_probe_max_tokens,
            "applies_identically_to_all_decode_arms": True,
            "output_editing": False,
        },
        "episode_timeout_s": args.episode_timeout,
        "load_timeout_s": args.load_timeout,
        "warmup_timeout_s": args.warmup_timeout,
        "arm_timeout_s": args.arm_timeout,
        "campaign_timeout_s": args.campaign_timeout,
        "equal_compute_max_samples": args.equal_compute_max_samples,
        "adapter_process_isolation": True,
        "worker_task_material": "public_manifest_only",
        "answer_reveal_protocol": "sealed_outputs_then_issuer_reveal_v1",
        "worker_origin_protocol": WORKER_ORIGIN_PROTOCOL,
        "worker_origin_attempt_slots": (
            args.max_infra_attempts * len(_campaign_looks(args))
        ),
        "vanilla_fallback_allowed": False,
        "exact_statistical_power": _statistical_power_plan(args),
        **(
            {
                "sequential_look_observations_per_domain": list(
                    args.sequential_look_values
                ),
                "sequential_alpha_weights": [
                    {
                        "numerator": value.numerator,
                        "denominator": value.denominator,
                    }
                    for value in args.sequential_alpha_weight_values
                ],
            }
            if getattr(args, "sequential_look_values", ())
            else {}
        ),
        "implementation_sha256": _implementation_sha256(),
    }


def _statistical_power_plan(args: argparse.Namespace) -> dict[str, Any]:
    arms = _arms(args)
    comparison_count = 4
    if BASE_EQUAL_COMPUTE in arms:
        comparison_count += 1
    if ADAPTER_EQUAL_COMPUTE in arms:
        comparison_count += 1
    planned = int(getattr(args, "seed_count", 0) or len(args.seed_values))
    sequential_looks = tuple(getattr(args, "sequential_look_values", ()))
    sequential_weights = tuple(
        getattr(args, "sequential_alpha_weight_values", ())
    )
    if sequential_looks or sequential_weights:
        return exact_group_sequential_power_plan(
            domain_count=len(args.domain_values),
            comparison_count=comparison_count,
            arm_count=len(arms),
            look_observations_per_domain=sequential_looks,
            alpha_weights=sequential_weights,
        )
    return exact_campaign_power_plan(
        domain_count=len(args.domain_values),
        comparison_count=comparison_count,
        arm_count=len(arms),
        planned_observations_per_domain=planned,
    )


def _campaign_looks(args: argparse.Namespace) -> tuple[int, ...]:
    values = tuple(getattr(args, "sequential_look_values", ()))
    return tuple(range(1, len(values) + 1)) if values else (0,)


def _worker_attempt_slot_range(
    args: argparse.Namespace,
    worker_look: int,
) -> range:
    if worker_look == 0:
        return range(1, args.max_infra_attempts + 1)
    start = (worker_look - 1) * args.max_infra_attempts + 1
    return range(start, start + args.max_infra_attempts)


def _task_look_assignments(plan: CampaignPlan) -> dict[str, int]:
    execution_config = plan.to_dict()["metadata"]["execution_config"]
    if not execution_config.get("sequential_look_observations_per_domain"):
        return {
            task["task_id"]: 0
            for task in plan.to_dict()["metadata"]["task_manifest"]["tasks"]
        }
    try:
        return sequential_task_look_assignments(plan)
    except SequentialCampaignEvidenceError as exc:
        raise CampaignProducerError(str(exc)) from exc


def _arm_cell_ids_for_look(
    plan: CampaignPlan,
    arm: str,
    worker_look: int,
    *,
    cumulative: bool,
) -> set[str]:
    assignments = _task_look_assignments(plan)
    return {
        cell_id
        for cell_id in plan.cell_ids
        if plan.cell_definition(cell_id).get("arm") == arm
        and (
            assignments[plan.cell_definition(cell_id)["task_id"]] == worker_look
            or (
                cumulative
                and worker_look > 0
                and assignments[plan.cell_definition(cell_id)["task_id"]] <= worker_look
            )
        )
    }


def _pending_worker_cell_ids(
    plan: CampaignPlan,
    *,
    arm: str,
    worker_look: int,
    runnable_cell_ids: Sequence[str],
    stage_sealed_cell_ids: set[str],
    canonical_sealed_cell_ids: set[str],
) -> list[str]:
    allowed_cells = _arm_cell_ids_for_look(
        plan,
        arm,
        worker_look,
        cumulative=False,
    )
    pending = [
        cell_id
        for cell_id in runnable_cell_ids
        if cell_id in allowed_cells
        and cell_id not in stage_sealed_cell_ids
        and cell_id not in canonical_sealed_cell_ids
    ]
    pending.sort(
        key=lambda cell_id: int(
            plan.cell_definition(cell_id)["execution_ordinal_within_arm"]
        )
    )
    return pending


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
        expected_training_corpus_sha256=_adapter_dataset_manifest_sha256(adapter_identity),
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
    power = _statistical_power_plan(args)
    powered = power.get("powered_for_zero_loss_noninferiority") is True
    if power.get("schema") == "aura.latent_cortex.exact_group_sequential_power.v1":
        powered = power.get("terminal_look_powered_for_zero_loss_noninferiority") is True
    seed_entropy_bits = int(
        getattr(args, "seed_entropy_bits", 0)
        or min(value.bit_length() for value in args.seed_values)
    )
    runtime_bundle = model_identity.get("runtime_bundle")
    return bool(
        args.confirmatory
        and powered
        and seed_entropy_bits >= 60
        and args.profile == "full"
        and set(args.domain_values) == set(FRONTIER_DOMAINS)
        and isinstance(runtime_bundle, dict)
        and runtime_bundle.get("model_type") == "qwen2"
        and runtime_bundle.get("logical_parameter_count_basis") == "architecture_config_logical"
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


def _load_adapter(model: Any, adapter_dir: Path, manifest: dict[str, Any]) -> int:
    try:
        return load_resident_adapter(model, adapter_dir, manifest)
    except ResidentAdapterLoadError as exc:
        raise CampaignProducerError(exc.code) from exc


def _resolve_projection(model: Any, projection: str) -> tuple[Any, str, Any]:
    """Compatibility boundary for campaign tests and older tooling."""

    try:
        return resolve_resident_adapter_projection(model, projection)
    except ResidentAdapterLoadError as exc:
        detail = {
            "resident_adapter_projection_path_invalid": "path is invalid",
            "resident_adapter_projection_index_invalid": "index is invalid",
            "resident_adapter_projection_owner_missing": "owner is missing",
            "resident_adapter_projection_missing": "projection is missing",
        }.get(exc.code, exc.code)
        raise CampaignProducerError(
            f"adapter projection {detail}: {projection}"
        ) from exc


def _vanilla_once(
    model: Any,
    tokenizer: Any,
    task: PublicTaskRecord,
    *,
    max_tokens: int,
    sample_seed: int | None = None,
    accounting_engine: Any,
) -> tuple[str, int, dict[str, Any], dict[str, Any], float]:
    import mlx.core as mx
    from mlx_lm import stream_generate

    from core.brain.llm.latent_cortex.answer_contract import (
        ContractDecodeDisposition,
        contract_decode_disposition,
    )

    kwargs: dict[str, Any] = {}
    if sample_seed is not None:
        from mlx_lm.sample_utils import make_sampler

        mx.random.seed(sample_seed)
        kwargs["sampler"] = make_sampler(temp=0.7, top_p=0.95)
    prompt_token_ids = accounting_engine._encode(
        None,
        [{"role": "user", "content": task.prompt}],
        None,
    )
    # CP180: uniform contract-aware stop for EVERY arm that decodes through
    # this path — the moment one FINAL_ANSWER JSON object completes, more
    # tokens can only break terminality. Within the same hard cap, budget
    # goes to reasoning instead of post-answer babble, and the cap stops
    # truncating answers that finish honestly.
    pieces: list[str] = []
    generated_tokens = 0
    contract_grace_tokens = min(max(0, int(max_tokens)), 512)
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt_token_ids,
        max_tokens=max_tokens + contract_grace_tokens,
        **kwargs,
    ):
        pieces.append(response.text)
        generated_tokens = int(response.generation_tokens)
        disposition = contract_decode_disposition("".join(pieces))
        if disposition in {
            ContractDecodeDisposition.COMPLETE,
            ContractDecodeDisposition.INVALID,
        }:
            break
    text = "".join(pieces)
    prompt_tokens = len(prompt_token_ids)
    output_tokens = max(1, generated_tokens)
    n_layers = len(model.model.layers)
    decode_forwards = max(0, output_tokens - 1)
    layer_apps = (prompt_tokens + decode_forwards) * n_layers

    from core.brain.llm.latent_cortex.resource_accounting import (
        ModelComputeProfile,
        ResourceLedger,
        triangular_attention_pairs,
    )
    from core.brain.llm.latent_cortex.task_verifiers import EpisodeTaskVerifier
    from core.brain.llm.latent_cortex.value_of_computation import (
        build_evidence_snapshot,
    )

    profile = ModelComputeProfile.from_model(model)
    ledger = ResourceLedger(profile)
    ledger.charge(
        "vanilla_prefill",
        transformer_layer_apps=prompt_tokens * n_layers,
        attention_query_key_pairs=(triangular_attention_pairs(prompt_tokens) * n_layers),
        output_head_tokens=1,
    )
    decode_pairs = sum(prompt_tokens + index + 1 for index in range(decode_forwards))
    ledger.charge(
        "vanilla_decode",
        transformer_layer_apps=decode_forwards * n_layers,
        attention_query_key_pairs=decode_pairs * n_layers,
        output_head_tokens=decode_forwards,
        tensor_element_reads=output_tokens * profile.vocab_size,
        tensor_element_writes=output_tokens * profile.vocab_size,
        host_scalar_ops=output_tokens * profile.vocab_size * 8,
    )
    verifier = EpisodeTaskVerifier(
        task.prompt,
        response_contract=task.response_contract,
    )
    verifier_score = float(verifier(text))
    ledger.charge(
        "task_verifier",
        verifier_calls=1,
        verifier_input_bytes=len(text.encode("utf-8")),
        verifier_output_bytes=len(repr(verifier_score).encode("ascii")),
        host_scalar_ops=max(1, len(text)),
    )
    encoded_tokens = json.dumps(
        prompt_token_ids,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    information = accounting_engine._information_receipt(
        encoded_tokens=encoded_tokens,
        token_count=len(prompt_token_ids),
        context_items=[],
        policy_evidence=build_evidence_snapshot(
            bucket=f"{str(task.domain or 'general')[:24]}|none|short|s:mid|u:mid",
            cells={},
        ),
        verifier=verifier,
    )
    return text, layer_apps, ledger.to_receipt(), information, verifier_score


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
    accounting_engine: Any,
    target_resource: Mapping[str, Any],
) -> tuple[str, int, int, dict[str, Any], dict[str, Any]]:
    from core.brain.llm.latent_cortex.resource_accounting import (
        NON_NEURAL_PARITY_COUNTERS,
        ResourceLedger,
        validate_resource_receipt,
    )

    try:
        target_resource = validate_resource_receipt(target_resource)
    except (TypeError, ValueError) as exc:
        raise CampaignProducerError("equal-compute target accounting is invalid") from exc
    if target_resource["accounting_complete"] is not True:
        raise CampaignProducerError("equal-compute target accounting is incomplete")

    outputs: list[str] = []
    scores: list[float] = []
    resources: list[dict[str, Any]] = []
    information: dict[str, Any] | None = None
    spent = 0
    target_reached = False
    seed_base = int(task.task_payload_sha256[:16], 16)
    for sample_index in range(max_samples):
        text, cost, resource, sample_information, verifier_score = _vanilla_once(
            model,
            tokenizer,
            task,
            max_tokens=max_tokens,
            sample_seed=(seed_base + sample_index) % (2**31 - 1),
            accounting_engine=accounting_engine,
        )
        outputs.append(text)
        scores.append(verifier_score)
        resources.append(resource)
        if information is None:
            information = sample_information
        elif information != sample_information:
            raise CampaignProducerError("equal-compute information envelope drifted")
        spent += cost
        aggregate = ResourceLedger.aggregate(resources).to_receipt()
        target_flops = int(target_resource.get("estimated_flops") or 0)
        if (
            spent >= target_layer_apps
            and int(aggregate.get("estimated_flops") or 0) >= target_flops
            and all(
                aggregate["totals"][name] >= target_resource["totals"][name]
                for name in NON_NEURAL_PARITY_COUNTERS
            )
        ):
            target_reached = True
            break
    if not outputs or information is None:
        raise CampaignProducerError("equal-compute control produced no samples")
    if not target_reached:
        raise CampaignProducerError(
            "equal-compute control exhausted its sample bound below the target"
        )
    best_score = max(scores)
    eligible = [
        output for output, score in zip(outputs, scores, strict=True) if score == best_score
    ]
    return (
        _majority_output(eligible),
        spent,
        len(outputs),
        ResourceLedger.aggregate(resources).to_receipt(),
        information,
    )


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
) -> tuple[str, int, dict[str, Any], dict[str, Any], dict[str, Any]]:
    from core.brain.llm.latent_cortex.resource_accounting import (
        validate_information_receipt,
        validate_resource_receipt,
    )
    from core.brain.llm.latent_cortex.task_verifiers import EpisodeTaskVerifier
    from core.brain.llm.latent_cortex.types import ComputeBudget

    verifier = EpisodeTaskVerifier(
        task.prompt,
        response_contract=task.response_contract,
    )
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
    scored_policy_failure = bool(
        not result.ok
        and (
            result.reason == "answer_replacement_abstained"
            or result.reason.startswith("decode_incomplete:")
        )
    )
    if not result.ok and not scored_policy_failure:
        raise CampaignProducerError(f"latent episode failed: {result.reason}")
    try:
        resource = validate_resource_receipt(receipt["budget"]["resource_accounting"])
        information = validate_information_receipt(receipt["budget"]["information_accounting"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CampaignProducerError("latent episode accounting is invalid") from exc
    if (
        resource["accounting_complete"] is not True
        or information["accounting_complete"] is not True
    ):
        raise CampaignProducerError("latent episode accounting is incomplete")
    return (
        result.text,
        budget.spent_layer_apps,
        receipt,
        resource,
        information,
    )


def _prior_rlc_costs(
    journal: CampaignJournal,
) -> dict[tuple[str, str], tuple[int, dict[str, Any]]]:
    from core.brain.llm.latent_cortex.resource_accounting import (
        validate_resource_receipt,
    )

    costs: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    for record in journal.result_records():
        definition = record["definition"]
        arm = definition["arm"]
        if arm in {BASE_RLC, ADAPTER_RLC}:
            try:
                resource = validate_resource_receipt(record["result"].get("resource_accounting"))
            except (TypeError, ValueError) as exc:
                raise CampaignProducerError(
                    "RLC prerequisite lacks valid resource accounting"
                ) from exc
            if resource["accounting_complete"] is not True:
                raise CampaignProducerError("RLC prerequisite resource accounting is incomplete")
            costs[(definition["task_id"], arm)] = (
                int(record["result"]["layer_apps"]),
                resource,
            )
    return costs


def _worker_origin_context(
    args: argparse.Namespace,
    plan: CampaignPlan,
) -> dict[str, Any] | None:
    claim_required = plan.to_dict()["metadata"].get("claim_eligible") is True
    stage_value = str(getattr(args, "worker_stage_journal", "") or "")
    supplied = bool(args.worker_attempt_slot and stage_value)
    if not claim_required:
        if supplied:
            raise CampaignProducerError("preflight worker received claim-only origin credentials")
        return None
    if not supplied:
        raise CampaignProducerError("claim worker stage and origin channel are required")
    if args.worker_attempt_slot not in _worker_attempt_slot_range(
        args,
        int(getattr(args, "worker_look", 0) or 0),
    ):
        raise CampaignProducerError("worker attempt slot is outside its frozen look")
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    paths = _worker_attempt_paths(
        campaign_dir,
        args.worker_arm,
        args.worker_attempt_slot,
    )
    if Path(stage_value).expanduser().resolve(strict=False) != paths["stage"]:
        raise CampaignProducerError("worker stage path substitution")
    if paths["stage"].exists() or paths["stage"].is_symlink():
        raise CampaignProducerError("worker stage attempt was already consumed")
    client = DetachedWorkerOriginChannelClient.from_environment()
    return {
        "client": client,
        "paths": paths,
    }


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

    cleanup = contextlib.ExitStack()
    if origin_context is not None:
        cleanup.callback(origin_context["client"].close)
    cleanup.enter_context(
        standalone_model_lane(
            owner_id=f"rlc-paired:{arm}:{os.getpid()}",
            model_path=model_path,
            purpose="benchmark",
            preemptible=False,
            metadata={"tool": "run_latent_cortex_paired_campaign", "arm": arm},
        )
    )
    with cleanup:
        load_started = time.monotonic()
        with _deadline_alarm(args.load_timeout, "model_load"):
            planned_model = metadata["model_identity"]
            planned_runtime = planned_model["runtime_bundle"]
            personality_path = str(planned_model.get("personality_adapter_path") or "") or None
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
            pre_load_boundary = _model_load_boundary_identity(model_dir, personality_path)
            if pre_load_boundary != planned_load_boundary:
                raise CampaignProducerError("model bytes differ from frozen plan before load")
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
                if actual_adapter_identity != metadata["adapter_identity"]["identity_receipt"]:
                    raise CampaignProducerError("adapter bytes differ from frozen plan before load")
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
                    raise CampaignProducerError("adapter identity changed across load boundary")
            post_load_boundary = _model_load_boundary_identity(model_dir, personality_path)
            if post_load_boundary != pre_load_boundary:
                raise CampaignProducerError("model identity changed across load boundary")
            from core.brain.llm.latent_cortex.worker_capture_identity import (
                build_worker_capture_identity,
            )

            worker_capture_signing_identity = build_worker_capture_identity(
                worker_boot_id=(
                    origin_context["client"].session_id
                    if origin_context is not None
                    else uuid.uuid4().hex
                ),
            )
            worker_identity = build_worker_identity(
                model,
                model_path=model_path,
                worker_boot_id=worker_capture_signing_identity.public_identity["worker_boot_id"],
                worker_source_path=Path(__file__).resolve(),
                worker_action_capture_identity=(worker_capture_signing_identity.public_identity),
                tokenizer=tokenizer,
            )
            worker_identity.update(
                {
                    "worker_weight_fingerprint": post_load_boundary["weight_fingerprint"],
                    "worker_weight_fingerprint_method": post_load_boundary["weight_method"],
                    "worker_weight_file_count": post_load_boundary["weight_file_count"],
                    "worker_runtime_bundle_sha256": post_load_boundary["runtime_bundle_sha256"],
                    "worker_personality_adapter": post_load_boundary["personality_adapter"],
                    "worker_effective_stack_sha256": post_load_boundary["effective_stack_sha256"],
                    "worker_load_boundary_verified": True,
                }
            )
            if (
                worker_identity["worker_model_parameter_count"]
                != planned_runtime["logical_parameter_count"]
                or worker_identity["worker_model_parameter_count_basis"]
                != planned_runtime["logical_parameter_count_basis"]
            ):
                raise CampaignProducerError("loaded model identity differs from frozen plan")
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
        execution_spec = raw_execution_spec if isinstance(raw_execution_spec, Mapping) else None
        # Every arm receives the same bound engine configuration so resource
        # profiles and information-policy commitments are reconstructed from
        # the identical resident model/runtime.  Non-RLC arms use it only for
        # claim accounting; they do not execute latent recurrence.
        accounting_engine = _make_rlc_engine(
            model,
            tokenizer,
            args,
            execution_spec,
        )
        rlc_engine = accounting_engine if arm.endswith("_rlc") else None
        campaign_dir = Path(args.campaign_dir).expanduser().resolve()
        canonical_journal_path = campaign_dir / JOURNAL_FILE
        with CampaignJournal(canonical_journal_path, plan) as canonical:
            costs = _prior_rlc_costs(canonical)
            canonical_sealed = set(canonical.resume().sealed_cell_ids)
        worker_journal_path = (
            origin_context["paths"]["stage"]
            if origin_context is not None
            else canonical_journal_path
        )
        with CampaignJournal(worker_journal_path, plan) as journal:
            sealed = set(journal.resume().sealed_cell_ids)
            pending = _pending_worker_cell_ids(
                plan,
                arm=arm,
                worker_look=args.worker_look,
                runnable_cell_ids=journal.resume().runnable_cell_ids,
                stage_sealed_cell_ids=sealed,
                canonical_sealed_cell_ids=canonical_sealed,
            )
            for cell_id in pending:
                definition = plan.cell_definition(cell_id)
                task = task_by_id[definition["task_id"]]
                attempt_id = journal.start_cell(cell_id)
                started = time.monotonic()
                try:
                    with _deadline_alarm(args.episode_timeout, "campaign_cell"):
                        receipt: dict[str, Any] = {}
                        resource_accounting: dict[str, Any]
                        information_accounting: dict[str, Any]
                        arm_verifier_score: float | None = None
                        samples = 1
                        if arm.endswith("_rlc"):
                            if rlc_engine is None:
                                raise CampaignProducerError("RLC engine is unavailable")
                            (
                                text,
                                layer_apps,
                                receipt,
                                resource_accounting,
                                information_accounting,
                            ) = _run_rlc(rlc_engine, task, args)
                        elif arm.endswith("_equal_compute"):
                            source_arm = BASE_RLC if arm == BASE_EQUAL_COMPUTE else ADAPTER_RLC
                            target = costs.get((task.task_id, source_arm))
                            if target is None:
                                raise CampaignProducerError("equal-compute prerequisite missing")
                            target_layer_apps, target_resource = target
                            (
                                text,
                                layer_apps,
                                samples,
                                resource_accounting,
                                information_accounting,
                            ) = _equal_compute(
                                model,
                                tokenizer,
                                task,
                                target_layer_apps=target_layer_apps,
                                max_tokens=args.decode_max_tokens,
                                max_samples=args.equal_compute_max_samples,
                                accounting_engine=accounting_engine,
                                target_resource=target_resource,
                            )
                        else:
                            (
                                text,
                                layer_apps,
                                resource_accounting,
                                information_accounting,
                                arm_verifier_score,
                            ) = _vanilla_once(
                                model,
                                tokenizer,
                                task,
                                max_tokens=args.decode_max_tokens,
                                accounting_engine=accounting_engine,
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
                        "adapter_identity_sha256": metadata["adapter_identity"]["identity_receipt"][
                            "composite_identity_sha256"
                        ]
                        if arm.startswith("adapter_")
                        else None,
                        "runtime_adapter_identity": actual_adapter_identity,
                        "runtime_model_identity": worker_identity,
                        "episode_receipt": receipt,
                        "resource_accounting": resource_accounting,
                        "information_accounting": information_accounting,
                        "arm_verifier_score": arm_verifier_score,
                    }
                    if origin_context is not None:
                        result = origin_context["client"].record_result(
                            result,
                            cell_id=cell_id,
                            cell_type=PAIRED_CAMPAIGN_CELL_TYPE,
                            attempt_id=attempt_id,
                        )
                    journal.record_arm_result(cell_id, attempt_id, result)
                    if origin_context is not None:
                        journal.record_verified(
                            cell_id,
                            attempt_id,
                            {
                                "schema": "aura.latent_cortex.worker_stage_transport_verification.v1",
                                "result_origin_sha256": result["worker_origin"]["origin_sha256"],
                            },
                        )
                        journal.commit_cell(
                            cell_id,
                            attempt_id,
                            {
                                "schema": "aura.latent_cortex.worker_stage_transport_commit.v1",
                                "result_sha256": _sha256_bytes(canonical_json_bytes(result)),
                            },
                        )
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
    worker_look: int = 0,
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
        "--sequential-look-observations",
        str(getattr(args, "sequential_look_observations", "") or ""),
        "--sequential-alpha-weights",
        str(getattr(args, "sequential_alpha_weights", "") or ""),
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
        "--worker-look",
        str(worker_look),
    ]
    if worker_attempt_slot is not None:
        campaign_dir = Path(args.campaign_dir).expanduser().resolve()
        paths = _worker_attempt_paths(campaign_dir, arm, worker_attempt_slot)
        command.extend(
            [
                "--worker-attempt-slot",
                str(worker_attempt_slot),
                "--worker-stage-journal",
                str(paths["stage"]),
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
    command.extend(["--worker-arm", arm])
    return command


def _arm_outputs_sealed(
    campaign_dir: Path,
    plan: CampaignPlan,
    arm: str,
    *,
    worker_look: int = 0,
) -> bool:
    with CampaignJournal(campaign_dir / JOURNAL_FILE, plan) as journal:
        sealed = set(journal.resume().sealed_cell_ids)
    expected = _arm_cell_ids_for_look(
        plan,
        arm,
        worker_look,
        cumulative=True,
    )
    return expected.issubset(sealed)


def _seal_output_manifest(
    campaign_dir: Path,
    plan: CampaignPlan,
    *,
    worker_execution: Mapping[str, Any] | None = None,
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
            "arm_result_event_sha256": by_cell[cell_id]["arm_result_event_sha256"],
            "result_sha256": _sha256_bytes(canonical_json_bytes(by_cell[cell_id]["result"])),
        }
        for cell_id in plan.cell_ids
    ]
    material = {
        "schema": SEALED_OUTPUT_MANIFEST_SCHEMA,
        "plan_sha256": plan.plan_sha256,
        "cell_count": len(cells),
        "cells": cells,
    }
    if worker_execution is not None:
        material.update(
            {
                "worker_execution_manifest_sha256": worker_execution["manifest_sha256"],
                "detached_plan_sha256": worker_execution["detached_plan_sha256"],
                "detached_classification_head_sha256": worker_execution[
                    "detached_classification_head_sha256"
                ],
                "detached_classifications_sha256": worker_execution[
                    "detached_classifications_sha256"
                ],
                "worker_imports_sha256": worker_execution["imports_sha256"],
                "worker_excluded_attempts_sha256": worker_execution["excluded_attempts_sha256"],
            }
        )
    manifest = {
        **material,
        "manifest_sha256": _sha256_bytes(canonical_json_bytes(material)),
    }
    _atomic_create_or_verify(
        campaign_dir / SEALED_OUTPUT_MANIFEST_FILE,
        canonical_json_bytes(manifest) + b"\n",
    )
    return manifest


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
        if _sha256_bytes(canonical_json_bytes(payload)) != public["answer_commitment_sha256"]:
            raise CampaignProducerError("answer reveal differs from prelaunch commitment")
        answers.append(
            {
                "task_id": task.task_id,
                "answer_commitment_sha256": public["answer_commitment_sha256"],
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
            signed_payload.get("signed_at_unix") if isinstance(signed_payload, dict) else None
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
            raise CampaignProducerError(f"{role} request differs from the current payload")
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
    if not isinstance(document, dict) or payload != canonical_json_bytes(document) + b"\n":
        raise CampaignProducerError(f"{role} is not canonical JSON")
    return document


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
        attestation_path = str(getattr(args, "answer_reveal_attestation", "") or "").strip()
        if not attestation_path:
            print(
                json.dumps(
                    {
                        "state": "answer_reveal_signature_required",
                        "request_path": str(campaign_dir / ANSWER_REVEAL_REQUEST_FILE),
                        "request_sha256": request_sha256,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
            return None
        attestation = _read_json_artifact(attestation_path, role="answer reveal attestation")
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
                "answer_commitment_sha256": (task.public.answer_commitment_sha256),
            }
            if state == "ARM_RESULT":
                journal.record_verified(record["cell_id"], record["attempt_id"], verification)
            elif state == "VERIFIED" and record["verification"] != verification:
                raise CampaignProducerError("persisted verification differs from post-seal scoring")
            elif state != "VERIFIED":
                raise CampaignProducerError("sealed output state is invalid")
            journal.commit_cell(
                record["cell_id"],
                record["attempt_id"],
                {
                    "result_sha256": _sha256_bytes(canonical_json_bytes(record["result"])),
                    "verification_sha256": _sha256_bytes(canonical_json_bytes(verification)),
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
    worker_execution: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if plan.to_dict()["metadata"].get("claim_eligible") is not True:
        return {
            "schema": "aura.latent_cortex.final_run_envelope.v4",
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
    if worker_execution is None:
        raise CampaignProducerError("final run is missing worker execution evidence")
    if worker_execution != _read_canonical_json_artifact(
        campaign_dir / WORKER_EXECUTION_MANIFEST_FILE,
        role="worker execution manifest",
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
        "worker_execution_manifest_sha256": worker_execution["manifest_sha256"],
        "detached_plan_sha256": worker_execution["detached_plan_sha256"],
        "detached_classification_head_sha256": worker_execution[
            "detached_classification_head_sha256"
        ],
        "detached_classifications_sha256": worker_execution["detached_classifications_sha256"],
        "worker_imports_sha256": worker_execution["imports_sha256"],
        "worker_excluded_attempts_sha256": worker_execution["excluded_attempts_sha256"],
    }
    request = _load_or_prepare_role_request(
        campaign_dir / FINAL_RUN_REQUEST_FILE,
        policy=policy,
        role=CAMPAIGN_RUNNER,
        payload=payload,
    )
    attestation_path = str(getattr(args, "final_run_attestation", "") or "").strip()
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
    attestation = _read_json_artifact(attestation_path, role="final run attestation")
    signed = verify_role_attestation(
        policy,
        attestation,
        role=CAMPAIGN_RUNNER,
        expected_payload=payload,
        not_before_unix=request["signed_payload"]["signed_at_unix"],
    )
    if signed != request["signed_payload"]:
        raise CampaignProducerError("final run attestation does not sign the issued request")
    material = {
        "payload": payload,
        "request_sha256": request["request_sha256"],
        "campaign_runner_attestation": attestation,
    }
    envelope = {
        "schema": "aura.latent_cortex.final_run_envelope.v4",
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
    minimum: int = 1,
    maximum: int,
) -> int | None:
    if type(minimum) is not int or minimum <= 0 or minimum > maximum:
        raise CampaignProducerError("worker attempt slot range is invalid")
    for attempt_slot in range(minimum, maximum + 1):
        paths = _worker_attempt_paths(campaign_dir, arm, attempt_slot)
        lifecycle_exists = paths["origin_dir"].is_dir() and any(
            paths["origin_dir"].glob("*.lifecycle.json")
        )
        consumed = any(
            path.exists() or path.is_symlink()
            for path in (
                paths["stage"],
                paths["broker_result"],
                paths["import_receipt"],
            )
        )
        if not lifecycle_exists and not consumed:
            return attempt_slot
    return None


def _persist_brokered_worker_result(
    paths: Mapping[str, Path],
    result: BrokeredProcessResult,
) -> dict[str, Any]:
    body = {
        "schema": "aura.latent_cortex.brokered_worker_result.v1",
        **asdict(result),
    }
    artifact = {
        **body,
        "artifact_sha256": _sha256_bytes(canonical_json_bytes(body)),
    }
    _atomic_create_or_verify(
        paths["broker_result"],
        canonical_json_bytes(artifact) + b"\n",
    )
    return artifact


def _load_brokered_worker_result(path: Path) -> tuple[BrokeredProcessResult, dict[str, Any]]:
    artifact = _read_canonical_json_artifact(path, role="brokered worker result")
    body = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    field_names = set(BrokeredProcessResult.__dataclass_fields__)
    if (
        set(artifact) != {"schema", *field_names, "artifact_sha256"}
        or artifact.get("schema") != "aura.latent_cortex.brokered_worker_result.v1"
        or artifact.get("artifact_sha256") != _sha256_bytes(canonical_json_bytes(body))
    ):
        raise CampaignProducerError("brokered worker result artifact is invalid")
    return (
        BrokeredProcessResult(**{field: artifact[field] for field in field_names}),
        artifact,
    )


def _verify_detached_worker_broker_result(
    result: BrokeredProcessResult,
    *,
    require_claim_eligible: bool,
) -> VerifiedDetachedBrokerEvidence:
    run_dir, plan_path, attempts_path, plan_sha256, supervisor_attempt = (
        _detached_evidence_environment()
    )
    try:
        verified = verify_detached_broker_evidence(
            run_dir=run_dir,
            broker_result=result,
            require_claim_eligible=require_claim_eligible,
        )
    except DetachedCampaignEvidenceError as exc:
        raise CampaignProducerError(f"detached supervisor evidence rejected: {exc.code}") from exc
    if (
        plan_path != run_dir / "detached_plan.json"
        or attempts_path != run_dir / "detached_attempts.jsonl"
        or verified.plan.get("plan_sha256") != plan_sha256
        or verified.attempt != supervisor_attempt
    ):
        raise CampaignProducerError("detached supervisor evidence identity differs")
    return verified


def _detached_evidence_environment() -> tuple[Path, Path, Path, str, int]:
    run_value = str(os.environ.get(DETACHED_RUN_DIR_ENV, "") or "")
    plan_value = str(os.environ.get(DETACHED_PLAN_PATH_ENV, "") or "")
    attempts_value = str(os.environ.get(DETACHED_ATTEMPTS_PATH_ENV, "") or "")
    plan_sha256 = str(os.environ.get(DETACHED_PLAN_SHA256_ENV, "") or "")
    attempt_value = str(os.environ.get(DETACHED_SUPERVISOR_ATTEMPT_ENV, "") or "")
    if not all((run_value, plan_value, attempts_value, plan_sha256, attempt_value)):
        raise CampaignProducerError("detached supervisor evidence environment is incomplete")
    try:
        run_dir = Path(run_value).expanduser().resolve(strict=True)
        plan_path = Path(plan_value).expanduser().resolve(strict=True)
        attempts_path = Path(attempts_value).expanduser().resolve(strict=True)
        supervisor_attempt = int(attempt_value)
    except (OSError, ValueError) as exc:
        raise CampaignProducerError("detached supervisor evidence environment is invalid") from exc
    if (
        plan_path != run_dir / "detached_plan.json"
        or attempts_path != run_dir / "detached_attempts.jsonl"
        or len(plan_sha256) != 64
        or any(character not in "0123456789abcdef" for character in plan_sha256)
        or supervisor_attempt <= 0
    ):
        raise CampaignProducerError("detached supervisor evidence environment is inconsistent")
    return run_dir, plan_path, attempts_path, plan_sha256, supervisor_attempt


def _import_brokered_worker_attempt(
    args: argparse.Namespace,
    plan: CampaignPlan,
    *,
    arm: str,
    attempt_slot: int,
    result: BrokeredProcessResult,
) -> dict[str, Any] | None:
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    paths = _worker_attempt_paths(campaign_dir, arm, attempt_slot)
    _persist_brokered_worker_result(paths, result)
    if result.returncode != 0 or result.status != "passed":
        return None
    summary = result.worker_origin_lifecycle
    lifecycle_value = summary.get("artifact_path") if isinstance(summary, Mapping) else None
    if not isinstance(lifecycle_value, str) or not lifecycle_value:
        raise CampaignProducerError("brokered worker has no terminal lifecycle")
    lifecycle_path = Path(lifecycle_value).expanduser().resolve(strict=True)
    if lifecycle_path.parent != paths["origin_dir"].resolve(strict=True):
        raise CampaignProducerError("brokered worker lifecycle path substitution")
    lifecycle = _read_canonical_json_artifact(
        lifecycle_path,
        role="brokered worker lifecycle",
    )
    authorization = lifecycle.get("authorization_payload")
    detached_plan_sha256 = (
        authorization.get("detached_plan_sha256") if isinstance(authorization, Mapping) else None
    )
    if not isinstance(detached_plan_sha256, str):
        raise CampaignProducerError("brokered worker detached plan is missing")
    detached_evidence = _verify_detached_worker_broker_result(
        result,
        require_claim_eligible=True,
    )
    if detached_evidence.plan["plan_sha256"] != detached_plan_sha256:
        raise CampaignProducerError("brokered worker detached plan differs")
    policy = _load_campaign_trust_policy(args, require_current=True)
    if policy is None:
        raise CampaignProducerError("brokered worker has no trusted policy")
    metadata = plan.to_dict()["metadata"]
    verified = verify_terminal_worker_stage(
        stage_path=paths["stage"],
        lifecycle_path=lifecycle_path,
        plan=plan,
        policy=policy,
        broker_result=result,
        arm=arm,
        worker_attempt_slot=attempt_slot,
        expected_protocol_sha256=_campaign_protocol_sha256(),
        expected_detached_plan_sha256=detached_plan_sha256,
        expected_broker_policy_sha256=result.policy_sha256,
        expected_model_identity_sha256=_sha256_bytes(
            canonical_json_bytes(metadata["model_identity"])
        ),
        expected_adapter_identity_sha256=_sha256_bytes(
            canonical_json_bytes(metadata["adapter_identity"])
        ),
    )
    _atomic_create_or_verify(
        paths["verified_stage"],
        canonical_json_bytes(verified.manifest) + b"\n",
    )
    return import_verified_worker_stage(
        canonical_journal_path=campaign_dir / JOURNAL_FILE,
        intent_path=paths["import_intent"],
        receipt_path=paths["import_receipt"],
        plan=plan,
        verified_stage=verified,
    )


def _build_worker_execution_manifest(
    args: argparse.Namespace,
    plan: CampaignPlan,
) -> dict[str, Any] | None:
    if plan.to_dict()["metadata"].get("claim_eligible") is not True:
        return None
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    policy = _load_campaign_trust_policy(args, require_current=True)
    if policy is None:
        raise CampaignProducerError("worker execution has no trusted policy")
    (
        detached_run_dir,
        detached_plan_path,
        detached_attempts_path,
        environment_plan_sha256,
        _supervisor_attempt,
    ) = _detached_evidence_environment()
    detached_plan_artifact = _read_stable_bytes(
        detached_plan_path,
        max_bytes=64 * 1024 * 1024,
    )
    imports: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    successful_batches: set[tuple[int, str]] = set()
    detached_plan_sha256: str | None = None
    detached_classification_head_sha256: str | None = None
    detached_terminals: list[dict[str, Any]] | None = None
    detached_quarantines: list[dict[str, Any]] | None = None

    for worker_look in _campaign_looks(args):
        for arm in _arms(args):
            for attempt_slot in _worker_attempt_slot_range(args, worker_look):
                paths = _worker_attempt_paths(campaign_dir, arm, attempt_slot)
                if not paths["broker_result"].exists():
                    origin_activity = paths["origin_dir"].is_dir() and any(
                        paths["origin_dir"].iterdir()
                    )
                    if origin_activity or paths["stage"].exists():
                        raise CampaignProducerError(
                            "worker attempt has activity without broker evidence"
                        )
                    continue
                broker_result, broker_artifact = _load_brokered_worker_result(
                    paths["broker_result"]
                )
                summary = broker_result.worker_origin_lifecycle
                if not isinstance(summary, dict):
                    raise CampaignProducerError(
                        "worker broker evidence has no lifecycle classification"
                    )
                detached_evidence = _verify_detached_worker_broker_result(
                    broker_result,
                    require_claim_eligible=False,
                )
                current_detached_plan = detached_evidence.plan["plan_sha256"]
                current_terminals = [
                    asdict(terminal) for terminal in detached_evidence.terminal_summaries
                ]
                current_quarantines = [
                    asdict(quarantine) for quarantine in detached_evidence.quarantine_summaries
                ]
                if detached_plan_sha256 is None:
                    detached_plan_sha256 = current_detached_plan
                    detached_classification_head_sha256 = (
                        detached_evidence.classification_head_sha256
                    )
                    detached_terminals = current_terminals
                    detached_quarantines = current_quarantines
                elif (
                    detached_plan_sha256 != current_detached_plan
                    or detached_classification_head_sha256
                    != detached_evidence.classification_head_sha256
                    or detached_terminals != current_terminals
                    or detached_quarantines != current_quarantines
                ):
                    raise CampaignProducerError(
                        "worker attempts do not share one detached evidence snapshot"
                    )
                matching_terminals = [
                    terminal
                    for terminal in current_terminals
                    if terminal["request_id"] == broker_result.request_id
                ]
                if len(matching_terminals) != 1:
                    raise CampaignProducerError(
                        "worker broker result has no unique detached terminal"
                    )
                detached_terminal = matching_terminals[0]
                expected_claim_eligible = bool(
                    broker_result.returncode == 0 and broker_result.status == "passed"
                )
                if detached_terminal["claim_eligible"] is not expected_claim_eligible:
                    raise CampaignProducerError(
                        "worker broker result eligibility differs from detached evidence"
                    )
                common = {
                    "arm": arm,
                    "worker_look": worker_look,
                    "worker_attempt_slot": attempt_slot,
                    "broker_result_artifact_sha256": broker_artifact[
                        "artifact_sha256"
                    ],
                    "broker_policy_sha256": broker_result.policy_sha256,
                    "broker_request_id": broker_result.request_id,
                    "broker_receipt_sha256": broker_result.receipt_sha256,
                    "broker_response_hmac_sha256": broker_result.response_hmac_sha256,
                    "worker_origin_lifecycle": summary,
                    "detached_supervisor_attempt": detached_evidence.attempt,
                    "detached_terminal_event_sha256": detached_terminal["event_sha256"],
                    "detached_classification_head_sha256": (
                        detached_evidence.classification_head_sha256
                    ),
                }
                if broker_result.returncode != 0 or broker_result.status != "passed":
                    excluded.append(
                        {
                            **common,
                            "classification": "terminal_excluded",
                            "status": broker_result.status,
                            "returncode": broker_result.returncode,
                            "reason": broker_result.error,
                        }
                    )
                    continue
                batch = (worker_look, arm)
                if batch in successful_batches:
                    raise CampaignProducerError(
                        "worker execution has multiple imported attempts for one batch"
                    )
                receipt = _import_brokered_worker_attempt(
                    args,
                    plan,
                    arm=arm,
                    attempt_slot=attempt_slot,
                    result=broker_result,
                )
                if receipt is None:
                    raise CampaignProducerError("passed worker attempt was not imported")
                verified_stage = _read_canonical_json_artifact(
                    paths["verified_stage"], role="verified worker stage"
                )
                import_intent = _read_canonical_json_artifact(
                    paths["import_intent"], role="worker import intent"
                )
                import_receipt = _read_canonical_json_artifact(
                    paths["import_receipt"], role="worker import receipt"
                )
                if import_receipt != receipt:
                    raise CampaignProducerError("worker import receipt differs")
                stage_detached_plan = verified_stage.get("detached_plan_sha256")
                if not isinstance(stage_detached_plan, str):
                    raise CampaignProducerError("worker stage detached plan is invalid")
                if detached_plan_sha256 != stage_detached_plan:
                    raise CampaignProducerError(
                        "worker attempts span different detached plans"
                    )
                imports.append(
                    {
                        **common,
                        "classification": "terminal_imported",
                        "session_id": summary["session_id"],
                        "detached_plan_sha256": stage_detached_plan,
                        "verified_stage_manifest_sha256": verified_stage[
                            "manifest_sha256"
                        ],
                        "stage_sha256": verified_stage["stage_sha256"],
                        "stage_journal_head_sha256": verified_stage[
                            "stage_journal_head_sha256"
                        ],
                        "result_chain_head_sha256": verified_stage[
                            "result_chain_head_sha256"
                        ],
                        "cell_ids": verified_stage["cell_ids"],
                        "import_intent_sha256": import_intent["intent_sha256"],
                        "import_receipt_sha256": import_receipt["receipt_sha256"],
                        "canonical_imports": import_receipt["imported"],
                    }
                )
                successful_batches.add(batch)

    if (
        successful_batches
        != {
            (worker_look, arm)
            for worker_look in _campaign_looks(args)
            for arm in _arms(args)
        }
        or detached_plan_sha256 is None
        or detached_classification_head_sha256 is None
        or detached_terminals is None
        or detached_quarantines is None
    ):
        raise CampaignProducerError("worker execution arm coverage is incomplete")
    if detached_plan_sha256 != environment_plan_sha256:
        raise CampaignProducerError("worker execution detached plan differs from env")
    if _read_stable_bytes(detached_plan_path, max_bytes=64 * 1024 * 1024) != detached_plan_artifact:
        raise CampaignProducerError("detached plan changed while worker evidence was assembled")
    with CampaignJournal(campaign_dir / JOURNAL_FILE, plan) as journal:
        canonical_records = journal.result_records()
    canonical_origins = {
        record["result"]["worker_origin"]["origin_sha256"] for record in canonical_records
    }
    imported_origins = {
        cell["result_origin_sha256"] for entry in imports for cell in entry["canonical_imports"]
    }
    if canonical_origins != imported_origins or len(canonical_records) != len(plan.cell_ids):
        raise CampaignProducerError("canonical worker origins differ from terminal imports")
    detached_classifications = {
        "terminal_count": len(detached_terminals),
        "terminals": detached_terminals,
        "quarantine_count": len(detached_quarantines),
        "quarantines": detached_quarantines,
    }
    material = {
        "schema": WORKER_EXECUTION_MANIFEST_SCHEMA,
        "campaign_name": plan.campaign_name,
        "policy_sha256": policy.policy_sha256,
        "protocol_sha256": _campaign_protocol_sha256(),
        "plan_sha256": plan.plan_sha256,
        "detached_run_dir": str(detached_run_dir),
        "detached_plan_path": str(detached_plan_path),
        "detached_attempts_path": str(detached_attempts_path),
        "detached_plan_artifact_sha256": _sha256_bytes(detached_plan_artifact),
        "detached_plan_sha256": detached_plan_sha256,
        "detached_classification_head_sha256": (detached_classification_head_sha256),
        "detached_classifications": detached_classifications,
        "detached_classifications_sha256": _sha256_bytes(
            canonical_json_bytes(detached_classifications)
        ),
        "import_count": len(imports),
        "imports": imports,
        "imports_sha256": _sha256_bytes(canonical_json_bytes(imports)),
        "excluded_count": len(excluded),
        "excluded_attempts": excluded,
        "excluded_attempts_sha256": _sha256_bytes(canonical_json_bytes(excluded)),
    }
    manifest = {
        **material,
        "manifest_sha256": _sha256_bytes(canonical_json_bytes(material)),
    }
    _atomic_create_or_verify(
        campaign_dir / WORKER_EXECUTION_MANIFEST_FILE,
        canonical_json_bytes(manifest) + b"\n",
    )
    return manifest


def _run_child(
    args: argparse.Namespace,
    arm: str,
    timeout_s: float,
    *,
    worker_attempt_slot: int | None = None,
    worker_look: int = 0,
) -> int | BrokeredProcessResult:
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    log_path = campaign_dir / LOG_FILE
    command = _worker_args(
        args,
        arm,
        worker_attempt_slot=worker_attempt_slot,
        worker_look=worker_look,
    )
    if broker_available():
        deadline = time.monotonic() + timeout_s
        authorization_announced = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return 124
            try:
                return run_brokered_process(
                    command,
                    cwd=REPO_ROOT,
                    stdout_path=log_path,
                    timeout_s=remaining,
                )
            except DetachedBrokerError as exc:
                message = str(exc)
                marker = "worker-origin external authorization required at "
                if worker_attempt_slot is None or marker not in message:
                    raise
                attestation_path = message.split(marker, 1)[1].strip()
                if not authorization_announced:
                    print(
                        json.dumps(
                            {
                                "state": "worker_origin_signature_required",
                                "arm": arm,
                                "worker_look": worker_look,
                                "worker_attempt_slot": worker_attempt_slot,
                                "attestation_path": attestation_path,
                            },
                            indent=2,
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    authorization_announced = True
                time.sleep(min(0.5, max(0.01, remaining)))
    if worker_attempt_slot is not None:
        raise CampaignProducerError("claim worker requires the detached supervisor broker")
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
    return returncode


def _detached_worker_origin_policy(
    args: argparse.Namespace,
    plan: CampaignPlan,
    *,
    arm: str,
    attempt_slot: int,
    worker_look: int = 0,
) -> dict[str, Any]:
    metadata = plan.to_dict()["metadata"]
    cells = [
        {
            "cell_id": cell_id,
            "cell_type": PAIRED_CAMPAIGN_CELL_TYPE,
        }
        for cell_id in _arm_cell_ids_for_look(
            plan,
            arm,
            worker_look,
            cumulative=False,
        )
    ]
    cells.sort(
        key=lambda cell: int(plan.cell_definition(cell["cell_id"])["execution_ordinal_within_arm"])
    )
    if not cells:
        raise CampaignProducerError("worker-origin arm has no planned cells")
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    paths = _worker_attempt_paths(campaign_dir, arm, attempt_slot)
    return {
        "schema": "aura.detached_step.worker_origin_policy.v1",
        "campaign_name": plan.campaign_name,
        "protocol_sha256": _campaign_protocol_sha256(),
        "trust_policy_path": str(
            Path(args.campaign_trust_policy).expanduser().resolve(strict=True)
        ),
        "trust_root_path": str(Path(args.campaign_trust_root).expanduser().resolve(strict=True)),
        "artifact_dir": str(paths["origin_dir"]),
        "arm": arm,
        "worker_attempt_slot": attempt_slot,
        "allowed_cells": cells,
        "model_identity_sha256": _sha256_bytes(canonical_json_bytes(metadata["model_identity"])),
        "adapter_identity_sha256": _sha256_bytes(
            canonical_json_bytes(metadata["adapter_identity"])
        ),
        "authorization_ttl_seconds": min(
            7 * 24 * 60 * 60,
            max(60, int(math.ceil(float(args.arm_timeout)))),
        ),
    }


def _detached_broker_policy(
    args: argparse.Namespace,
    plan: CampaignPlan | None = None,
) -> list[dict[str, Any]]:
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    if args.confirmatory:
        if plan is None:
            raise CampaignProducerError("claim broker policy requires the frozen campaign plan")
        policies: list[dict[str, Any]] = []
        for worker_look in _campaign_looks(args):
            for arm in _arms(args):
                for attempt_slot in _worker_attempt_slot_range(args, worker_look):
                    policies.append(
                        {
                            "command": _worker_args(
                                args,
                                arm,
                                worker_attempt_slot=attempt_slot,
                                worker_look=worker_look,
                            ),
                            "cwd": str(REPO_ROOT),
                            "stdout_path": str(campaign_dir / LOG_FILE),
                            "timeout_s_max": float(args.arm_timeout),
                            "max_invocations": 1,
                            "worker_origin": _detached_worker_origin_policy(
                                args,
                                plan,
                                arm=arm,
                                attempt_slot=attempt_slot,
                                worker_look=worker_look,
                            ),
                        }
                    )
        return policies
    return [
        {
            "command": _worker_args(args, arm, worker_look=worker_look),
            "cwd": str(REPO_ROOT),
            "stdout_path": str(campaign_dir / LOG_FILE),
            "timeout_s_max": float(args.arm_timeout),
            "max_invocations": int(args.max_infra_attempts),
        }
        for worker_look in _campaign_looks(args)
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
    worker_origin_required = metadata.get("claim_eligible") is True
    arm_execution_order = tuple(metadata["arm_execution_order"])
    if set(arm_execution_order) != set(_arms(args)):
        raise CampaignProducerError("frozen arm execution order is invalid")
    for worker_look in _campaign_looks(args):
        for arm in arm_execution_order:
            if arm not in _arms(args):
                continue
            attempts = 0
            while not _arm_outputs_sealed(
                campaign_dir,
                plan,
                arm,
                worker_look=worker_look,
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    print(
                        "campaign deadline exceeded; resumable evidence preserved",
                        flush=True,
                    )
                    return 3
                attempts += 1
                if attempts > args.max_infra_attempts:
                    print(
                        f"look {worker_look} arm {arm} exhausted infrastructure attempts",
                        flush=True,
                    )
                    return 4
                worker_attempt_slot = None
                if worker_origin_required:
                    slot_range = _worker_attempt_slot_range(args, worker_look)
                    worker_attempt_slot = _next_worker_attempt_slot(
                        campaign_dir,
                        arm,
                        minimum=slot_range.start,
                        maximum=slot_range.stop - 1,
                    )
                    if worker_attempt_slot is None:
                        print(
                            f"look {worker_look} arm {arm} exhausted "
                            "pre-authorized worker slots",
                            flush=True,
                        )
                        return 4
                outcome = _run_child(
                    args,
                    arm,
                    min(args.arm_timeout, remaining),
                    worker_attempt_slot=worker_attempt_slot,
                    worker_look=worker_look,
                )
                code = (
                    outcome.returncode
                    if isinstance(outcome, BrokeredProcessResult)
                    else outcome
                )
                if worker_origin_required:
                    if not isinstance(outcome, BrokeredProcessResult):
                        if code == 124:
                            print(
                                f"look {worker_look} arm {arm} authorization "
                                "or execution timed out",
                                flush=True,
                            )
                            return code
                        raise CampaignProducerError(
                            "claim worker has no authenticated broker result"
                        )
                    _import_brokered_worker_attempt(
                        args,
                        plan,
                        arm=arm,
                        attempt_slot=int(worker_attempt_slot),
                        result=outcome,
                    )
                print(
                    f"look {worker_look} arm {arm} process exit={code} "
                    f"attempt={attempts}",
                    flush=True,
                )
                if code != 0 and attempts >= args.max_infra_attempts:
                    return code or 4

    worker_execution = _build_worker_execution_manifest(args, plan)
    sealed_outputs = _seal_output_manifest(
        campaign_dir,
        plan,
        worker_execution=worker_execution,
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
    final_material["sealed_output_manifest_sha256"] = sealed_outputs["manifest_sha256"]
    final_material["answer_reveal_sha256"] = answer_reveal["reveal_sha256"]
    if worker_origin_required:
        if worker_execution is None:
            raise CampaignProducerError("worker execution evidence is missing")
        final_material["worker_execution_manifest_sha256"] = worker_execution["manifest_sha256"]
        final_material["detached_plan_sha256"] = worker_execution["detached_plan_sha256"]
        final_material["detached_classification_head_sha256"] = worker_execution[
            "detached_classification_head_sha256"
        ]
        final_material["detached_classifications_sha256"] = worker_execution[
            "detached_classifications_sha256"
        ]
        final_material["worker_imports_sha256"] = worker_execution["imports_sha256"]
        final_material["worker_excluded_attempts_sha256"] = worker_execution[
            "excluded_attempts_sha256"
        ]
    final = {
        **final_material,
        "grade_sha256": _sha256_bytes(canonical_json_bytes(final_material)),
    }
    _atomic_create_or_verify(campaign_dir / GRADE_FILE, canonical_json_bytes(final) + b"\n")
    final_run_envelope = _admit_final_run_envelope(
        args,
        plan,
        sealed_outputs=sealed_outputs,
        answer_reveal=answer_reveal,
        campaign_manifest=manifest,
        grade=final,
        worker_execution=worker_execution,
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
                "final_run_envelope_sha256": final_run_envelope.get("envelope_sha256"),
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
        expected_training_corpus_sha256=_adapter_dataset_manifest_sha256(adapter_identity),
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
    parser.add_argument("--seed-entropy-bits", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--sequential-look-observations",
        default="",
        help="comma-separated cumulative observations per domain",
    )
    parser.add_argument(
        "--sequential-alpha-weights",
        default="",
        help="comma-separated exact rational alpha shares, such as 1/100,99/100",
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
        choices=RLC_MECHANISM_PROFILES,
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
    parser.add_argument("--worker-arm", choices=FULL_ARMS, default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-look", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-attempt-slot", type=_positive_int, default=0, help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-stage-journal", default="", help=argparse.SUPPRESS)
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
    if bool(args.sequential_look_observations) != bool(args.sequential_alpha_weights):
        parser.error(
            "--sequential-look-observations and --sequential-alpha-weights "
            "must be supplied together"
        )
    if args.sequential_look_observations:
        args.sequential_look_values = _csv_ints(
            parser,
            args.sequential_look_observations,
            "--sequential-look-observations",
        )
        args.sequential_alpha_weight_values = _csv_rationals(
            parser,
            args.sequential_alpha_weights,
            "--sequential-alpha-weights",
        )
        expected_final = int(args.seed_count) if args.worker_arm else len(args.seed_values)
        if args.sequential_look_values[-1] != expected_final:
            parser.error(
                "the terminal sequential look must equal the campaign seed count"
            )
    else:
        args.sequential_look_values = ()
        args.sequential_alpha_weight_values = ()
    if args.worker_arm:
        if args.worker_look not in _campaign_looks(args):
            parser.error("worker look is outside the frozen sequential contract")
    elif args.worker_look != 0:
        parser.error("worker look is reserved for isolated workers")
    worker_origin_values = (
        args.worker_attempt_slot,
        args.worker_stage_journal,
    )
    if args.worker_arm:
        if any(worker_origin_values) and not all(worker_origin_values):
            parser.error("worker origin arguments must be supplied together")
    elif any(worker_origin_values):
        parser.error("worker origin arguments are reserved for isolated workers")
    if args.worker_arm:
        if args.seed_count <= 0 or not 1 <= args.seed_entropy_bits <= 63:
            parser.error("worker process requires public seed count and entropy bounds")
    elif args.seed_count != 0 or args.seed_entropy_bits != 0:
        parser.error("--seed-count/--seed-entropy-bits are reserved for isolated workers")
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    args.campaign_dir = str(campaign_dir)
    if args.worker_stage_journal:
        args.worker_stage_journal = str(
            Path(args.worker_stage_journal).expanduser().resolve(strict=False)
        )
    if args.contamination_audit:
        args.contamination_audit = str(
            Path(args.contamination_audit).expanduser().resolve(strict=True)
        )
        if not args.contamination_trust_root:
            parser.error("--contamination-trust-root is required with --contamination-audit")
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
    if any(trust_paths) and not (args.campaign_trust_policy and args.campaign_trust_root):
        parser.error("--campaign-trust-policy and --campaign-trust-root are required together")
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
            parser.error("--prepare-trust requires campaign trust policy and independent root")
        if args.worker_arm:
            parser.error("--prepare-trust cannot be combined with --worker-arm")
        print(json.dumps(_prepare_trust_requests(args), indent=2, sort_keys=True))
        return 0
    campaign_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_arm:
        persisted = _load_persisted_plan(campaign_dir)
        expected, public_tasks = _expected_worker_plan(args, persisted)
        if persisted.to_dict() != expected.to_dict():
            raise CampaignProducerError("persisted plan does not match requested worker campaign")
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
                    "detached_broker_policy": _detached_broker_policy(args, expected),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    return _orchestrate(args, persisted, tasks)


if __name__ == "__main__":
    raise SystemExit(main())
