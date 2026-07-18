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
import subprocess
import sys
import time
import uuid
from collections import Counter
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
from core.brain.llm.latent_cortex.frontier_tasks import (  # noqa: E402
    FRONTIER_DOMAINS,
    FrontierTask,
    build_task_manifest,
    generate_task_battery,
    parse_final_answer,
)
from core.brain.llm.latent_cortex.paired_campaign import (  # noqa: E402
    ADAPTER_RLC,
    BASE_EQUAL_COMPUTE,
    BASE_RLC,
    FULL_ARMS,
    PRIMARY_ARMS,
    build_campaign_plan,
    grade_campaign,
)
from core.brain.llm.latent_cortex.runtime_identity import (  # noqa: E402
    build_worker_identity,
    logical_model_parameter_count,
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
OBJECTIVE_SOURCE = REPO_ROOT / "core/learning/recurrence_native_objective.py"
CONTAMINATION_AUDIT_SCHEMA = "aura.latent_cortex.contamination_audit.v2"

class CampaignProducerError(RuntimeError):
    pass


@contextlib.contextmanager
def _deadline_alarm(seconds: float, stage: str):
    """Hard wall deadline for one worker stage on POSIX main threads."""

    def expired(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"{stage} exceeded {seconds:.3f}s hard deadline")

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
    return {"fingerprint": combined.hexdigest(), "method": "sha256", "files": len(files)}


def _atomic_create_or_verify(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise CampaignProducerError(f"symlink output rejected: {path}")
    if path.exists():
        if path.read_bytes() != payload:
            raise CampaignProducerError(f"existing artifact differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(temporary, flags, 0o600)
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
        os.link(temporary, path)
    except FileExistsError as exc:
        if path.read_bytes() != payload:
            raise CampaignProducerError(f"concurrent artifact differs: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


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


def _identity_material(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    model_path = Path(args.model).expanduser().resolve(strict=True)
    adapter_dir = Path(args.adapter).expanduser().resolve(strict=True)
    weight_identity = _fresh_checkpoint_file_fingerprint(model_path)
    model_identity = {
        "model_path": str(model_path),
        **weight_identity,
        "runtime_bundle": _runtime_bundle_identity(
            model_path,
            weight_identity=weight_identity,
        ),
    }
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
        "manifest": manifest.to_dict(),
        "identity_receipt": identity.to_dict(),
    }
    return model_identity, adapter_identity


def _model_load_boundary_identity(model_path: Path) -> dict[str, Any]:
    weight_identity = _fresh_checkpoint_file_fingerprint(model_path)
    runtime_bundle = _runtime_bundle_identity(
        model_path,
        weight_identity=weight_identity,
    )
    return {
        "weight_fingerprint": weight_identity["fingerprint"],
        "weight_method": weight_identity["method"],
        "weight_file_count": weight_identity["files"],
        "runtime_bundle_sha256": runtime_bundle["bundle_sha256"],
    }


def _adapter_load_boundary_identity(
    adapter_dir: Path,
    manifest: dict[str, Any],
    *,
    base_checkpoint_fingerprint: str,
) -> dict[str, Any]:
    adapter_binding = manifest["adapter"]
    receipt_binding = manifest["training_receipt"]
    adapter_path = adapter_dir / adapter_binding["path"]
    receipt_path = adapter_dir / receipt_binding["path"]
    receipt = validate_adapter_identity(
        manifest,
        actual_base_checkpoint_fingerprint=base_checkpoint_fingerprint,
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
    )


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


def _contamination_audit(
    args: argparse.Namespace,
    tasks: tuple[FrontierTask, ...],
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
    trust_root_path = str(
        getattr(args, "contamination_trust_root", "") or ""
    ).strip()
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
    manifest = build_task_manifest(tasks)
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


def _build_rlc_config(args: argparse.Namespace) -> Any:
    from core.brain.llm.latent_cortex.types import (
        BranchConfig,
        CortexConfig,
        FastWeightsConfig,
        LatentOptConfig,
        RecurrenceConfig,
        WorkspaceConfig,
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


def _execution_config(args: argparse.Namespace) -> dict[str, Any]:
    effective = _build_rlc_config(args)
    return {
        "profile": args.profile,
        "difficulty": args.difficulty,
        "generation_seeds": list(args.seed_values),
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
        "rlc_profile": args.rlc_profile,
        "decode_max_tokens": args.decode_max_tokens,
        "episode_timeout_s": args.episode_timeout,
        "load_timeout_s": args.load_timeout,
        "warmup_timeout_s": args.warmup_timeout,
        "arm_timeout_s": args.arm_timeout,
        "campaign_timeout_s": args.campaign_timeout,
        "equal_compute_max_samples": args.equal_compute_max_samples,
        "adapter_process_isolation": True,
        "vanilla_fallback_allowed": False,
        "implementation_sha256": _implementation_sha256(),
    }


def _expected_plan(args: argparse.Namespace) -> tuple[CampaignPlan, tuple[FrontierTask, ...]]:
    tasks = _tasks(args)
    model_identity, adapter_identity = _identity_material(args)
    contamination_audit = _contamination_audit(args, tasks)
    claim_eligible = _claim_eligible(args, model_identity, contamination_audit)
    plan = build_campaign_plan(
        args.campaign_name,
        tasks,
        model_identity=model_identity,
        adapter_identity=adapter_identity,
        execution_config=_execution_config(args),
        contamination_audit=contamination_audit,
        arms=_arms(args),
        claim_eligible=claim_eligible,
    )
    return plan, tasks


def _claim_eligible(
    args: argparse.Namespace,
    model_identity: dict[str, Any],
    contamination_audit: dict[str, Any],
) -> bool:
    per_domain = len(args.seed_values)
    runtime_bundle = model_identity.get("runtime_bundle")
    return bool(
        args.confirmatory
        and per_domain >= 20
        and args.profile == "full"
        and set(args.domain_values) == set(FRONTIER_DOMAINS)
        and isinstance(runtime_bundle, dict)
        and runtime_bundle.get("model_type") == "qwen2"
        and runtime_bundle.get("logical_parameter_count_basis")
        == "architecture_config_logical"
        and int(runtime_bundle.get("logical_parameter_count") or 0) >= 30_000_000_000
        and contamination_audit.get("status") == "passed_zero_overlap"
        and isinstance(contamination_audit.get("signature"), dict)
        and contamination_audit["signature"].get("verified") is True
    )


def _persist_plan(campaign_dir: Path, plan: CampaignPlan) -> None:
    _atomic_create_or_verify(
        campaign_dir / PLAN_FILE,
        canonical_json_bytes(plan.to_dict()) + b"\n",
    )


def _load_persisted_plan(campaign_dir: Path) -> CampaignPlan:
    payload = json.loads((campaign_dir / PLAN_FILE).read_text(encoding="utf-8"))
    return CampaignPlan.from_dict(payload)


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

    rank = int(manifest["lora"]["rank"])
    targets = tuple(manifest["lora"]["targets"])
    expected = int(manifest["lora"]["wrapped_projection_count"])
    tensor_records = {record["key"]: record for record in manifest["tensors"]}
    projections = sorted(
        {
            key.removesuffix(".lora_a").removesuffix(".lora_b")
            for key in tensor_records
        }
    )
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
            setattr(parent, leaf, LoRALinear.from_base(original, r=rank))

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
            not bool(mx.array_equal(loaded[key], weights[key]))
            for key in expected_keys
        ):
            raise CampaignProducerError("adapter weight readback mismatch")
        after_load = _read_stable_bytes(
            weights_path,
            max_bytes=max(1, expected_adapter_size),
        )
        if after_load != before_load:
            raise CampaignProducerError("adapter bytes changed across load boundary")
    except BaseException:
        for parent, leaf, original in reversed(originals):
            setattr(parent, leaf, original)
        raise
    mx.eval(model.parameters())
    return len(originals)


def _render_prompt(tokenizer: Any, task: FrontierTask) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": task.public.prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )


def _vanilla_once(
    model: Any,
    tokenizer: Any,
    task: FrontierTask,
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
    task: FrontierTask,
    *,
    target_layer_apps: int,
    max_tokens: int,
    max_samples: int,
) -> tuple[str, int, int]:
    outputs: list[str] = []
    spent = 0
    seed_base = int(task.public.task_payload_sha256[:16], 16)
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


def _make_rlc_engine(model: Any, tokenizer: Any, args: argparse.Namespace) -> Any:
    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    config = _build_rlc_config(args)

    return LatentCortexEngine(
        model,
        tokenizer,
        config,
        model_path=str(Path(args.model).expanduser().resolve()),
    )


def _run_rlc(
    engine: Any,
    task: FrontierTask,
    args: argparse.Namespace,
) -> tuple[str, int, dict[str, Any]]:
    from core.brain.llm.latent_cortex.task_verifiers import EpisodeTaskVerifier
    from core.brain.llm.latent_cortex.types import ComputeBudget

    verifier = EpisodeTaskVerifier(task.public.prompt)
    budget = ComputeBudget(wall_clock_s=args.episode_timeout)
    result = engine.reason(
        messages=[{"role": "user", "content": task.public.prompt}],
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
    for record in journal.committed_records():
        definition = record["definition"]
        arm = definition["arm"]
        if arm in {BASE_RLC, ADAPTER_RLC}:
            costs[(definition["task_id"], arm)] = int(record["result"]["layer_apps"])
    return costs


def _execute_worker(
    args: argparse.Namespace, plan: CampaignPlan, tasks: tuple[FrontierTask, ...]
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
            planned_load_boundary = {
                "weight_fingerprint": planned_model["fingerprint"],
                "weight_method": planned_model["method"],
                "weight_file_count": planned_model["files"],
                "runtime_bundle_sha256": planned_runtime["bundle_sha256"],
            }
            pre_load_boundary = _model_load_boundary_identity(model_dir)
            if pre_load_boundary != planned_load_boundary:
                raise CampaignProducerError(
                    "model bytes differ from frozen plan before load"
                )
            actual_adapter_identity: dict[str, Any] | None = None
            if arm.startswith("adapter_"):
                actual_adapter_identity = _adapter_load_boundary_identity(
                    adapter_dir,
                    manifest,
                    base_checkpoint_fingerprint=planned_model["fingerprint"],
                )
                if actual_adapter_identity != metadata["adapter_identity"][
                    "identity_receipt"
                ]:
                    raise CampaignProducerError(
                        "adapter bytes differ from frozen plan before load"
                    )
            model, tokenizer = load(model_path)
            wrapped = 0
            if arm.startswith("adapter_"):
                wrapped = _load_adapter(model, adapter_dir, manifest)
                post_adapter_identity = _adapter_load_boundary_identity(
                    adapter_dir,
                    manifest,
                    base_checkpoint_fingerprint=planned_model["fingerprint"],
                )
                if post_adapter_identity != actual_adapter_identity:
                    raise CampaignProducerError(
                        "adapter identity changed across load boundary"
                    )
            post_load_boundary = _model_load_boundary_identity(model_dir)
            if post_load_boundary != pre_load_boundary:
                raise CampaignProducerError(
                    "model identity changed across load boundary"
                )
            worker_identity = build_worker_identity(
                model,
                model_path=model_path,
                worker_boot_id=uuid.uuid4().hex,
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
                    "worker_weight_file_count": post_load_boundary[
                        "weight_file_count"
                    ],
                    "worker_runtime_bundle_sha256": post_load_boundary[
                        "runtime_bundle_sha256"
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

        rlc_engine = _make_rlc_engine(model, tokenizer, args) if arm.endswith("_rlc") else None
        campaign_dir = Path(args.campaign_dir).expanduser().resolve()
        with CampaignJournal(campaign_dir / JOURNAL_FILE, plan) as journal:
            costs = _prior_rlc_costs(journal)
            pending = [
                cell_id
                for cell_id in journal.resume().runnable_cell_ids
                if plan.cell_definition(cell_id)["arm"] == arm
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
                            source_arm = BASE_RLC if arm == BASE_EQUAL_COMPUTE else ADAPTER_RLC
                            target = costs.get((task.task_id, source_arm))
                            if target is None:
                                raise CampaignProducerError("equal-compute prerequisite missing")
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
                        "adapter_identity_sha256": metadata["adapter_identity"]["identity_receipt"][
                            "composite_identity_sha256"
                        ]
                        if arm.startswith("adapter_")
                        else None,
                        "runtime_adapter_identity": actual_adapter_identity,
                        "runtime_model_identity": worker_identity,
                        "episode_receipt": receipt,
                    }
                    score = task.score(text).to_dict()
                    verification = {
                        "correct": score["correct"],
                        "score_receipt": score,
                        "answer_commitment_sha256": (
                            task.public.answer_commitment_sha256
                        ),
                    }
                    journal.record_arm_result(cell_id, attempt_id, result)
                    journal.record_verified(cell_id, attempt_id, verification)
                    journal.commit_cell(
                        cell_id,
                        attempt_id,
                        {
                            "result_sha256": _sha256_bytes(canonical_json_bytes(result)),
                            "verification_sha256": _sha256_bytes(
                                canonical_json_bytes(verification)
                            ),
                        },
                    )
                    print(
                        f"[{arm}] committed {task.task_id} correct={score['correct']} "
                        f"latency={elapsed:.2f}s layer_apps={layer_apps}",
                        flush=True,
                    )
                except BaseException as exc:
                    try:
                        journal.fail_cell(
                            cell_id,
                            attempt_id,
                            reason=f"infrastructure_failure:{type(exc).__name__}",
                            details={"message": str(exc)[:2000]},
                        )
                    except BaseException:
                        pass
                    raise
    return 0


def _worker_args(args: argparse.Namespace, arm: str) -> list[str]:
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
        "--seeds",
        args.seeds,
        "--domains",
        args.domains,
        "--difficulty",
        str(args.difficulty),
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
    if args.confirmatory:
        command.append("--confirmatory")
    if args.contamination_audit:
        command.extend(["--contamination-audit", args.contamination_audit])
        command.extend(
            ["--contamination-trust-root", args.contamination_trust_root]
        )
    return command


def _arm_complete(campaign_dir: Path, plan: CampaignPlan, arm: str) -> bool:
    with CampaignJournal(campaign_dir / JOURNAL_FILE, plan) as journal:
        committed = set(journal.resume().committed_cell_ids)
    expected = {cell_id for cell_id in plan.cell_ids if plan.cell_definition(cell_id)["arm"] == arm}
    return expected.issubset(committed)


def _run_child(args: argparse.Namespace, arm: str, timeout_s: float) -> int:
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    log_path = campaign_dir / LOG_FILE
    command = _worker_args(args, arm)
    if broker_available():
        return run_brokered_process(
            command,
            cwd=REPO_ROOT,
            stdout_path=log_path,
            timeout_s=timeout_s,
        ).returncode
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
            return 124


def _detached_broker_policy(args: argparse.Namespace) -> list[dict[str, Any]]:
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
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
    arm_execution_order = tuple(metadata["arm_execution_order"])
    if set(arm_execution_order) != set(_arms(args)):
        raise CampaignProducerError("frozen arm execution order is invalid")
    for arm in arm_execution_order:
        if arm not in _arms(args):
            continue
        attempts = 0
        while not _arm_complete(campaign_dir, plan, arm):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print("campaign deadline exceeded; resumable evidence preserved", flush=True)
                return 3
            attempts += 1
            if attempts > args.max_infra_attempts:
                print(f"arm {arm} exhausted infrastructure attempts", flush=True)
                return 4
            code = _run_child(args, arm, min(args.arm_timeout, remaining))
            print(f"arm {arm} process exit={code} attempt={attempts}", flush=True)
            if code != 0 and attempts >= args.max_infra_attempts:
                return code or 4

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
    )
    final_material = dict(grade)
    final_material.pop("grade_sha256", None)
    final_material["campaign_manifest_sha256"] = manifest["manifest_sha256"]
    final = {
        **final_material,
        "grade_sha256": _sha256_bytes(canonical_json_bytes(final_material)),
    }
    _atomic_create_or_verify(campaign_dir / GRADE_FILE, canonical_json_bytes(final) + b"\n")
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
                "grade_path": str(campaign_dir / GRADE_FILE),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if final["verdict"] == "gain_proven" else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--campaign-name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--adapter-id", default="resident-32b-r1")
    parser.add_argument("--seeds", default="7001,7002,7003,7004")
    parser.add_argument("--domains", default=",".join(FRONTIER_DOMAINS))
    parser.add_argument("--difficulty", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--profile", choices=("primary", "full"), default="primary")
    parser.add_argument("--confirmatory", action="store_true")
    parser.add_argument("--contamination-audit", default="")
    parser.add_argument("--contamination-trust-root", default="")
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
    parser.add_argument("--worker-arm", choices=FULL_ARMS, default="", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    args.seed_values = _csv_ints(parser, args.seeds, "--seeds")
    args.domain_values = _csv_domains(parser, args.domains)
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    args.campaign_dir = str(campaign_dir)
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
        parser.error(
            "--contamination-trust-root requires --contamination-audit"
        )
    campaign_dir.mkdir(parents=True, exist_ok=True)
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
    if args.worker_arm:
        return _execute_worker(args, persisted, tasks)
    return _orchestrate(args, persisted, tasks)


if __name__ == "__main__":
    raise SystemExit(main())
