"""Runtime manifest emission for canonical Aura boots."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation

_GIT_ERRORS = (OSError, UnicodeDecodeError, ValueError)
_MANIFEST_ERRORS = (OSError, TypeError, ValueError, RuntimeError, AttributeError)
_CONTAINER_LOCK_TIMEOUT_S = 2.0

_ROLE_HEALTH_KEYS: dict[str, tuple[str, ...]] = {
    "runtime": ("kernel_interface",),
    "model": ("inference_gate", "llm_router"),
    "memory_writer": ("memory_write_gateway",),
    "memory_interface": ("memory_facade",),
    "state_writer": ("state_repository",),
    "event_bus": ("event_bus",),
    "output_gate": ("output_gate",),
    "governance": ("unified_will", "authority_gateway", "capability_engine"),
}


def _git_commit(root: Path) -> str:
    try:
        git_dir = root / ".git"
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref_name = head.split(" ", 1)[1].strip()
            ref_path = git_dir / ref_name
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()
            packed_refs = git_dir / "packed-refs"
            if packed_refs.exists():
                for line in packed_refs.read_text(encoding="utf-8").splitlines():
                    if line and not line.startswith("#") and line.endswith(f" {ref_name}"):
                        return line.split(" ", 1)[0].strip()
            return "unknown_ref_not_found"
        return head
    except _GIT_ERRORS:
        return "unknown"


def _source_owner(obj: Any) -> str:
    target = obj
    if not inspect.isclass(target):
        target = target.__class__
    try:
        path = inspect.getsourcefile(target) or inspect.getfile(target)
    except (TypeError, OSError):
        return "unknown"
    if not path:
        return "unknown"
    try:
        return str(Path(path).resolve())
    except _MANIFEST_ERRORS:
        return str(path)


def _health_contract_snapshot(
    readiness_snapshot: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    # During canonical boot the caller already captured the runtime-health
    # verdict. Re-running all liveness checks here can deadlock on the container
    # lock while the API is still coming up, so manifest health collection stays
    # opportunistic once a boot readiness snapshot exists.
    if isinstance(readiness_snapshot, dict):
        return {}
    try:
        from core.runtime.health_contract import runtime_health_report

        report = runtime_health_report()
        return {
            str(service.get("container_key")): dict(service)
            for service in report.get("services", [])
            if isinstance(service, dict) and service.get("container_key")
        }
    except _MANIFEST_ERRORS:
        return {}


def _service_snapshot(
    health_by_key: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    from core.container import ServiceContainer

    contract_health = health_by_key or {}
    try:
        statuses = ServiceContainer.get_all_subsystem_statuses()
    except _MANIFEST_ERRORS:
        statuses = {}

    services: dict[str, dict[str, Any]] = {}
    acquired = ServiceContainer._lock.acquire(timeout=_CONTAINER_LOCK_TIMEOUT_S)
    if not acquired:
        record_degradation(
            "runtime_manifest",
            TimeoutError("service container lock timed out during manifest service snapshot"),
            severity="warning",
            action="emitted partial runtime manifest instead of blocking desktop boot",
            enforce_failure_policy=False,
        )
        return {
            "_manifest_snapshot": {
                "service": "_manifest_snapshot",
                "owner": "core/runtime/runtime_manifest.py",
                "required": False,
                "initialized": False,
                "health_status": "partial_container_lock_timeout",
                "health_contract": None,
                "dependencies": [],
            }
        }
    try:
        items = list(ServiceContainer._services.items())
    finally:
        ServiceContainer._lock.release()

    for name, desc in items:
        instance = getattr(desc, "instance", None)
        factory = getattr(desc, "factory", None)
        owner_target = instance if instance is not None else factory
        health_evidence = contract_health.get(name)
        if health_evidence is not None:
            present = bool(health_evidence.get("present", False))
            liveness = str(health_evidence.get("liveness", "") or "")
            health_status = "ok" if present and liveness == "ok" else "liveness_failed"
        else:
            health_status = statuses.get(name, "registered_unchecked")
        services[name] = {
            "service": name,
            "owner": _source_owner(owner_target) if owner_target is not None else "unknown",
            "required": bool(getattr(desc, "required", False)),
            "initialized": bool(getattr(desc, "initialized", False)),
            "health_status": health_status,
            "health_contract": health_evidence,
            "dependencies": list(getattr(desc, "dependencies", []) or []),
        }
    return dict(sorted(services.items()))


def _role_snapshot(
    health_by_key: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    from core.container import ServiceContainer
    from core.runtime.service_manifest import SERVICE_MANIFEST, verify_manifest

    contract_health = health_by_key or {}
    acquired = ServiceContainer._lock.acquire(timeout=_CONTAINER_LOCK_TIMEOUT_S)
    if not acquired:
        record_degradation(
            "runtime_manifest",
            TimeoutError("service container lock timed out during manifest role snapshot"),
            severity="warning",
            action="emitted partial runtime role manifest instead of blocking desktop boot",
            enforce_failure_policy=False,
        )
        return {
            "_manifest_snapshot": {
                "service": "_manifest_snapshot",
                "description": "Runtime manifest could not acquire the service container lock within the boot budget.",
                "criticality": "optional",
                "boot_phase": "canonical_runtime",
                "shutdown_policy": "degrade_with_receipt",
                "receipts_required": False,
                "allowed_callers": ["AuraRuntime.boot", "aura_main._boot_runtime_orchestrator"],
                "resolved_owners": [],
                "health_status": "partial_container_lock_timeout",
                "health_evidence": {},
                "violations": ["warning: container lock timeout"],
            }
        }
    try:
        registered = {
            name: desc.instance
            for name, desc in ServiceContainer._services.items()
            if desc.instance is not None
        }
    finally:
        ServiceContainer._lock.release()

    violations = verify_manifest(registered)
    by_role: dict[str, list[str]] = {}
    for violation in violations:
        by_role.setdefault(violation.role, []).append(
            f"{violation.severity}: {violation.reason}"
        )

    roles: dict[str, dict[str, Any]] = {}
    for role_name, role in SERVICE_MANIFEST.items():
        candidates = [role.canonical_owner, *sorted(role.aliases)]
        resolved = [name for name in candidates if name in registered]
        health_keys = _ROLE_HEALTH_KEYS.get(role_name, ())
        health_evidence = {
            key: contract_health.get(key)
            for key in health_keys
        }
        if by_role.get(role_name):
            health_status = "violation"
        elif not resolved:
            health_status = "missing"
        elif health_keys:
            probes_ok = all(
                isinstance(health_evidence.get(key), dict)
                and bool(health_evidence[key].get("present", False))
                and str(health_evidence[key].get("liveness", "") or "") == "ok"
                for key in health_keys
            )
            health_status = "ok" if probes_ok else "liveness_failed"
        else:
            health_status = "registered_unchecked"
        roles[role_name] = {
            "service": role.canonical_owner,
            "description": role.description,
            "criticality": "critical" if role.critical else "optional",
            "boot_phase": "canonical_runtime",
            "shutdown_policy": "fail_closed" if role.critical else "degrade_with_receipt",
            "receipts_required": role_name
            in {
                "governance",
                "memory_writer",
                "memory_interface",
                "state_writer",
                "model",
                "runtime",
                "event_bus",
                "output_gate",
            },
            "allowed_callers": ["AuraRuntime.boot", "aura_main._boot_runtime_orchestrator"],
            "resolved_owners": resolved,
            "health_status": health_status,
            "health_evidence": health_evidence,
            "violations": by_role.get(role_name, []),
        }
    return roles


def build_runtime_manifest(
    *,
    profile: str,
    ready_label: str,
    project_root: Path,
    artifact_root: Path,
    readiness_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from core.runtime.launch_provenance import collect_runtime_launch_provenance

    health_by_key = _health_contract_snapshot(readiness_snapshot)
    services = _service_snapshot(health_by_key)
    roles = _role_snapshot(health_by_key)
    launch_provenance = collect_runtime_launch_provenance(project_root)
    payload = {
        "schema": "aura.runtime_manifest.v1",
        "generated_at_unix": time.time(),
        "commit_sha": _git_commit(project_root),
        "profile": profile,
        "ready_label": ready_label,
        "python": sys.version,
        "platform": platform.platform(),
        "launch_provenance": launch_provenance,
        "models": {
            "AURA_LOCAL_BACKEND": os.environ.get("AURA_LOCAL_BACKEND", ""),
            "AURA_MODEL": os.environ.get("AURA_MODEL", ""),
        },
        "services": services,
        "service_roles": roles,
        "gateways": {
            name: role
            for name, role in roles.items()
            if role["receipts_required"] or name in {"autonomy", "task_supervisor"}
        },
        "enabled_subsystems": {
            name: data
            for name, data in services.items()
            if data.get("health_status")
            not in {"missing", "optional_missing", "liveness_failed"}
        },
        "disabled_subsystems": {
            name: data
            for name, data in services.items()
            if data.get("health_status")
            in {"missing", "optional_missing", "liveness_failed"}
        },
        "policy": {
            "strict_runtime": os.environ.get("AURA_STRICT_RUNTIME", "0"),
            "foreground_only": os.environ.get("AURA_FOREGROUND_ONLY", "0"),
            "test_mode": os.environ.get("AURA_TEST_MODE", "0"),
        },
        "readiness_snapshot": readiness_snapshot
        or {
            "ready": "unknown",
            "status": "not_captured",
            "required_probe_blockers": ["readiness_snapshot_missing"],
        },
        "artifact_root": str(artifact_root),
    }
    serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(serialized).hexdigest()
    return payload


def write_runtime_manifest(
    *,
    profile: str,
    ready_label: str,
    project_root: Path,
    artifact_root: Path,
    readiness_snapshot: dict[str, Any] | None = None,
) -> Path:
    artifact_root.mkdir(parents=True, exist_ok=True)
    manifest = build_runtime_manifest(
        profile=profile,
        ready_label=ready_label,
        project_root=project_root,
        artifact_root=artifact_root,
        readiness_snapshot=readiness_snapshot,
    )
    path = artifact_root / "runtime_manifest.json"
    atomic_write_text(path, json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return path
