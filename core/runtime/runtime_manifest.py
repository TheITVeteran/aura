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

_GIT_ERRORS = (OSError, UnicodeDecodeError, ValueError)
_MANIFEST_ERRORS = (OSError, TypeError, ValueError, RuntimeError, AttributeError)


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


def _service_snapshot() -> dict[str, dict[str, Any]]:
    from core.container import ServiceContainer

    try:
        statuses = ServiceContainer.get_all_subsystem_statuses()
    except _MANIFEST_ERRORS:
        statuses = {}

    services: dict[str, dict[str, Any]] = {}
    with ServiceContainer._lock:
        items = list(ServiceContainer._services.items())

    for name, desc in items:
        instance = getattr(desc, "instance", None)
        factory = getattr(desc, "factory", None)
        owner_target = instance if instance is not None else factory
        services[name] = {
            "service": name,
            "owner": _source_owner(owner_target) if owner_target is not None else "unknown",
            "required": bool(getattr(desc, "required", False)),
            "initialized": bool(getattr(desc, "initialized", False)),
            "health_status": statuses.get(name, "registered_unchecked"),
            "dependencies": list(getattr(desc, "dependencies", []) or []),
        }
    return dict(sorted(services.items()))


def _role_snapshot() -> dict[str, dict[str, Any]]:
    from core.container import ServiceContainer
    from core.runtime.service_manifest import SERVICE_MANIFEST, verify_manifest

    with ServiceContainer._lock:
        registered = {
            name: desc.instance
            for name, desc in ServiceContainer._services.items()
            if desc.instance is not None
        }

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
                "state_writer",
                "model",
                "runtime",
                "event_bus",
                "output_gate",
            },
            "allowed_callers": ["AuraRuntime.boot", "aura_main._boot_runtime_orchestrator"],
            "resolved_owners": resolved,
            "health_status": "ok" if resolved and not by_role.get(role_name) else "violation",
            "violations": by_role.get(role_name, []),
        }
    return roles


def build_runtime_manifest(
    *,
    profile: str,
    ready_label: str,
    project_root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    services = _service_snapshot()
    roles = _role_snapshot()
    payload = {
        "schema": "aura.runtime_manifest.v1",
        "generated_at_unix": time.time(),
        "commit_sha": _git_commit(project_root),
        "profile": profile,
        "ready_label": ready_label,
        "python": sys.version,
        "platform": platform.platform(),
        "models": {
            "AURA_LOCAL_BACKEND": os.environ.get("AURA_LOCAL_BACKEND", ""),
            "AURA_MODEL": os.environ.get("AURA_MODEL", ""),
            "AURA_SOLVER_GGUF": os.environ.get("AURA_SOLVER_GGUF", ""),
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
            if data.get("health_status") not in {"missing", "optional_missing"}
        },
        "disabled_subsystems": {
            name: data
            for name, data in services.items()
            if data.get("health_status") in {"missing", "optional_missing"}
        },
        "policy": {
            "strict_runtime": os.environ.get("AURA_STRICT_RUNTIME", "0"),
            "foreground_only": os.environ.get("AURA_FOREGROUND_ONLY", "0"),
            "test_mode": os.environ.get("AURA_TEST_MODE", "0"),
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
) -> Path:
    artifact_root.mkdir(parents=True, exist_ok=True)
    manifest = build_runtime_manifest(
        profile=profile,
        ready_label=ready_label,
        project_root=project_root,
        artifact_root=artifact_root,
    )
    path = artifact_root / "runtime_manifest.json"
    atomic_write_text(path, json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return path
