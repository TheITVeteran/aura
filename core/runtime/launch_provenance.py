"""Build, verify, and expose the provenance of a launched Aura desktop runtime.

The installed ``Aura.app`` is intentionally a thin launcher over live source.
That makes source identity a runtime contract: the app must prove which root,
commit, and exact dirty workspace state it was built to launch before cleanup
or process replacement can occur.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.subprocess_gateway import get_subprocess_gateway

LAUNCH_PROVENANCE_SCHEMA = "aura.launch_provenance.v1"
EXPECTED_BUNDLE_ID = "com.aura.desktop"

_SOURCE_PATHS = (
    "aura_main.py",
    "aura_cleanup.py",
    "main_daemon.py",
    "launch_aura.sh",
    "build_app.sh",
    "pyproject.toml",
    "requirements.txt",
    "requirements_hardened.txt",
    "requirements_lock.txt",
    "aura",
    "autonomy_engine",
    "cloud",
    "config",
    "core",
    "executors",
    "infrastructure",
    "integration",
    "interface",
    "llm",
    "memory",
    "native",
    "optimizer",
    "proof_kernel",
    "rust_extensions",
    "scoping",
    "scripts",
    "security",
    "senses",
    "skills",
    "storage",
    "utils",
)
_SOURCE_CACHE_TTL_S = 2.0
_SOURCE_CACHE_LOCK = threading.Lock()
_SOURCE_CACHE: dict[tuple[str, str, str], tuple[float, dict[str, Any]]] = {}
_BUNDLE_CACHE_LOCK = threading.Lock()
_BUNDLE_CACHE: dict[str, tuple[int, dict[str, Any]]] = {}
_RECOVERABLE_ERRORS = (
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _run_git(root: Path, arguments: Sequence[str], *, timeout: float = 3.0) -> str:
    completed = get_subprocess_gateway().run(
        ["git", "-C", str(root), *arguments],
        timeout=timeout,
        read_only=True,
        capture_output=True,
        source="runtime_launch_provenance.git",
    )
    if completed.returncode != 0:
        detail = str(completed.stderr or completed.stdout or "git command failed").strip()
        raise RuntimeError(detail[:500])
    return str(completed.stdout or "")


def _git_identity(root: Path) -> dict[str, str]:
    canonical_root = Path(
        _run_git(root, ("rev-parse", "--show-toplevel"))
        .strip()
    ).expanduser().resolve()
    commit = _run_git(canonical_root, ("rev-parse", "HEAD")).strip()
    branch_result = get_subprocess_gateway().run(
        ["git", "-C", str(canonical_root), "symbolic-ref", "--quiet", "--short", "HEAD"],
        timeout=3.0,
        read_only=True,
        capture_output=True,
        source="runtime_launch_provenance.git_branch",
    )
    branch = str(branch_result.stdout or "").strip() if branch_result.returncode == 0 else "DETACHED"
    if len(commit) != 40 or any(character not in "0123456789abcdefABCDEF" for character in commit):
        raise RuntimeError("git HEAD did not resolve to a full commit SHA")
    return {
        "source_root": str(canonical_root),
        "commit_sha": commit.lower(),
        "branch": branch,
    }


def _status_paths(status_output: str) -> list[str]:
    records = status_output.split("\0")
    paths: list[str] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(record) < 4 or record[2] != " ":
            continue
        status = record[:2]
        path = record[3:]
        if path:
            paths.append(path)
        if any(marker in status for marker in ("R", "C")) and index < len(records):
            prior_path = records[index]
            index += 1
            if prior_path:
                paths.append(prior_path)
    return sorted(set(paths))


def _hash_workspace_state(root: Path, status_output: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    digest.update(b"aura-workspace-state-v1\0")
    digest.update(status_output.encode("utf-8", errors="surrogateescape"))
    paths = _status_paths(status_output)
    for relative in paths:
        raw_candidate = root / relative
        candidate = raw_candidate.resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError:
            digest.update(b"outside-root\0" + relative.encode("utf-8", errors="surrogateescape"))
            continue
        encoded_path = relative.encode("utf-8", errors="surrogateescape")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        if raw_candidate.is_symlink():
            target = os.readlink(raw_candidate)
            digest.update(b"symlink\0" + target.encode("utf-8", errors="surrogateescape"))
        elif candidate.is_file():
            digest.update(b"file\0")
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            digest.update(b"missing\0")
    return {
        "workspace_state_sha256": digest.hexdigest(),
        "source_dirty": bool(status_output),
        "source_change_count": len(paths),
        "source_changed_paths": paths[:128],
        "source_changed_paths_truncated": len(paths) > 128,
    }


def _workspace_state(root: Path, *, commit_sha: str) -> dict[str, Any]:
    cache_key = (str(root), commit_sha, os.getenv("AURA_LAUNCH_MANIFEST_PATH", ""))
    now = time.monotonic()
    with _SOURCE_CACHE_LOCK:
        cached = _SOURCE_CACHE.get(cache_key)
        if cached is not None and now - cached[0] < _SOURCE_CACHE_TTL_S:
            return dict(cached[1])
    output = _run_git(
        root,
        (
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--",
            *_SOURCE_PATHS,
        ),
        timeout=8.0,
    )
    result = _hash_workspace_state(root, output)
    with _SOURCE_CACHE_LOCK:
        _SOURCE_CACHE[cache_key] = (time.monotonic(), dict(result))
    return result


def build_launch_manifest(
    root: str | Path,
    *,
    version: str,
    launcher_source: str | Path,
) -> dict[str, Any]:
    canonical_input = Path(root).expanduser().resolve()
    identity = _git_identity(canonical_input)
    canonical_root = Path(identity["source_root"])
    workspace = _workspace_state(canonical_root, commit_sha=identity["commit_sha"])
    launcher_path = Path(launcher_source).expanduser().resolve()
    launcher_bytes = launcher_path.read_bytes()
    return {
        "schema": LAUNCH_PROVENANCE_SCHEMA,
        "generated_at_unix": time.time(),
        "version": str(version),
        **identity,
        **workspace,
        "launcher_source": str(launcher_path.relative_to(canonical_root)),
        "launcher_source_sha256": hashlib.sha256(launcher_bytes).hexdigest(),
        "bundle_identifier": EXPECTED_BUNDLE_ID,
    }


def write_launch_manifest(
    output: str | Path,
    *,
    root: str | Path,
    version: str,
    launcher_source: str | Path,
) -> dict[str, Any]:
    manifest = build_launch_manifest(root, version=version, launcher_source=launcher_source)
    atomic_write_text(
        output,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("launch manifest must contain a JSON object")
    return payload


def _bundle_for_executable(executable: Path) -> Path | None:
    parts = executable.resolve(strict=False).parts
    for index, part in enumerate(parts):
        if part.endswith(".app"):
            return Path(*parts[: index + 1])
    return None


def _manifest_belongs_to_executable(manifest_path: Path, executable: Path) -> bool:
    bundle = _bundle_for_executable(executable)
    if bundle is None:
        return False
    expected = bundle / "Contents" / "Resources" / "aura-launch-provenance.json"
    return manifest_path.resolve(strict=False) == expected.resolve(strict=False)


def validate_launch_source(
    root: str | Path,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    environment = os.environ if env is None else env
    canonical_input = Path(root).expanduser().resolve()
    launched_from_app = _truthy(environment.get("AURA_LAUNCHED_FROM_APP"))
    if not launched_from_app:
        return {
            "schema": LAUNCH_PROVENANCE_SCHEMA,
            "required": False,
            "launch_mode": "direct",
            "source_verified": True,
            "verified": False,
            "source_root": str(canonical_input),
            "issues": [],
        }

    issues: list[str] = []
    manifest_path_text = str(environment.get("AURA_LAUNCH_MANIFEST_PATH") or "").strip()
    executable_text = str(environment.get("AURA_LAUNCH_APP_EXECUTABLE") or "").strip()
    expected_root = str(environment.get("AURA_LAUNCH_EXPECTED_ROOT") or "").strip()
    expected_commit = str(environment.get("AURA_LAUNCH_EXPECTED_COMMIT") or "").strip().lower()
    expected_branch = str(environment.get("AURA_LAUNCH_EXPECTED_BRANCH") or "").strip()
    expected_workspace = str(
        environment.get("AURA_LAUNCH_EXPECTED_WORKSPACE_SHA256") or ""
    ).strip().lower()
    expected_bundle_id = str(environment.get("AURA_LAUNCH_BUNDLE_ID") or "").strip()

    required_values = {
        "manifest_path": manifest_path_text,
        "app_executable": executable_text,
        "expected_root": expected_root,
        "expected_commit": expected_commit,
        "expected_branch": expected_branch,
        "expected_workspace_sha256": expected_workspace,
        "bundle_identifier": expected_bundle_id,
    }
    for name, value in required_values.items():
        if not value:
            issues.append(f"missing_{name}")

    manifest: dict[str, Any] = {}
    manifest_path = Path(manifest_path_text).expanduser() if manifest_path_text else None
    executable = Path(executable_text).expanduser() if executable_text else None
    if manifest_path is not None:
        try:
            manifest = _load_manifest(manifest_path)
        except _RECOVERABLE_ERRORS as exc:
            issues.append(f"manifest_unreadable:{type(exc).__name__}")
    if manifest and manifest.get("schema") != LAUNCH_PROVENANCE_SCHEMA:
        issues.append("manifest_schema_mismatch")
    if executable is not None and manifest_path is not None:
        if not executable.is_file():
            issues.append("app_executable_missing")
        if not _manifest_belongs_to_executable(manifest_path, executable):
            issues.append("manifest_outside_app_bundle")

    actual: dict[str, Any] = {}
    try:
        identity = _git_identity(canonical_input)
        actual.update(identity)
        actual.update(_workspace_state(Path(identity["source_root"]), commit_sha=identity["commit_sha"]))
    except _RECOVERABLE_ERRORS as exc:
        issues.append(f"source_identity_unavailable:{type(exc).__name__}")

    comparisons = {
        "source_root": (str(Path(expected_root).expanduser().resolve()) if expected_root else "", actual.get("source_root")),
        "commit_sha": (expected_commit, actual.get("commit_sha")),
        "branch": (expected_branch, actual.get("branch")),
        "workspace_state_sha256": (expected_workspace, actual.get("workspace_state_sha256")),
        "bundle_identifier": (expected_bundle_id, EXPECTED_BUNDLE_ID),
    }
    for field, (expected, observed) in comparisons.items():
        if expected and str(expected) != str(observed or ""):
            issues.append(f"{field}_mismatch")
        manifest_value = manifest.get(field) if manifest else None
        if expected and str(expected) != str(manifest_value or ""):
            issues.append(f"manifest_{field}_mismatch")

    source_verified = not issues
    return {
        "schema": LAUNCH_PROVENANCE_SCHEMA,
        "required": True,
        "launch_mode": "signed_app",
        "source_verified": source_verified,
        "verified": False,
        "issues": sorted(set(issues)),
        "manifest_path": str(manifest_path or ""),
        "app_executable": str(executable or ""),
        "expected": {
            "source_root": expected_root,
            "commit_sha": expected_commit,
            "branch": expected_branch,
            "workspace_state_sha256": expected_workspace,
            "bundle_identifier": expected_bundle_id,
        },
        "actual": actual,
        "manifest": manifest,
    }


def _strict_bundle_verification(executable: Path) -> dict[str, Any]:
    bundle = _bundle_for_executable(executable)
    if bundle is None or not bundle.is_dir():
        return {"ok": False, "reason": "app_bundle_missing", "bundle_path": ""}
    revision_material: list[tuple[int, int]] = []
    for path in (
        executable,
        bundle / "Contents" / "Info.plist",
        bundle / "Contents" / "_CodeSignature" / "CodeResources",
        bundle / "Contents" / "Resources" / "aura-launch-provenance.json",
    ):
        try:
            stat = path.stat()
            revision_material.append((int(stat.st_mtime_ns), int(stat.st_size)))
        except OSError:
            revision_material.append((0, 0))
    revision = hash(tuple(revision_material))
    key = str(bundle.resolve(strict=False))
    with _BUNDLE_CACHE_LOCK:
        cached = _BUNDLE_CACHE.get(key)
        if cached is not None and cached[0] == revision:
            return dict(cached[1])
    try:
        completed = get_subprocess_gateway().run(
            ["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(bundle)],
            timeout=3.0,
            read_only=True,
            capture_output=True,
            source="runtime_launch_provenance.codesign_verify",
        )
        result = {
            "ok": completed.returncode == 0,
            "returncode": int(completed.returncode),
            "bundle_path": str(bundle),
            "detail": str(completed.stderr or completed.stdout or "").strip()[:500],
        }
    except _RECOVERABLE_ERRORS as exc:
        result = {
            "ok": False,
            "bundle_path": str(bundle),
            "reason": f"{type(exc).__name__}: {exc}",
        }
    with _BUNDLE_CACHE_LOCK:
        _BUNDLE_CACHE[key] = (revision, dict(result))
    return result


def collect_runtime_launch_provenance(
    root: str | Path,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    source = validate_launch_source(root, env=env)
    if not source.get("required"):
        return source

    executable_text = str(source.get("app_executable") or "").strip()
    executable = Path(executable_text).expanduser() if executable_text else Path("/__aura_missing_app__")
    try:
        from core.security.native_desktop_bridge import native_desktop_bridge_identity

        native_identity = native_desktop_bridge_identity(executable=executable)
    except _RECOVERABLE_ERRORS as exc:
        native_identity = {
            "resident_running": False,
            "code_signature": {"available": False},
            "error": f"{type(exc).__name__}: {exc}",
        }
    strict_verification = _strict_bundle_verification(executable)
    signature = native_identity.get("code_signature", {})
    if not isinstance(signature, dict):
        signature = {}
    signature_valid = bool(
        signature.get("available")
        and signature.get("stable_tcc_identity")
        and signature.get("identifier") == EXPECTED_BUNDLE_ID
    )
    resident_running = bool(native_identity.get("resident_running"))
    issues = list(source.get("issues", []))
    if not resident_running:
        issues.append("resident_app_not_running")
    if not signature_valid:
        issues.append("app_signature_unverified")
    if not strict_verification.get("ok"):
        issues.append("strict_bundle_verification_failed")
    verified = bool(
        source.get("source_verified")
        and resident_running
        and signature_valid
        and strict_verification.get("ok")
    )
    return {
        **source,
        "verified": verified,
        "issues": sorted(set(str(issue) for issue in issues if str(issue))),
        "resident_bridge": native_identity,
        "strict_bundle_verification": strict_verification,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aura launch provenance tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    emit = subparsers.add_parser("emit", help="write an Aura.app source manifest")
    emit.add_argument("--root", required=True)
    emit.add_argument("--output", required=True)
    emit.add_argument("--version", required=True)
    emit.add_argument("--launcher-source", required=True)

    preflight = subparsers.add_parser("preflight", help="verify app-pinned source before cleanup")
    preflight.add_argument("--root", required=True)

    inspect = subparsers.add_parser("inspect", help="print the current runtime launch evidence")
    inspect.add_argument("--root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "emit":
        manifest = write_launch_manifest(
            args.output,
            root=args.root,
            version=args.version,
            launcher_source=args.launcher_source,
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0
    if args.command == "preflight":
        result = collect_runtime_launch_provenance(args.root)
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("verified") else 2
    result = collect_runtime_launch_provenance(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if (not result.get("required") or result.get("verified")) else 2


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "EXPECTED_BUNDLE_ID",
    "LAUNCH_PROVENANCE_SCHEMA",
    "build_launch_manifest",
    "collect_runtime_launch_provenance",
    "validate_launch_source",
    "write_launch_manifest",
]
