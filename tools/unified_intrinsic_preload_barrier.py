#!/usr/bin/env python3
"""Block a resident model load until its exact-PID sentinel is observable."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import stat
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_detached_step as detached  # noqa: E402

READY_SCHEMA: Final = "aura.unified_intrinsic.preload_ready.v1"
RELEASE_SCHEMA: Final = "aura.unified_intrinsic.preload_release.v1"
MAX_DOCUMENT_BYTES: Final = 64 * 1024
PID_PLACEHOLDER: Final = "{pid}"
READY_MAX_AGE_S: Final = 300.0
RELEASE_VALIDITY_S: Final = 120.0
CLOCK_SKEW_S: Final = 5.0


class UnifiedPreloadBarrierError(RuntimeError):
    """The pre-load supervisor handshake is absent, stale, or forged."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_nonce(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 32 and all(
        character in "0123456789abcdef" for character in value
    )


def _stable_private_bytes(path: Path, *, max_bytes: int) -> bytes:
    if not path.is_absolute() or path.is_symlink():
        raise UnifiedPreloadBarrierError("preload artifact path is invalid")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) & 0o077
                or not 0 < before.st_size <= max_bytes
            ):
                raise UnifiedPreloadBarrierError("preload artifact custody differs")
            chunks: list[bytes] = []
            remaining = max_bytes + 1
            while remaining > 0:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise UnifiedPreloadBarrierError("preload artifact is unreadable") from exc
    raw = b"".join(chunks)
    if (
        len(raw) != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise UnifiedPreloadBarrierError("preload artifact changed while read")
    return raw


def _read_document(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _stable_private_bytes(path, max_bytes=MAX_DOCUMENT_BYTES)
    try:
        value = json.loads(raw.decode("ascii"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise UnifiedPreloadBarrierError("preload artifact JSON is invalid") from exc
    if not isinstance(value, dict) or raw != _canonical(value):
        raise UnifiedPreloadBarrierError("preload artifact is not canonical")
    return value, raw


def _write_once(path: Path, payload: Mapping[str, Any]) -> bytes:
    if not path.is_absolute() or path.is_symlink() or not path.parent.is_dir():
        raise UnifiedPreloadBarrierError("preload artifact path is invalid")
    raw = _canonical(dict(payload))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        try:
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short preload write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise UnifiedPreloadBarrierError("preload artifact create failed") from exc
    return raw


def _key(path: Path) -> bytes:
    key = _stable_private_bytes(path, max_bytes=128)
    if len(key) != 32:
        raise UnifiedPreloadBarrierError("preload HMAC key differs")
    return key


def command_sha256(command: Sequence[str]) -> str:
    if not command or any(not isinstance(value, str) or not value for value in command):
        raise UnifiedPreloadBarrierError("preload target command is invalid")
    return hashlib.sha256(_canonical(list(command))).hexdigest()


def expand_pid_template(value: str, *, target_pid: int | None = None) -> str:
    """Bind one immutable launch template to the current target identity."""

    pid = os.getpid() if target_pid is None else target_pid
    if type(pid) is not int or pid < 1 or not isinstance(value, str) or not value:
        raise UnifiedPreloadBarrierError("preload PID template is invalid")
    expanded = value.replace(PID_PLACEHOLDER, str(pid))
    if PID_PLACEHOLDER in expanded:
        raise UnifiedPreloadBarrierError("preload PID template did not resolve")
    return expanded


def publish_ready(
    path: Path,
    *,
    config_sha256: str,
    command: Sequence[str],
    target_pid: int | None = None,
) -> tuple[dict[str, Any], bytes]:
    pid = os.getpid() if target_pid is None else target_pid
    if type(pid) is not int or pid < 1 or not _is_sha(config_sha256):
        raise UnifiedPreloadBarrierError("preload ready identity is invalid")
    start_token = detached._process_start_token(pid)  # noqa: SLF001
    if not start_token:
        raise UnifiedPreloadBarrierError("preload target process is unobservable")
    body = {
        "schema": READY_SCHEMA,
        "target_pid": pid,
        "target_start_token": start_token,
        "config_sha256": config_sha256,
        "command_sha256": command_sha256(command),
        "nonce": os.urandom(16).hex(),
        "written_at_unix_ns": time.time_ns(),
    }
    raw = _write_once(path, body)
    return body, raw


def publish_release(
    release_path: Path,
    *,
    ready_path: Path,
    key_path: Path,
    sentinel_pid: int,
    sentinel_start_token: str,
    sentinel_ring_entry_sha256: str,
    host_pressure: Mapping[str, Any],
    expected_target_pid: int,
    expected_target_start_token: str,
    expected_command_sha256: str,
) -> dict[str, Any]:
    ready, ready_raw = _read_document(ready_path)
    if type(expected_target_pid) is not int or expected_target_pid < 1:
        raise UnifiedPreloadBarrierError("expected preload target PID is invalid")
    if ready.get("target_pid") != expected_target_pid:
        raise UnifiedPreloadBarrierError("preload ready target PID differs")
    if (
        not isinstance(expected_target_start_token, str)
        or not expected_target_start_token
        or ready.get("target_start_token") != expected_target_start_token
        or detached._identity_state(  # noqa: SLF001
            expected_target_pid,
            expected_target_start_token,
        )
        != "alive"
    ):
        raise UnifiedPreloadBarrierError("preload ready target incarnation differs")
    if not _is_sha(expected_command_sha256):
        raise UnifiedPreloadBarrierError("expected preload command identity is invalid")
    if ready.get("command_sha256") != expected_command_sha256:
        raise UnifiedPreloadBarrierError("preload ready command identity differs")
    written_at_unix_ns = ready.get("written_at_unix_ns")
    issued_at_unix_ns = time.time_ns()
    if (
        set(ready)
        != {
            "schema",
            "target_pid",
            "target_start_token",
            "config_sha256",
            "command_sha256",
            "nonce",
            "written_at_unix_ns",
        }
        or ready.get("schema") != READY_SCHEMA
        or not _is_sha(ready.get("config_sha256"))
        or not _is_sha(ready.get("command_sha256"))
        or not _is_nonce(ready.get("nonce"))
        or type(written_at_unix_ns) is not int
        or written_at_unix_ns <= 0
        or written_at_unix_ns > issued_at_unix_ns + int(CLOCK_SKEW_S * 1e9)
        or issued_at_unix_ns - written_at_unix_ns > int(READY_MAX_AGE_S * 1e9)
        or type(sentinel_pid) is not int
        or sentinel_pid < 1
        or not isinstance(sentinel_start_token, str)
        or not sentinel_start_token
        or not _is_sha(sentinel_ring_entry_sha256)
        or not isinstance(host_pressure, Mapping)
        or host_pressure.get("available") is not True
        or host_pressure.get("under_pressure") is not False
    ):
        raise UnifiedPreloadBarrierError("preload release evidence is invalid")
    body = {
        "schema": RELEASE_SCHEMA,
        "target_pid": ready["target_pid"],
        "target_start_token": ready["target_start_token"],
        "config_sha256": ready["config_sha256"],
        "command_sha256": ready["command_sha256"],
        "ready_sha256": hashlib.sha256(ready_raw).hexdigest(),
        "sentinel_pid": sentinel_pid,
        "sentinel_start_token": sentinel_start_token,
        "sentinel_ring_entry_sha256": sentinel_ring_entry_sha256,
        "host_pressure": dict(host_pressure),
        "issued_at_unix_ns": issued_at_unix_ns,
        "expires_at_unix_ns": issued_at_unix_ns
        + int(RELEASE_VALIDITY_S * 1e9),
    }
    signature = hmac.new(_key(key_path), _canonical(body), hashlib.sha256).hexdigest()
    complete = {**body, "hmac_sha256": signature}
    _write_once(release_path, complete)
    return complete


def verify_release(
    release_path: Path,
    *,
    ready_path: Path,
    key_path: Path,
    config_sha256: str,
    expected_target_pid: int | None = None,
    expected_target_start_token: str | None = None,
    expected_command_sha256: str | None = None,
    require_fresh: bool = True,
    require_live_evidence: bool = False,
) -> dict[str, Any]:
    ready, ready_raw = _read_document(ready_path)
    release, _release_raw = _read_document(release_path)
    pid = os.getpid() if expected_target_pid is None else expected_target_pid
    start_token = (
        detached._process_start_token(pid)  # noqa: SLF001
        if expected_target_start_token is None
        else expected_target_start_token
    )
    signature = release.get("hmac_sha256")
    body = {key: value for key, value in release.items() if key != "hmac_sha256"}
    now_unix_ns = time.time_ns()
    issued_at_unix_ns = release.get("issued_at_unix_ns")
    expires_at_unix_ns = release.get("expires_at_unix_ns")
    if (
        set(ready)
        != {
            "schema",
            "target_pid",
            "target_start_token",
            "config_sha256",
            "command_sha256",
            "nonce",
            "written_at_unix_ns",
        }
        or set(release)
        != {
            "schema",
            "target_pid",
            "target_start_token",
            "config_sha256",
            "command_sha256",
            "ready_sha256",
            "sentinel_pid",
            "sentinel_start_token",
            "sentinel_ring_entry_sha256",
            "host_pressure",
            "issued_at_unix_ns",
            "expires_at_unix_ns",
            "hmac_sha256",
        }
        or ready.get("schema") != READY_SCHEMA
        or release.get("schema") != RELEASE_SCHEMA
        or ready.get("target_pid") != pid
        or release.get("target_pid") != pid
        or not isinstance(start_token, str)
        or not start_token
        or ready.get("target_start_token") != start_token
        or release.get("target_start_token") != start_token
        or ready.get("config_sha256") != config_sha256
        or release.get("config_sha256") != config_sha256
        or release.get("command_sha256") != ready.get("command_sha256")
        or (
            expected_command_sha256 is not None
            and (
                not _is_sha(expected_command_sha256)
                or ready.get("command_sha256") != expected_command_sha256
            )
        )
        or release.get("ready_sha256") != hashlib.sha256(ready_raw).hexdigest()
        or type(release.get("sentinel_pid")) is not int
        or int(release["sentinel_pid"]) < 1
        or not isinstance(release.get("sentinel_start_token"), str)
        or not release["sentinel_start_token"]
        or not _is_sha(release.get("sentinel_ring_entry_sha256"))
        or not isinstance(release.get("host_pressure"), dict)
        or release["host_pressure"].get("available") is not True
        or release["host_pressure"].get("under_pressure") is not False
        or type(issued_at_unix_ns) is not int
        or type(expires_at_unix_ns) is not int
        or issued_at_unix_ns <= 0
        or expires_at_unix_ns <= issued_at_unix_ns
        or expires_at_unix_ns - issued_at_unix_ns
        != int(RELEASE_VALIDITY_S * 1e9)
        or issued_at_unix_ns
        > now_unix_ns + int(CLOCK_SKEW_S * 1e9)
        or (require_fresh and now_unix_ns > expires_at_unix_ns)
        or (
            require_live_evidence
            and (
                detached._identity_state(pid, start_token) != "alive"  # noqa: SLF001
                or detached._identity_state(  # noqa: SLF001
                    int(release.get("sentinel_pid") or 0),
                    str(release.get("sentinel_start_token") or ""),
                )
                != "alive"
            )
        )
        or not isinstance(signature, str)
        or not hmac.compare_digest(
            signature,
            hmac.new(_key(key_path), _canonical(body), hashlib.sha256).hexdigest(),
        )
    ):
        raise UnifiedPreloadBarrierError("preload release contract differs")
    return release


def await_release(
    release_path: Path,
    *,
    ready_path: Path,
    key_path: Path,
    config_sha256: str,
    timeout_s: float,
) -> dict[str, Any]:
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise UnifiedPreloadBarrierError("preload timeout is invalid")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if release_path.exists():
            return verify_release(
                release_path,
                ready_path=ready_path,
                key_path=key_path,
                config_sha256=config_sha256,
                expected_target_start_token=detached._process_start_token(  # noqa: SLF001
                    os.getpid()
                ),
                require_live_evidence=True,
            )
        time.sleep(0.1)
    raise UnifiedPreloadBarrierError("preload release timed out")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    try:
        target_pid = os.getpid()
        ready_path = Path(
            expand_pid_template(str(args.ready.expanduser()), target_pid=target_pid)
        )
        release_path = Path(
            expand_pid_template(str(args.release.expanduser()), target_pid=target_pid)
        )
        command = [
            expand_pid_template(value, target_pid=target_pid) for value in command
        ]
        publish_ready(
            ready_path,
            config_sha256=args.config_sha256,
            command=command,
            target_pid=target_pid,
        )
        await_release(
            release_path,
            ready_path=ready_path,
            key_path=args.key.expanduser(),
            config_sha256=args.config_sha256,
            timeout_s=args.timeout,
        )
        executable = Path(command[0]).expanduser().resolve(strict=True)
        if not executable.is_file():
            raise UnifiedPreloadBarrierError("preload executable is invalid")
        os.execve(str(executable), [str(executable), *command[1:]], dict(os.environ))
    except Exception as exc:  # noqa: BLE001 - stable CLI failure boundary
        print(
            f"unified_intrinsic_preload_barrier: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    return 127


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "READY_SCHEMA",
    "RELEASE_SCHEMA",
    "UnifiedPreloadBarrierError",
    "await_release",
    "command_sha256",
    "expand_pid_template",
    "publish_ready",
    "publish_release",
    "verify_release",
]
