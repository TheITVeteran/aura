#!/usr/bin/env python3
"""Publish, inspect, or revoke typed recurrent serving authority."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import stat
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.brain.llm.unified_recurrent_qualified_activation import (  # noqa: E402
    QUALIFIED_CANARY_SCHEMA,
    activation_matches_shadow_receipt,
    candidate_activation_from_pending,  # noqa: F401 - promotion API re-export
    pending_activation_from_serving,  # noqa: F401 - promotion API re-export
    qualified_serving_canary_errors,
    seal_qualified_activation,
    seal_serving_qualified_activation,
    seal_verified_qualified_activation,
)
from core.brain.llm.unified_recurrent_qualified_activation_store import (  # noqa: E402
    deactivate_qualified_activation,
    default_qualified_activation_path,
    publish_qualified_activation,
    read_qualified_activation,
)
from core.brain.llm.unified_recurrent_shadow import (  # noqa: E402
    inspect_shadow_package,
)
from core.brain.llm.unified_recurrent_shadow_battery import (  # noqa: E402
    validate_shadow_canary_battery,
)
from core.brain.llm.unified_recurrent_shadow_pointer import (  # noqa: E402
    deactivate_shadow_pointer,
    default_shadow_activation_paths,
    publish_shadow_pointer,
    read_shadow_pointer,
    resolve_shadow_pointer,
)
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402

MAX_LIFECYCLE_BYTES: Final = 4 * 1024 * 1024
class UnifiedRecurrentQualifiedActivationCommandError(RuntimeError):
    """A qualified activation operation could not preserve exact custody."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _publish_private_result(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().absolute()
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent = destination.parent
    metadata = parent.stat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "qualified canary output custody differs"
        )
    current = Path(parent.anchor)
    for part in parent.parts[1:]:
        current /= part
        if current.is_symlink():
            raise UnifiedRecurrentQualifiedActivationCommandError(
                "qualified canary output path contains a symlink"
            )
    payload = _canonical_bytes(value)
    descriptor = -1
    created = False
    completed = False
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o400,
        )
        created = True
        if os.write(descriptor, payload) != len(payload):
            raise UnifiedRecurrentQualifiedActivationCommandError(
                "qualified canary result write was short"
            )
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        completed = True
    except OSError as exc:
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "qualified canary result publication failed"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if created and not completed:
            try:
                destination.unlink()
            except FileNotFoundError:
                pass


def _read_lifecycle(path: Path) -> dict[str, Any]:
    target = path.expanduser().absolute()
    if target.is_symlink():
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "qualified lifecycle result is a symlink"
        )
    try:
        before = target.stat()
        payload = read_stable_bytes(target, max_bytes=MAX_LIFECYCLE_BYTES)
        after = target.stat()
        value = json.loads(payload.decode("ascii"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "qualified lifecycle result is unreadable"
        ) from exc
    identity = lambda item: (  # noqa: E731
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or not 0 < before.st_size <= MAX_LIFECYCLE_BYTES
        or identity(before) != identity(after)
        or not isinstance(value, dict)
        or payload not in {_canonical_bytes(value), _canonical_bytes(value) + b"\n"}
    ):
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "qualified lifecycle result custody differs"
        )
    return value


def _paths(arguments: argparse.Namespace) -> tuple[Path, Path, Path]:
    default_pointer, default_releases = default_shadow_activation_paths()
    pointer = (
        arguments.pointer.expanduser().absolute()
        if arguments.pointer is not None
        else default_pointer
    )
    releases = (
        arguments.releases_root.expanduser().absolute()
        if arguments.releases_root is not None
        else default_releases
    )
    activation = (
        arguments.activation.expanduser().absolute()
        if arguments.activation is not None
        else default_qualified_activation_path()
    )
    if pointer.parent != activation.parent:
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "qualified activation and shadow pointer custody differ"
        )
    return pointer, releases, activation


def _shadow_receipt(
    manifest: Mapping[str, Any],
    *,
    controller_sha256: str,
) -> dict[str, Any]:
    domain = manifest.get("domain_contract")
    if not isinstance(domain, Mapping):
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "qualified package domain contract is unavailable"
        )
    return {
        "package_id": manifest.get("package_id"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "controller_sha256": controller_sha256,
        "families": domain.get("families"),
        "task_depths": domain.get("task_depths"),
        "recurrence_depth": domain.get("recurrence_depth"),
    }


def _activate(arguments: argparse.Namespace) -> dict[str, Any]:
    raise UnifiedRecurrentQualifiedActivationCommandError(
        "unverified activation is disabled; use activate-verified"
    )


async def _run_qualified_canary(
    arguments: argparse.Namespace,
    activation_result: Mapping[str, Any],
    *,
    candidate_activation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Cold-load serving authority and prove every sealed typed case end to end."""

    verified = await asyncio.to_thread(
        inspect_shadow_package,
        arguments.package.expanduser().absolute(),
    )
    manifest = verified.get("manifest")
    battery_value = verified.get("canary_battery")
    if not isinstance(manifest, Mapping) or not isinstance(battery_value, Mapping):
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "qualified canary package evidence is unavailable"
        )
    battery = validate_shadow_canary_battery(battery_value)
    cases = battery["cases"]
    activation = (
        dict(candidate_activation)
        if isinstance(candidate_activation, Mapping)
        else await asyncio.to_thread(
            read_qualified_activation,
            Path(str(activation_result["activation_path"])),
        )
    )
    if activation.get("activation_sha256") != activation_result.get(
        "activation_sha256"
    ):
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "qualified canary activation identity differs"
        )

    from core.brain.llm.mlx_client import get_mlx_client

    client = get_mlx_client(str(arguments.model.expanduser().absolute()))
    primary_error: BaseException | None = None
    started_at = time.time()
    evidence: list[dict[str, Any]] = []
    total_latency_ms = 0
    try:
        ready = await client.warmup(
            foreground_request=True,
            skip_swap_cooldown=True,
        )
        if not ready:
            raise UnifiedRecurrentQualifiedActivationCommandError(
                "qualified canary resident worker did not become ready"
            )
        qualified_status = dict(
            getattr(client, "_unified_recurrent_qualified_activation_status", {})
            or {}
        )
        loaded_activation = qualified_status.get("activation")
        if candidate_activation is None:
            if (
                qualified_status.get("loaded") is not True
                or qualified_status.get("serving_authority") is not True
                or not isinstance(loaded_activation, Mapping)
                or loaded_activation.get("activation_sha256")
                != activation["activation_sha256"]
            ):
                raise UnifiedRecurrentQualifiedActivationCommandError(
                    "qualified canary worker did not load exact serving authority"
                )
        else:
            pending_matches = bool(
                qualified_status.get("loaded") is False
                and qualified_status.get("serving_authority") is False
                and isinstance(loaded_activation, Mapping)
                and dict(loaded_activation) == activation
                and activation.get("mode") == "qualified_typed_pending"
            )
            no_persisted_authority = bool(
                qualified_status.get("loaded") is False
                and qualified_status.get("serving_authority") is False
                and loaded_activation is None
            )
            if not (pending_matches or no_persisted_authority):
                raise UnifiedRecurrentQualifiedActivationCommandError(
                    "qualified canary requires exact inert authority state"
                )
        for index, case in enumerate(cases):
            began = time.monotonic_ns()
            if candidate_activation is None:
                result = await client.unified_recurrent_qualified_decode_async(
                    case["public_token_ids"],
                    family=case["family"],
                    task_depth=case["task_depth"],
                    max_tokens=case["max_tokens"],
                    timeout_s=arguments.case_timeout,
                )
            else:
                result = await client.unified_recurrent_qualified_canary_decode_async(
                    case["public_token_ids"],
                    family=case["family"],
                    task_depth=case["task_depth"],
                    max_tokens=case["max_tokens"],
                    activation=activation,
                    battery_sha256=battery["battery_sha256"],
                    case_index=index,
                    nonce=secrets.token_hex(32),
                    timeout_s=arguments.case_timeout,
                )
            elapsed_ms = max(0, (time.monotonic_ns() - began) // 1_000_000)
            receipt = result.get("receipt") if isinstance(result, Mapping) else None
            exact = bool(
                isinstance(receipt, Mapping)
                and result.get("ok") is True
                and result.get("status") == "completed"
                and receipt.get("generated_token_ids") == case["expected_token_ids"]
                and receipt.get("family") == case["family"]
                and receipt.get("task_depth") == case["task_depth"]
                and receipt.get("qualified_activation_sha256")
                == activation["activation_sha256"]
            )
            total_latency_ms += elapsed_ms
            evidence.append(
                {
                    "index": index,
                    "task_id": case["task_id"],
                    "family": case["family"],
                    "task_depth": case["task_depth"],
                    "request_sha256": case["request_sha256"],
                    "expected_token_ids_sha256": _canonical_sha256(
                        case["expected_token_ids"]
                    ),
                    "generated_token_ids_sha256": _canonical_sha256(
                        receipt.get("generated_token_ids")
                        if isinstance(receipt, Mapping)
                        else None
                    ),
                    "qualified_result_sha256": (
                        str(receipt.get("result_sha256") or "")
                        if isinstance(receipt, Mapping)
                        else ""
                    ),
                    "latency_ms": elapsed_ms,
                    "exact": exact,
                }
            )
            if not exact:
                reason = (
                    str(result.get("reason") or "qualified_result_not_exact")
                    if isinstance(result, Mapping)
                    else "qualified_result_unavailable"
                )
                raise UnifiedRecurrentQualifiedActivationCommandError(
                    f"qualified canary case {index} failed: {reason}"
                )
    except BaseException as exc:  # noqa: BLE001 - preserve close failure context
        primary_error = exc
        raise
    finally:
        try:
            await client.aclose()
        except BaseException as close_exc:  # noqa: BLE001
            if primary_error is None:
                raise
            primary_error.add_note(
                f"qualified canary resident worker close also failed: {close_exc}"
            )

    body = {
        "schema": QUALIFIED_CANARY_SCHEMA,
        "package_id": manifest["package_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "controller_sha256": activation["controller_sha256"],
        "activation_sha256": activation["activation_sha256"],
        "battery_sha256": battery["battery_sha256"],
        "started_at_unix": started_at,
        "completed_at_unix": time.time(),
        "case_count": len(cases),
        "exact_count": sum(row["exact"] for row in evidence),
        "total_latency_ms": total_latency_ms,
        "maximum_latency_ms": max((row["latency_ms"] for row in evidence), default=0),
        "evidence": evidence,
        "supported": len(evidence) == len(cases) and all(row["exact"] for row in evidence),
        "serving_authority": candidate_activation is None,
        "authority_remains_active": candidate_activation is None,
        "canary_authority_was_request_scoped": candidate_activation is not None,
        "output_exposed": False,
    }
    result = {**body, "result_sha256": _canonical_sha256(body)}
    errors = qualified_serving_canary_errors(
        result,
        expected_activation=activation,
        expected_battery=battery,
    )
    if errors:
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "qualified canary result is invalid:" + ",".join(errors)
        )
    _publish_private_result(arguments.canary_output, result)
    return result


def _candidate_canary_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.candidate{path.suffix}")


def _quarantine_existing_canary(path: Path) -> None:
    if path.exists() or path.is_symlink():
        os.replace(
            path,
            path.with_name(f"{path.name}.interrupted-{time.time_ns()}"),
        )


def _activate_verified(arguments: argparse.Namespace) -> dict[str, Any]:
    """Prove request-scoped IPC before publishing durable serving authority."""

    pointer_path, releases_root, activation_path = _paths(arguments)
    package = arguments.package.expanduser().absolute()
    verified = inspect_shadow_package(package)
    manifest = verified.get("manifest")
    if not isinstance(manifest, Mapping):
        # The package inspector deliberately exposes only a receipt. Reopen
        # the already-bound manifest for activation sealing.
        from tools.materialize_unified_intrinsic_shadow_package import _read_document

        manifest = _read_document(package / "manifest.json")
    lifecycle = _read_lifecycle(arguments.lifecycle_result)
    if pointer_path.exists() or pointer_path.is_symlink():
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "verified activation requires an inactive shadow pointer"
        )
    if activation_path.exists() or activation_path.is_symlink():
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "verified activation requires inactive durable authority"
        )
    final_canary_path = arguments.canary_output.expanduser().absolute()
    candidate_canary_path = _candidate_canary_path(final_canary_path)
    _quarantine_existing_canary(candidate_canary_path)
    _quarantine_existing_canary(final_canary_path)
    pointer: dict[str, Any] | None = None
    published: dict[str, Any] | None = None
    try:
        pointer = publish_shadow_pointer(
            package,
            pointer_path=pointer_path,
            releases_root=releases_root,
            expected_current_sha256=arguments.expected_current_pointer_sha256,
        )
        candidate = seal_qualified_activation(manifest, lifecycle, pointer)
        staged = {
            "action": "activate_verified_staged",
            "active": False,
            "package": str(package),
            "pointer_path": str(pointer_path),
            "activation_path": str(activation_path),
            "pointer_sha256": pointer["pointer_sha256"],
            "activation_sha256": candidate["activation_sha256"],
            "mode": candidate["mode"],
            "families": candidate["families"],
            "task_depths": candidate["task_depths"],
        }
        candidate_arguments = argparse.Namespace(
            **{**vars(arguments), "canary_output": candidate_canary_path}
        )
        candidate_canary = asyncio.run(
            _run_qualified_canary(
                candidate_arguments,
                staged,
                candidate_activation=candidate,
            )
        )
        if candidate_canary.get("supported") is not True:
            raise UnifiedRecurrentQualifiedActivationCommandError(
                "qualified request-scoped canary did not pass"
            )
        pending = seal_verified_qualified_activation(candidate, candidate_canary)
        published = publish_qualified_activation(
            pending,
            activation_path=activation_path,
            shadow_pointer_path=pointer_path,
            expected_current_sha256=arguments.expected_current_activation_sha256,
        )
        if (
            resolve_shadow_pointer(pointer_path, releases_root=releases_root) != package
            or not activation_matches_shadow_receipt(
                published,
                _shadow_receipt(
                    manifest,
                    controller_sha256=str(lifecycle.get("controller_sha256") or ""),
                ),
            )
        ):
            raise UnifiedRecurrentQualifiedActivationCommandError(
                "verified activation reopened a different package identity"
            )
        pending_staged = {
            **staged,
            "active": False,
            "activation_sha256": published["activation_sha256"],
            "mode": published["mode"],
        }
        canary = asyncio.run(
            _run_qualified_canary(
                arguments,
                pending_staged,
                candidate_activation=published,
            )
        )
        if (
            canary.get("supported") is not True
            or canary.get("serving_authority") is not False
            or canary.get("authority_remains_active") is not False
            or canary.get("canary_authority_was_request_scoped") is not True
        ):
            raise UnifiedRecurrentQualifiedActivationCommandError(
                "qualified persisted-pending cold-load canary did not pass"
            )
        durable = seal_serving_qualified_activation(published, canary)
        published = publish_qualified_activation(
            durable,
            activation_path=activation_path,
            shadow_pointer_path=pointer_path,
            expected_current_sha256=pending["activation_sha256"],
        )
    except BaseException as exc:  # noqa: BLE001 - rollback must follow cancellation too
        if published is not None:
            try:
                deactivate_qualified_activation(
                    activation_path=activation_path,
                    expected_current_sha256=published["activation_sha256"],
                )
            except BaseException as rollback_exc:  # noqa: BLE001
                exc.add_note(
                    f"qualified canary authority rollback also failed: {rollback_exc}"
                )
        if pointer is not None:
            try:
                deactivate_shadow_pointer(
                    pointer_path=pointer_path,
                    releases_root=releases_root,
                    expected_current_sha256=pointer["pointer_sha256"],
                )
            except BaseException as rollback_exc:  # noqa: BLE001
                exc.add_note(
                    f"qualified canary pointer rollback also failed: {rollback_exc}"
                )
        for canary_path in (candidate_canary_path, final_canary_path):
            try:
                _quarantine_existing_canary(canary_path)
            except BaseException as rollback_exc:  # noqa: BLE001
                exc.add_note(
                    f"qualified canary artifact quarantine also failed: {rollback_exc}"
                )
        raise
    if pointer is None or published is None:  # pragma: no cover - guarded above
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "verified activation publication returned no authority"
        )
    return {
        "action": "activate_verified",
        "active": True,
        "package": str(package),
        "pointer_path": str(pointer_path),
        "activation_path": str(activation_path),
        "pointer_sha256": pointer["pointer_sha256"],
        "activation_sha256": published["activation_sha256"],
        "mode": published["mode"],
        "families": published["families"],
        "task_depths": published["task_depths"],
        "canary": canary,
        "candidate_canary": candidate_canary,
    }


def _deactivate(arguments: argparse.Namespace) -> dict[str, Any]:
    pointer_path, releases_root, activation_path = _paths(arguments)
    activation = deactivate_qualified_activation(
        activation_path=activation_path,
        expected_current_sha256=arguments.expected_current_activation_sha256,
    )
    pointer = deactivate_shadow_pointer(
        pointer_path=pointer_path,
        releases_root=releases_root,
        expected_current_sha256=arguments.expected_current_pointer_sha256,
    )
    if activation["pointer_sha256"] != pointer["pointer_sha256"]:
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "retired qualified activation and pointer identity differ"
        )
    return {
        "action": "deactivate",
        "active": False,
        "activation_sha256": activation["activation_sha256"],
        "pointer_sha256": pointer["pointer_sha256"],
        "pointer_path": str(pointer_path),
        "activation_path": str(activation_path),
    }


def _status(arguments: argparse.Namespace) -> dict[str, Any]:
    pointer_path, releases_root, activation_path = _paths(arguments)
    has_pointer = pointer_path.exists() or pointer_path.is_symlink()
    has_activation = activation_path.exists() or activation_path.is_symlink()
    if not has_activation:
        return {
            "action": "status",
            "active": False,
            "shadow_pointer_active": has_pointer,
            "pointer_path": str(pointer_path),
            "activation_path": str(activation_path),
        }
    if not has_pointer:
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "qualified authority exists without a shadow pointer"
        )
    activation = read_qualified_activation(activation_path)
    pointer = read_shadow_pointer(pointer_path)
    package = resolve_shadow_pointer(pointer_path, releases_root=releases_root)
    manifest = inspect_shadow_package(package).get("manifest")
    if (
        not isinstance(manifest, Mapping)
        or activation["pointer_sha256"] != pointer["pointer_sha256"]
        or not activation_matches_shadow_receipt(
            activation,
            _shadow_receipt(
                manifest,
                controller_sha256=str(activation.get("controller_sha256") or ""),
            ),
        )
    ):
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "qualified authority no longer matches its active package"
        )
    return {
        "action": "status",
        "active": True,
        "package": str(package),
        "pointer_path": str(pointer_path),
        "activation_path": str(activation_path),
        "pointer_sha256": pointer["pointer_sha256"],
        "activation_sha256": activation["activation_sha256"],
        "mode": activation["mode"],
        "families": activation["families"],
        "task_depths": activation["task_depths"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointer", type=Path)
    parser.add_argument("--releases-root", type=Path)
    parser.add_argument("--activation", type=Path)
    subparsers = parser.add_subparsers(dest="action", required=True)

    activate_verified = subparsers.add_parser("activate-verified")
    activate_verified.add_argument("package", type=Path)
    activate_verified.add_argument("--lifecycle-result", type=Path, required=True)
    activate_verified.add_argument("--model", type=Path, required=True)
    activate_verified.add_argument("--canary-output", type=Path, required=True)
    activate_verified.add_argument("--case-timeout", type=float, default=180.0)
    activate_verified.add_argument("--expected-current-pointer-sha256")
    activate_verified.add_argument("--expected-current-activation-sha256")

    deactivate = subparsers.add_parser("deactivate")
    deactivate.add_argument("--expected-current-pointer-sha256", required=True)
    deactivate.add_argument("--expected-current-activation-sha256", required=True)

    subparsers.add_parser("status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = {
            "activate-verified": _activate_verified,
            "deactivate": _deactivate,
            "status": _status,
        }[arguments.action](arguments)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"unified qualified activation operation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
