#!/usr/bin/env python3
"""Run a sealed recurrent-shadow battery through Aura's real resident worker."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import stat
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.brain.llm import unified_recurrent_shadow_canary as canary  # noqa: E402
from core.brain.llm.unified_recurrent_shadow import (  # noqa: E402
    inspect_shadow_package,
)

RESULT_SCHEMA: Final = "aura.unified_intrinsic.live_shadow_canary_run.v2"


class UnifiedRecurrentLiveCanaryError(RuntimeError):
    """The live worker could not produce admissible shadow-canary evidence."""


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


def _absolute(path: Path) -> Path:
    return path.expanduser().absolute()


def _private_output(path: Path, payload: bytes) -> None:
    destination = path.expanduser().absolute()
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent = destination.parent
    current = Path(parent.anchor)
    for part in parent.parts[1:]:
        current /= part
        if current.is_symlink():
            raise UnifiedRecurrentLiveCanaryError(
                "live shadow canary output path contains a symlink"
            )
    metadata = parent.stat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise UnifiedRecurrentLiveCanaryError(
            "live shadow canary output custody differs"
        )
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
            raise UnifiedRecurrentLiveCanaryError(
                "live shadow canary evidence write was short"
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
        raise UnifiedRecurrentLiveCanaryError(
            "live shadow canary evidence publication failed"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if created and not completed:
            try:
                destination.unlink()
            except FileNotFoundError:
                pass


def _validated_result(
    result: Any,
    *,
    manifest: Mapping[str, Any],
    worker_status: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise UnifiedRecurrentLiveCanaryError(
            "live shadow canary result is unavailable"
        )
    plan = result.get("plan")
    verdict = result.get("verdict")
    if not isinstance(plan, Mapping) or not isinstance(verdict, Mapping):
        raise UnifiedRecurrentLiveCanaryError(
            "live shadow canary plan or verdict is unavailable"
        )
    try:
        canary._validate_plan(plan)  # noqa: SLF001 - independent evidence reopening
    except (TypeError, ValueError, canary.UnifiedRecurrentShadowCanaryError) as exc:
        raise UnifiedRecurrentLiveCanaryError(
            "live shadow canary plan is invalid"
        ) from exc
    verdict_body = {
        key: value for key, value in verdict.items() if key != "verdict_sha256"
    }
    checks = verdict.get("checks")
    evidence = verdict.get("evidence")
    recomputed_support = isinstance(checks, Mapping) and bool(checks) and all(
        value is True for value in checks.values()
    )
    package_id = manifest.get("package_id")
    controller_sha256 = worker_status.get("controller_sha256")
    if (
        worker_status.get("loaded") is not True
        or worker_status.get("serving_authority") is not False
        or worker_status.get("package_id") != package_id
        or worker_status.get("manifest_sha256") != manifest.get("manifest_sha256")
        or plan.get("package_id") != package_id
        or plan.get("controller_sha256") != controller_sha256
        or verdict.get("plan_sha256") != plan.get("plan_sha256")
        or verdict.get("schema") != canary.VERDICT_SCHEMA
        or verdict.get("package_id") != package_id
        or verdict.get("controller_sha256") != controller_sha256
        or verdict.get("verdict_sha256") != _canonical_sha256(verdict_body)
        or verdict.get("supported") is not recomputed_support
        or verdict.get("verdict")
        != (canary.SUPPORTED if recomputed_support else canary.REFUTED)
        or not isinstance(evidence, list)
        or any(
            not isinstance(row, Mapping)
            or any("text" in key or "token" in key for key in row)
            for row in evidence
        )
        or verdict.get("serving_authority") is not False
        or verdict.get("output_exposed") is not False
        or result.get("supported") is not bool(verdict.get("supported"))
    ):
        raise UnifiedRecurrentLiveCanaryError(
            "live shadow canary identity or authority differs"
        )
    return {
        "plan": dict(plan),
        "verdict": dict(verdict),
        "supported": result.get("supported") is True,
        "reason": str(result.get("reason") or ""),
    }


async def run_live_canary(
    package: Path,
    *,
    model_path: Path,
    output: Path,
    discovery_mode: str = "environment_override",
    minimum_wrong_to_right: int,
    maximum_shadow_latency_ms: int,
    maximum_latency_ratio_numerator: int,
    maximum_latency_ratio_denominator: int,
) -> dict[str, Any]:
    """Cold-load one package, run its hidden battery, and publish the verdict."""

    integer_thresholds = (
        minimum_wrong_to_right,
        maximum_shadow_latency_ms,
        maximum_latency_ratio_numerator,
        maximum_latency_ratio_denominator,
    )
    if (
        any(type(value) is not int for value in integer_thresholds)
        or minimum_wrong_to_right < 0
        or maximum_shadow_latency_ms < 1
        or maximum_latency_ratio_numerator < 1
        or maximum_latency_ratio_denominator < 1
    ):
        raise UnifiedRecurrentLiveCanaryError(
            "live shadow canary threshold is invalid"
        )
    if discovery_mode not in {"environment_override", "durable_pointer"}:
        raise UnifiedRecurrentLiveCanaryError(
            "live shadow canary discovery mode is invalid"
        )
    package = _absolute(package)
    model_path = _absolute(model_path)
    verified = await asyncio.to_thread(inspect_shadow_package, package)
    manifest = verified.get("manifest")
    if not isinstance(manifest, Mapping):
        raise UnifiedRecurrentLiveCanaryError(
            "live shadow package manifest is unavailable"
        )
    environment_key = "AURA_UNIFIED_RECURRENT_SHADOW_PACKAGE"
    previous_environment = os.environ.get(environment_key)
    if discovery_mode == "environment_override":
        os.environ[environment_key] = str(package)
    else:
        from core.brain.llm.unified_recurrent_shadow_pointer import (
            default_shadow_activation_paths,
            resolve_shadow_pointer,
        )

        pointer_path, releases_root = default_shadow_activation_paths()
        selected = await asyncio.to_thread(
            resolve_shadow_pointer,
            pointer_path,
            releases_root=releases_root,
        )
        if selected != package:
            raise UnifiedRecurrentLiveCanaryError(
                "live shadow canary durable pointer selected a different package"
            )
        os.environ.pop(environment_key, None)
    from core.brain.llm.mlx_client import get_mlx_client

    try:
        client = get_mlx_client(str(model_path))
        started_at = time.time()
        primary_error: BaseException | None = None
        try:
            ready = await client.warmup(
                foreground_request=True,
                skip_swap_cooldown=True,
            )
            if not ready:
                raise UnifiedRecurrentLiveCanaryError(
                    "resident worker did not become ready for live shadow canary"
                )
            worker_status = dict(
                getattr(client, "_unified_recurrent_shadow_status", {}) or {}
            )
            result = await client.unified_recurrent_shadow_package_canary_async(
                package,
                minimum_wrong_to_right=minimum_wrong_to_right,
                maximum_shadow_latency_ms=maximum_shadow_latency_ms,
                maximum_latency_ratio_numerator=maximum_latency_ratio_numerator,
                maximum_latency_ratio_denominator=maximum_latency_ratio_denominator,
            )
            accepted = _validated_result(
                result,
                manifest=manifest,
                worker_status=worker_status,
            )
        except BaseException as exc:  # noqa: BLE001 - preserve cancellation through cleanup
            primary_error = exc
            raise
        finally:
            try:
                await client.aclose()
            except BaseException as close_exc:  # noqa: BLE001 - do not hide evidence failure
                if primary_error is None:
                    raise
                primary_error.add_note(f"resident worker close also failed: {close_exc}")
    finally:
        if previous_environment is None:
            os.environ.pop(environment_key, None)
        else:
            os.environ[environment_key] = previous_environment
    body = {
        "schema": RESULT_SCHEMA,
        "package_id": manifest["package_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "model_path": str(model_path),
        "discovery_mode": discovery_mode,
        "started_at_unix": started_at,
        "completed_at_unix": time.time(),
        "worker_shadow_status": worker_status,
        **accepted,
        "serving_authority": False,
        "output_exposed": False,
    }
    document = {**body, "result_sha256": _canonical_sha256(body)}
    _private_output(output, _canonical_bytes(document))
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--discovery-mode",
        choices=("environment_override", "durable_pointer"),
        default="environment_override",
    )
    parser.add_argument("--minimum-wrong-to-right", type=int, default=1)
    parser.add_argument("--maximum-shadow-latency-ms", type=int, default=120_000)
    parser.add_argument("--maximum-latency-ratio-numerator", type=int, default=8)
    parser.add_argument("--maximum-latency-ratio-denominator", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = asyncio.run(
            run_live_canary(
                arguments.package,
                model_path=arguments.model,
                output=arguments.output,
                discovery_mode=arguments.discovery_mode,
                minimum_wrong_to_right=arguments.minimum_wrong_to_right,
                maximum_shadow_latency_ms=arguments.maximum_shadow_latency_ms,
                maximum_latency_ratio_numerator=(
                    arguments.maximum_latency_ratio_numerator
                ),
                maximum_latency_ratio_denominator=(
                    arguments.maximum_latency_ratio_denominator
                ),
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"live shadow canary failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["supported"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
