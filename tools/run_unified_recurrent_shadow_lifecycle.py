#!/usr/bin/env python3
"""Prove durable recurrent-shadow restart and rollback on Aura's real worker."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.brain.llm.unified_recurrent_shadow import inspect_shadow_package  # noqa: E402
from core.brain.llm.unified_recurrent_shadow_contract import (  # noqa: E402
    shadow_load_receipt_errors,
)
from core.brain.llm.unified_recurrent_shadow_pointer import (  # noqa: E402
    deactivate_shadow_pointer,
    default_shadow_activation_paths,
    publish_shadow_pointer,
    resolve_shadow_pointer,
)
from tools.run_unified_recurrent_shadow_live_canary import (  # noqa: E402
    _private_output,
    run_live_canary,
)

RESULT_SCHEMA: Final = "aura.unified_intrinsic.shadow_lifecycle_run.v1"
_ENVIRONMENT_KEY: Final = "AURA_UNIFIED_RECURRENT_SHADOW_PACKAGE"


class UnifiedRecurrentShadowLifecycleError(RuntimeError):
    """Durable activation, restart, or rollback evidence was not established."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _prepare_paths(
    package: Path,
    model_path: Path,
    output_directory: Path,
) -> tuple[Path, Path, Path]:
    package = package.expanduser().absolute()
    model_path = model_path.expanduser().absolute()
    output_directory = output_directory.expanduser().absolute()
    output_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if any(output_directory.iterdir()):
        raise UnifiedRecurrentShadowLifecycleError(
            "recurrent shadow lifecycle output directory is not empty"
        )
    return package, model_path, output_directory


async def _inactive_worker_receipt(model_path: Path) -> dict[str, Any]:
    """Cold-start after rollback and prove the package is no longer discovered."""

    from core.brain.llm.mlx_client import get_mlx_client

    previous = os.environ.pop(_ENVIRONMENT_KEY, None)
    primary_error: BaseException | None = None
    try:
        client = get_mlx_client(str(model_path))
        try:
            ready = await client.warmup(
                foreground_request=True,
                skip_swap_cooldown=True,
            )
            if not ready:
                raise UnifiedRecurrentShadowLifecycleError(
                    "post-rollback resident worker did not become ready"
                )
            receipt = dict(
                getattr(client, "_unified_recurrent_shadow_status", {}) or {}
            )
            errors = shadow_load_receipt_errors(receipt)
            if (
                errors
                or receipt.get("configured") is not False
                or receipt.get("loaded") is not False
                or receipt.get("reason") != "not_configured"
                or receipt.get("serving_authority") is not False
            ):
                raise UnifiedRecurrentShadowLifecycleError(
                    "post-rollback worker retained recurrent shadow state:"
                    + ",".join(errors or ["active_after_rollback"])
                )
            return receipt
        except BaseException as exc:  # noqa: BLE001 - preserve cleanup context
            primary_error = exc
            raise
        finally:
            try:
                await client.aclose()
            except BaseException as close_exc:  # noqa: BLE001
                if primary_error is None:
                    raise
                primary_error.add_note(
                    f"post-rollback resident worker close also failed: {close_exc}"
                )
    finally:
        if previous is not None:
            os.environ[_ENVIRONMENT_KEY] = previous


async def run_lifecycle(
    package: Path,
    *,
    model_path: Path,
    output_directory: Path,
    minimum_wrong_to_right: int,
    maximum_shadow_latency_ms: int,
    maximum_latency_ratio_numerator: int,
    maximum_latency_ratio_denominator: int,
) -> dict[str, Any]:
    """Activate, cold-load twice, rollback, and cold-load inactive state."""

    package, model_path, output_directory = await asyncio.to_thread(
        _prepare_paths,
        package,
        model_path,
        output_directory,
    )
    verified = await asyncio.to_thread(inspect_shadow_package, package)
    manifest = verified.get("manifest")
    if not isinstance(manifest, Mapping):
        raise UnifiedRecurrentShadowLifecycleError(
            "recurrent shadow lifecycle package manifest is unavailable"
        )
    pointer_path, releases_root = default_shadow_activation_paths()
    if await asyncio.to_thread(lambda: pointer_path.exists() or pointer_path.is_symlink()):
        raise UnifiedRecurrentShadowLifecycleError(
            "recurrent shadow lifecycle requires an inactive initial pointer"
        )

    pointer: dict[str, Any] | None = None
    completed = False
    failure: BaseException | None = None
    started_at = time.time()
    try:
        pointer = await asyncio.to_thread(
            publish_shadow_pointer,
            package,
            pointer_path=pointer_path,
            releases_root=releases_root,
        )
        if await asyncio.to_thread(
            resolve_shadow_pointer,
            pointer_path,
            releases_root=releases_root,
        ) != package:
            raise UnifiedRecurrentShadowLifecycleError(
                "durable recurrent shadow activation reopened another package"
            )
        canary_arguments = {
            "model_path": model_path,
            "discovery_mode": "durable_pointer",
            "minimum_wrong_to_right": minimum_wrong_to_right,
            "maximum_shadow_latency_ms": maximum_shadow_latency_ms,
            "maximum_latency_ratio_numerator": maximum_latency_ratio_numerator,
            "maximum_latency_ratio_denominator": maximum_latency_ratio_denominator,
        }
        print("recurrent shadow lifecycle: cold-load 1 started", flush=True)
        first = await run_live_canary(
            package,
            output=output_directory / "cold-load-01.json",
            **canary_arguments,
        )
        if first.get("supported") is not True:
            raise UnifiedRecurrentShadowLifecycleError(
                "recurrent shadow first cold-load evidence is refuted"
            )
        print("recurrent shadow lifecycle: cold-load 1 supported", flush=True)
        print("recurrent shadow lifecycle: cold-load 2 started", flush=True)
        second = await run_live_canary(
            package,
            output=output_directory / "cold-load-02.json",
            **canary_arguments,
        )
        if (
            first.get("supported") is not True
            or second.get("supported") is not True
            or first.get("manifest_sha256") != manifest.get("manifest_sha256")
            or second.get("manifest_sha256") != manifest.get("manifest_sha256")
            or first.get("plan", {}).get("plan_sha256")
            != second.get("plan", {}).get("plan_sha256")
            or first.get("worker_shadow_status", {}).get("controller_sha256")
            != second.get("worker_shadow_status", {}).get("controller_sha256")
        ):
            raise UnifiedRecurrentShadowLifecycleError(
                "recurrent shadow cold-load evidence differs or is refuted"
            )
        retired = await asyncio.to_thread(
            deactivate_shadow_pointer,
            pointer_path=pointer_path,
            releases_root=releases_root,
            expected_current_sha256=pointer["pointer_sha256"],
        )
        pointer = None
        inactive = await _inactive_worker_receipt(model_path)
        body = {
            "schema": RESULT_SCHEMA,
            "package_id": manifest["package_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "model_path": str(model_path),
            "started_at_unix": started_at,
            "completed_at_unix": time.time(),
            "activation_pointer": retired,
            "cold_load_result_sha256s": [
                first["result_sha256"],
                second["result_sha256"],
            ],
            "canary_plan_sha256": first["plan"]["plan_sha256"],
            "controller_sha256": first["worker_shadow_status"]["controller_sha256"],
            "post_rollback_load_receipt_sha256": inactive["receipt_sha256"],
            "checks": {
                "durable_pointer_reopened": True,
                "first_cold_load_supported": True,
                "restart_cold_load_supported": True,
                "restart_identity_stable": True,
                "pointer_rollback_completed": True,
                "post_rollback_worker_inactive": True,
            },
            "supported": True,
            "serving_authority": False,
            "output_exposed": False,
        }
        result = {**body, "result_sha256": _sha(body)}
        _private_output(
            output_directory / "lifecycle-result.json",
            _canonical_bytes(result),
        )
        completed = True
        return result
    except BaseException as exc:  # noqa: BLE001 - preserve lifecycle failure through rollback
        failure = exc
        raise
    finally:
        if pointer is not None and not completed:
            try:
                await asyncio.to_thread(
                    deactivate_shadow_pointer,
                    pointer_path=pointer_path,
                    releases_root=releases_root,
                    expected_current_sha256=pointer["pointer_sha256"],
                )
            except Exception as exc:  # noqa: BLE001 - retain original failure
                if failure is not None:
                    failure.add_note(
                        f"emergency recurrent shadow rollback also failed: {exc}"
                    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--minimum-wrong-to-right", type=int, default=1)
    parser.add_argument("--maximum-shadow-latency-ms", type=int, default=120_000)
    parser.add_argument("--maximum-latency-ratio-numerator", type=int, default=8)
    parser.add_argument("--maximum-latency-ratio-denominator", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = asyncio.run(
            run_lifecycle(
                arguments.package,
                model_path=arguments.model,
                output_directory=arguments.output_directory,
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
        print(f"recurrent shadow lifecycle failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
