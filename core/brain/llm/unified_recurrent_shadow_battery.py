"""Private, committed task battery for a unified recurrent shadow package."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any, Final

from core.brain.llm.unified_recurrent_shadow_probe_contract import (
    seal_shadow_probe_request,
)

BATTERY_SCHEMA: Final = "aura.unified_intrinsic.shadow_canary_battery.v1"
MAX_BATTERY_CASES: Final = 128
_HEX = frozenset("0123456789abcdef")


class UnifiedRecurrentShadowBatteryError(ValueError):
    """The private package canary battery is malformed or inconsistent."""


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def seal_shadow_canary_battery(
    cases: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replication_plan_sha256: str,
    replication_verdict_sha256: str,
    excluded_task_ids_sha256: str,
    excluded_prompt_sha256s_sha256: str,
    generator_source_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    """Seal private token payloads and the evidence they were disjoint from."""

    if (
        not isinstance(cases, (list, tuple))
        or not 1 <= len(cases) <= MAX_BATTERY_CASES
        or type(seed) is not int
        or seed < 0
        or not all(
            _is_sha256(value)
            for value in (
                replication_plan_sha256,
                replication_verdict_sha256,
                excluded_task_ids_sha256,
                excluded_prompt_sha256s_sha256,
            )
        )
        or not isinstance(generator_source_sha256s, Mapping)
        or not generator_source_sha256s
        or any(
            not isinstance(path, str)
            or not path
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            or str(PurePosixPath(path)) != path
            or not _is_sha256(digest)
            for path, digest in generator_source_sha256s.items()
        )
    ):
        raise UnifiedRecurrentShadowBatteryError(
            "shadow canary battery identity invalid"
        )
    normalized: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    prompt_sha256s: set[str] = set()
    request_sha256s: set[str] = set()
    for raw in cases:
        if not isinstance(raw, Mapping):
            raise UnifiedRecurrentShadowBatteryError(
                "shadow canary battery case invalid"
            )
        task_id = raw.get("task_id")
        family = raw.get("family")
        task_depth = raw.get("task_depth")
        prompt_sha256 = raw.get("prompt_sha256")
        expected_sha256 = raw.get("expected_sha256")
        if (
            not isinstance(task_id, str)
            or not task_id
            or len(task_id) > 256
            or task_id in task_ids
            or not isinstance(family, str)
            or not family
            or len(family) > 128
            or type(task_depth) is not int
            or task_depth < 1
            or not _is_sha256(prompt_sha256)
            or prompt_sha256 in prompt_sha256s
            or not _is_sha256(expected_sha256)
        ):
            raise UnifiedRecurrentShadowBatteryError(
                "shadow canary battery case identity invalid"
            )
        try:
            request = seal_shadow_probe_request(
                raw.get("public_token_ids"),
                raw.get("expected_token_ids"),
                max_tokens=raw.get("max_tokens"),
            )
        except (TypeError, ValueError) as exc:
            raise UnifiedRecurrentShadowBatteryError(
                f"shadow canary battery request invalid: {exc}"
            ) from exc
        if request["request_sha256"] in request_sha256s:
            raise UnifiedRecurrentShadowBatteryError(
                "shadow canary battery request is duplicated"
            )
        task_ids.add(task_id)
        prompt_sha256s.add(prompt_sha256)
        request_sha256s.add(request["request_sha256"])
        normalized.append(
            {
                "task_id": task_id,
                "family": family,
                "task_depth": task_depth,
                "prompt_sha256": prompt_sha256,
                "expected_sha256": expected_sha256,
                **request,
            }
        )
    body = {
        "schema": BATTERY_SCHEMA,
        "seed": seed,
        "replication_plan_sha256": replication_plan_sha256,
        "replication_verdict_sha256": replication_verdict_sha256,
        "excluded_task_ids_sha256": excluded_task_ids_sha256,
        "excluded_prompt_sha256s_sha256": excluded_prompt_sha256s_sha256,
        "generator_source_sha256s": dict(sorted(generator_source_sha256s.items())),
        "cases": normalized,
        "task_count": len(normalized),
        "task_ids_sha256": _sha(sorted(task_ids)),
        "prompt_sha256s_sha256": _sha(sorted(prompt_sha256s)),
        "request_sha256s_sha256": _sha(sorted(request_sha256s)),
        "output_exposed": False,
        "serving_authority": False,
    }
    return {**body, "battery_sha256": _sha(body)}


def validate_shadow_canary_battery(value: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct and verify a private battery without accepting extra fields."""

    if not isinstance(value, Mapping):
        raise UnifiedRecurrentShadowBatteryError(
            "shadow canary battery unavailable"
        )
    expected_keys = {
        "schema",
        "seed",
        "replication_plan_sha256",
        "replication_verdict_sha256",
        "excluded_task_ids_sha256",
        "excluded_prompt_sha256s_sha256",
        "generator_source_sha256s",
        "cases",
        "task_count",
        "task_ids_sha256",
        "prompt_sha256s_sha256",
        "request_sha256s_sha256",
        "output_exposed",
        "serving_authority",
        "battery_sha256",
    }
    cases = value.get("cases")
    if (
        set(value) != expected_keys
        or value.get("schema") != BATTERY_SCHEMA
        or not isinstance(cases, list)
        or value.get("task_count") != len(cases)
        or value.get("output_exposed") is not False
        or value.get("serving_authority") is not False
    ):
        raise UnifiedRecurrentShadowBatteryError(
            "shadow canary battery contract invalid"
        )
    raw_cases = [
        {
            key: row.get(key)
            for key in (
                "task_id",
                "family",
                "task_depth",
                "prompt_sha256",
                "expected_sha256",
                "public_token_ids",
                "expected_token_ids",
                "max_tokens",
            )
        }
        for row in cases
        if isinstance(row, Mapping)
    ]
    if len(raw_cases) != len(cases):
        raise UnifiedRecurrentShadowBatteryError(
            "shadow canary battery case invalid"
        )
    rebuilt = seal_shadow_canary_battery(
        raw_cases,
        seed=value.get("seed"),
        replication_plan_sha256=value.get("replication_plan_sha256"),
        replication_verdict_sha256=value.get("replication_verdict_sha256"),
        excluded_task_ids_sha256=value.get("excluded_task_ids_sha256"),
        excluded_prompt_sha256s_sha256=value.get(
            "excluded_prompt_sha256s_sha256"
        ),
        generator_source_sha256s=value.get("generator_source_sha256s"),
    )
    if dict(value) != rebuilt:
        raise UnifiedRecurrentShadowBatteryError(
            "shadow canary battery commitment differs"
        )
    return rebuilt


def shadow_canary_cases(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the exact private cases accepted by the live canary runner."""

    battery = validate_shadow_canary_battery(value)
    return [
        {
            "task_id": row["task_id"],
            "family": row["family"],
            "public_token_ids": list(row["public_token_ids"]),
            "expected_token_ids": list(row["expected_token_ids"]),
            "max_tokens": row["max_tokens"],
        }
        for row in battery["cases"]
    ]


__all__ = [
    "BATTERY_SCHEMA",
    "MAX_BATTERY_CASES",
    "UnifiedRecurrentShadowBatteryError",
    "seal_shadow_canary_battery",
    "shadow_canary_cases",
    "validate_shadow_canary_battery",
]
