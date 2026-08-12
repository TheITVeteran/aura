from __future__ import annotations

import pytest

from core.brain.llm.unified_recurrent_shadow_battery import (
    UnifiedRecurrentShadowBatteryError,
    seal_shadow_canary_battery,
    shadow_canary_cases,
    validate_shadow_canary_battery,
)


def _battery():
    return seal_shadow_canary_battery(
        [
            {
                "task_id": "fresh-khop-1",
                "family": "khop",
                "task_depth": 2,
                "prompt_sha256": "a" * 64,
                "expected_sha256": "b" * 64,
                "public_token_ids": [1, 201, 2],
                "expected_token_ids": [12, 999],
                "max_tokens": 2,
            }
        ],
        seed=101,
        replication_plan_sha256="c" * 64,
        replication_verdict_sha256="d" * 64,
        excluded_task_ids_sha256="e" * 64,
        excluded_prompt_sha256s_sha256="f" * 64,
        generator_source_sha256s={"core/learning/example.py": "1" * 64},
    )


def test_battery_round_trip_preserves_private_cases_without_serving_authority() -> None:
    battery = _battery()

    assert validate_shadow_canary_battery(battery) == battery
    assert shadow_canary_cases(battery) == [
        {
            "task_id": "fresh-khop-1",
            "family": "khop",
            "public_token_ids": [1, 201, 2],
            "expected_token_ids": [12, 999],
            "max_tokens": 2,
        }
    ]
    assert battery["output_exposed"] is False
    assert battery["serving_authority"] is False


def test_battery_rejects_tampered_token_payload() -> None:
    battery = _battery()
    battery["cases"][0]["expected_token_ids"] = [13, 999]

    with pytest.raises(
        UnifiedRecurrentShadowBatteryError,
        match="commitment differs",
    ):
        validate_shadow_canary_battery(battery)


def test_battery_rejects_duplicate_requests() -> None:
    battery = _battery()
    case = dict(battery["cases"][0])
    case["task_id"] = "fresh-khop-2"
    case["prompt_sha256"] = "2" * 64

    with pytest.raises(
        UnifiedRecurrentShadowBatteryError,
        match="request is duplicated",
    ):
        seal_shadow_canary_battery(
            [battery["cases"][0], case],
            seed=101,
            replication_plan_sha256="c" * 64,
            replication_verdict_sha256="d" * 64,
            excluded_task_ids_sha256="e" * 64,
            excluded_prompt_sha256s_sha256="f" * 64,
            generator_source_sha256s={"core/learning/example.py": "1" * 64},
        )


def test_battery_rejects_unsafe_generator_source_path() -> None:
    battery = _battery()

    with pytest.raises(
        UnifiedRecurrentShadowBatteryError,
        match="identity invalid",
    ):
        seal_shadow_canary_battery(
            battery["cases"],
            seed=101,
            replication_plan_sha256="c" * 64,
            replication_verdict_sha256="d" * 64,
            excluded_task_ids_sha256="e" * 64,
            excluded_prompt_sha256s_sha256="f" * 64,
            generator_source_sha256s={"../outside.py": "1" * 64},
        )
