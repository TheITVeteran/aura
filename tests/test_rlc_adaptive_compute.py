"""SPARK-052 adaptive compute routing contracts."""

from __future__ import annotations

from copy import deepcopy

import pytest

from core.brain.llm.latent_cortex.adaptive_compute import (
    apply_adaptive_compute_plan,
    build_adaptive_acquisition_receipt,
    build_adaptive_compute_plan,
    build_adaptive_execution_receipt,
    estimate_objective_difficulty,
    resource_pressure_signal,
    validate_adaptive_acquisition_receipt,
    validate_adaptive_compute_plan,
    validate_adaptive_execution_receipt,
)


def _snapshot(*, memory: float = 40.0, lag: float = 0.02) -> dict:
    return {
        "observation_source": "test_probe",
        "resource_observation_available": True,
        "memory_percent": memory,
        "thermal_level": 0,
        "loop_lag_s": lag,
        "red_zones": [],
        "suspended_capabilities": [],
    }


def _plan(**overrides):
    kwargs = {
        "objective": "Compare both designs, choose one, and verify it under restart faults.",
        "stakes": 0.8,
        "uncertainty": 0.8,
        "body_pressure": 0.1,
        "deadline_s": 240.0,
        "resource_snapshot": _snapshot(),
        "foreground_request": True,
        "model_parameter_count": 32_000_000_000,
        "requested_decode_tokens": 256,
    }
    kwargs.update(overrides)
    return build_adaptive_compute_plan(**kwargs)


def test_objective_difficulty_is_structural_and_bounded():
    simple = estimate_objective_difficulty("What time is it?")
    compound = estimate_objective_difficulty(
        "Compare A and B, choose the safer design, then explain and verify the choice."
    )

    assert 0.0 <= simple["difficulty"] < compound["difficulty"] <= 1.0
    assert compound["prompt_shape"]["question_parts"] >= 2


def test_unobserved_resource_pressure_is_conservative():
    unknown = resource_pressure_signal(None)
    observed = resource_pressure_signal(_snapshot())

    assert unknown["observed"] is False
    assert unknown["pressure"] >= 0.6
    assert observed["observed"] is True
    assert observed["pressure"] < unknown["pressure"]


def test_demand_increases_compute_inside_equal_capacity():
    easy = _plan(objective="What time is it?", stakes=0.1, uncertainty=0.1)
    hard = _plan(stakes=0.95, uncertainty=0.95)

    assert hard["economy"]["level"] > easy["economy"]["level"]
    assert hard["routing"]["recurrence"]["max_steps"] >= easy["routing"][
        "recurrence"
    ]["max_steps"]
    assert hard["routing"]["branches"]["target"] >= easy["routing"]["branches"][
        "target"
    ]
    assert hard["routing"]["lookahead"]["max_nodes"] >= easy["routing"][
        "lookahead"
    ]["max_nodes"]


def test_pressure_and_deadline_ration_compute_and_tools():
    calm = _plan()
    constrained = _plan(
        body_pressure=0.95,
        deadline_s=35.0,
        resource_snapshot=_snapshot(memory=96.0, lag=3.0),
    )

    assert constrained["economy"]["capacity"] < calm["economy"]["capacity"]
    assert constrained["routing"]["recurrence"]["max_steps"] <= calm["routing"][
        "recurrence"
    ]["max_steps"]
    assert constrained["routing"]["branches"]["target"] == 1
    assert constrained["routing"]["tools"]["max_acquisitions"] == 0
    assert calm["routing"]["tools"]["max_acquisitions"] == 1


def test_apply_preserves_answer_surface_and_binds_all_mechanisms():
    plan = _plan()
    config, budget = apply_adaptive_compute_plan(
        {"decode_max_tokens": 256},
        {"wall_clock_s": 120.0, "max_layer_apps": 1_000_000},
        plan,
    )

    assert config["decode_max_tokens"] == 256
    assert config["max_steps"] == plan["routing"]["recurrence"]["max_steps"]
    assert config["n_branches"] == plan["routing"]["branches"]["target"]
    assert config["latent_tree_search"] == plan["routing"]["lookahead"]
    assert config["verifier_probe_max_tokens"] == plan["routing"]["verifier"][
        "probe_max_tokens"
    ]
    assert budget["wall_clock_s"] == 120.0


def test_plan_and_execution_receipts_reject_tampering():
    plan = _plan()
    tampered = deepcopy(plan)
    tampered["routing"]["recurrence"]["max_steps"] += 1
    with pytest.raises(ValueError, match="commitment"):
        validate_adaptive_compute_plan(tampered)

    config, budget = apply_adaptive_compute_plan(
        {"decode_max_tokens": 256},
        {"wall_clock_s": 120.0},
        plan,
    )
    receipt = build_adaptive_execution_receipt(
        plan=plan,
        config=config,
        budget=budget,
        worker_receipt={
            "episode_id": "episode-1",
            "steps_taken": config["max_steps"],
            "n_branches": config["n_branches"],
            "verifier_probe_max_tokens": config["verifier_probe_max_tokens"],
        },
    )
    assert receipt["within_plan"] is True
    assert receipt["plan"] == plan
    assert validate_adaptive_execution_receipt(receipt) == receipt
    forged = deepcopy(receipt)
    forged["actual"]["steps_taken"] += 1
    with pytest.raises(ValueError):
        validate_adaptive_execution_receipt(forged)

    with pytest.raises(ValueError, match="differs"):
        build_adaptive_execution_receipt(
            plan=plan,
            config=config,
            budget=budget,
            worker_receipt={
                "episode_id": "episode-1",
                "steps_taken": config["max_steps"] + 1,
                "n_branches": config["n_branches"],
                "verifier_probe_max_tokens": config["verifier_probe_max_tokens"],
            },
        )


def test_acquisition_receipt_enforces_zero_or_one_budget():
    denied_plan = _plan(deadline_s=35.0)
    denied = build_adaptive_acquisition_receipt(
        plan=denied_plan,
        request_sha256="a" * 64,
        attempted=False,
    )
    assert denied["authorized"] is False
    assert validate_adaptive_acquisition_receipt(denied) == denied
    with pytest.raises(ValueError, match="exceeded"):
        build_adaptive_acquisition_receipt(
            plan=denied_plan,
            request_sha256="a" * 64,
            attempted=True,
        )

    admitted = build_adaptive_acquisition_receipt(
        plan=_plan(),
        request_sha256="b" * 64,
        attempted=True,
    )
    assert admitted["authorized"] is True
    assert admitted["max_continuation_rounds"] == 1
    assert validate_adaptive_acquisition_receipt(admitted) == admitted
