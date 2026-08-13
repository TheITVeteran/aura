from __future__ import annotations

from types import SimpleNamespace

from tools.closeout.prove_clean_live_skill_boot import (
    evaluate_skill_surfaces,
    is_competing_model_owner,
)


def _health() -> dict[str, object]:
    return {
        "backend": "python_ast",
        "digest": "a" * 64,
        "expected_live_count": 2,
        "live_count": 2,
        "missing_live": [],
        "parity_status": "ready",
        "quarantined": [],
        "quarantined_count": 0,
        "ready": True,
        "reason": "ready",
        "execution_preflight": {
            "complete": True,
            "failed": [],
            "ok": True,
        },
    }


def _surfaces() -> tuple[dict, dict, dict]:
    catalog = [{"name": "clock"}, {"name": "web_search"}]
    health = _health()
    return (
        {"tools": catalog, "count": 2, "health": health},
        {"catalog": catalog, "count": 2, "health": health},
        {
            "tools": catalog,
            "skill_catalog": health,
            "ui": {"status_flags": []},
        },
    )


def test_external_surface_evaluator_requires_one_ready_identical_catalog() -> None:
    result = evaluate_skill_surfaces(*_surfaces())

    assert result["passed"] is True
    assert result["catalog_count"] == 2
    assert all(result["checks"].values())


def test_external_surface_evaluator_rejects_hidden_quarantine_and_ui_blocker() -> None:
    tools, skills, bootstrap = _surfaces()
    bootstrap["skill_catalog"] = {
        **bootstrap["skill_catalog"],
        "quarantined": [{"name": "web_search"}],
        "quarantined_count": 1,
        "ready": False,
    }
    bootstrap["ui"]["status_flags"] = ["skill_quarantined"]

    result = evaluate_skill_surfaces(tools, skills, bootstrap)

    assert result["passed"] is False
    assert result["checks"]["health_identical"] is False
    assert result["checks"]["ui_has_no_skill_blocker"] is False


def test_external_surface_evaluator_rejects_catalog_divergence() -> None:
    tools, skills, bootstrap = _surfaces()
    bootstrap["tools"] = [{"name": "clock"}]

    result = evaluate_skill_surfaces(tools, skills, bootstrap)

    assert result["passed"] is False
    assert result["checks"]["catalogs_identical"] is False


def test_model_owner_classifier_matches_argv_not_unrelated_python() -> None:
    evaluator = SimpleNamespace(
        pid=1,
        create_time=1.0,
        cmdline=("python", "/repo/tools/evaluate_unified_intrinsic_decoding.py"),
    )
    audit = SimpleNamespace(
        pid=2,
        create_time=1.0,
        cmdline=("python", "/repo/tools/closeout/audit_skill_catalog.py"),
    )

    assert is_competing_model_owner(evaluator) is True
    assert is_competing_model_owner(audit) is False
