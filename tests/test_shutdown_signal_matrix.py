from __future__ import annotations

from typing import Any

from tools.shutdown_signal_matrix import (
    CASE_SPECS,
    SHUTDOWN_PHASES,
    _is_aura_main_process,
    _parse_cases,
    evaluate_terminal_report,
)


def _clean_terminal_report() -> dict[str, Any]:
    return {
        "schema": "aura.shutdown_verdict.v1",
        "pid": 4242,
        "stage": "root_process_exit",
        "final": True,
        "terminal_receipt_sequence": 1,
        "request": {
            "first_reason": "desktop_signal:SIGTERM",
            "request_count": 4,
        },
        "admission": {"counts": {"survived": 0}},
        "components": {
            "coordinator": {
                "clean": True,
                "completed_phases": list(SHUTDOWN_PHASES),
            },
            "container": {"clean": True},
            "runtime_hygiene": {"clean": True},
            "final_tasks": {"count": 0},
        },
        "root_exit": {
            "lock_released": True,
            "multiprocessing_finalizers_completed": True,
            "logging_shutdown_completed": True,
            "resources": {"clean": True, "blockers": []},
            "exit_code": 0,
        },
        "verdict": {"clean": True, "blockers": []},
    }


def test_terminal_report_requires_every_shutdown_and_root_exit_invariant() -> None:
    checks = evaluate_terminal_report(
        _clean_terminal_report(),
        root_pid=4242,
        expected_first_reason="desktop_signal:SIGTERM",
        minimum_signal_requests=2,
    )

    assert checks
    assert all(checks.values())


def test_terminal_report_fails_on_resurrection_and_out_of_order_phases() -> None:
    report = _clean_terminal_report()
    report["admission"]["counts"]["survived"] = 1
    report["components"]["coordinator"]["completed_phases"] = list(
        reversed(SHUTDOWN_PHASES)
    )

    checks = evaluate_terminal_report(
        report,
        root_pid=4242,
        expected_first_reason="desktop_signal:SIGTERM",
        minimum_signal_requests=2,
    )

    assert checks["no_shutdown_resurrection"] is False
    assert checks["all_phases_once"] is False


def test_aura_process_classifier_matches_argv_not_shell_text() -> None:
    assert _is_aura_main_process(
        ["/opt/homebrew/bin/python3.12", "/repo/aura_main.py", "--desktop"]
    )
    assert not _is_aura_main_process(
        ["/bin/zsh", "-c", "ps aux | grep aura_main.py"]
    )
    assert not _is_aura_main_process(
        ["/opt/homebrew/bin/python3.12", "/repo/tools/shutdown_signal_matrix.py"]
    )


def test_case_selection_is_deduplicated_and_unknown_names_fail() -> None:
    assert _parse_cases(["ready_repeated,launcher_bootstrap", "ready_repeated"]) == [
        "ready_repeated",
        "launcher_bootstrap",
    ]
    assert _parse_cases(["all"]) == list(CASE_SPECS)

    try:
        _parse_cases(["not-a-case"])
    except ValueError as exc:
        assert "unknown case" in str(exc)
    else:
        raise AssertionError("unknown shutdown matrix case was accepted")


def test_matrix_covers_boot_ready_active_and_late_finalization_boundaries() -> None:
    assert {
        "launcher_bootstrap",
        "orchestrator_boot_repeated",
        "ready_repeated",
        "state_vault_repeated",
        "model_runtime_repeated",
        "container_repeated",
        "root_finalization_repeated",
        "active_foreground_repeated",
    } == set(CASE_SPECS)

    assert CASE_SPECS["state_vault_repeated"].probe_target == (
        "coordinator:state_vault"
    )
    assert CASE_SPECS["model_runtime_repeated"].probe_target == (
        "coordinator:model_runtime"
    )
    assert CASE_SPECS["container_repeated"].probe_target == "container"
    assert CASE_SPECS["root_finalization_repeated"].probe_target == (
        "root_finalization"
    )
