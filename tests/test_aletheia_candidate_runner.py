from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _runner_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "aletheia" / "run_candidate_battery.py"
    spec = importlib.util.spec_from_file_location("aletheia_candidate_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_optimal_schedule_exposes_evaluator_tasks_and_feasible_wall_clock_schedule():
    runner = _runner_module()
    tasks = [
        {"task": "A", "duration": "1", "prereqs": ""},
        {"task": "B", "duration": "5", "prereqs": "A"},
        {"task": "C", "duration": "4", "prereqs": "A"},
        {"task": "D", "duration": "1", "prereqs": "B"},
        {"task": "E", "duration": "2", "prereqs": "B;C"},
        {"task": "F", "duration": "5", "prereqs": "C"},
        {"task": "G", "duration": "4", "prereqs": "D;E;F"},
    ]

    schedule = runner.optimal_schedule(tasks)

    assert set(schedule) >= {"tasks", "wall_clock_tasks", "actual_makespan", "time_origin_offset"}
    assert max(entry["end"] for entry in schedule["tasks"]) == 11
    assert schedule["actual_makespan"] == 14
    by_task = {entry["task"]: entry for entry in schedule["tasks"]}
    for task in tasks:
        entry = by_task[task["task"]]
        assert entry["end"] - entry["start"] == int(task["duration"])


def test_report_solver_emits_exact_rounded_values_expected_by_grader(tmp_path):
    runner = _runner_module()
    world = tmp_path / "W0010_report_generation"
    (world / "data" / "raw").mkdir(parents=True)
    (world / "tickets").mkdir()
    (world / "data" / "raw" / "measurements.csv").write_text(
        "\n".join(
            [
                "id,signal,passed",
                "R0,27,false",
                "R1,7,false",
                "R2,22,true",
                "R3,bad,false",
                "R4,17,false",
                "R5,10,false",
                "R6,7,true",
                "R7,17,true",
                "R8,34,false",
                "R9,12,true",
                "R10,23,true",
                "R11,30,false",
                "",
            ]
        ),
        encoding="utf-8",
    )
    ctx = runner.BatteryContext(root=tmp_path, changed=set())

    runner.solve_report(ctx, world)

    report = (world / "reports" / "analysis.md").read_text(encoding="utf-8")
    assert "15.78" in report
    assert "45.45" in report
    assert "0.4545" in report


def test_candidate_runner_refuses_evaluator_material(tmp_path, monkeypatch):
    root = tmp_path / "candidate_battery_500"
    (root / "worlds").mkdir(parents=True)
    (root / "hidden_grader").mkdir()
    runner = _runner_module()

    monkeypatch.setattr(sys, "argv", ["run_candidate_battery.py", str(root)])

    with pytest.raises(SystemExit) as excinfo:
        runner.main()
    assert "forbidden evaluator/private paths" in str(excinfo.value)
