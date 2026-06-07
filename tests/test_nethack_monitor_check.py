from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_monitor(monkeypatch, tmp_path: Path):
    trace = tmp_path / "kernel_trace.jsonl"
    monkeypatch.setenv("AURA_NETHACK_LOG", str(trace))
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "nethack_monitor_check.py"
    spec = importlib.util.spec_from_file_location("nethack_monitor_check_under_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, trace


def test_nethack_monitor_tolerates_missing_nested_trace_fields(monkeypatch, tmp_path):
    module, trace = _load_monitor(monkeypatch, tmp_path)
    trace.write_text(
        json.dumps(["not", "a", "mapping"])
        + "\n"
        + json.dumps(
            {
                "timestamp": "not-a-number",
                "sequence_id": 7,
                "action_intent": None,
                "outcome_assessment": None,
                "execution_result": {"observation_after": None},
                "context_id": "dlvl_1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    status = module.check_status()

    assert status["runner_crashed"] is False
    assert status["died"] is False
    assert status["latest_step"]["action"] is None
    assert status["latest_step"]["events"] == []
