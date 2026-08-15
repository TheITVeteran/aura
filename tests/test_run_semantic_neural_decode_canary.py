from __future__ import annotations

from tools.run_semantic_neural_decode_canary import _arm_order, _lane_kwargs, _summary


def test_semantic_neural_decode_arm_order_is_complete_and_deterministic():
    first = _arm_order("task-a")
    assert first == _arm_order("task-a")
    assert set(first) == {
        "ordinary_base",
        "matched_wire_base",
        "treatment",
        "coefficient_lesion",
        "matched_wrong_state",
    }


def test_semantic_neural_decode_summary_counts_exact_results():
    rows = [
        {
            "arm": "treatment",
            "correct": True,
            "parsed": True,
            "prompt_tokens": 10,
            "generated_tokens": 4,
            "latency_ms": 20,
        },
        {
            "arm": "treatment",
            "correct": False,
            "parsed": True,
            "prompt_tokens": 12,
            "generated_tokens": 6,
            "latency_ms": 30,
        },
    ]
    summary = _summary(rows, "treatment")
    assert summary["examples"] == 2
    assert summary["exact_accuracy"] == 0.5
    assert summary["parsed_accuracy"] == 1.0
    assert summary["mean_latency_ms"] == 25.0


def test_semantic_neural_decode_uses_current_nonpreemptible_lane_contract(tmp_path):
    values = _lane_kwargs(tmp_path / "model", tmp_path / "result.json")
    assert values["model_path"] == str(tmp_path / "model")
    assert values["purpose"] == "evaluation"
    assert values["preemptible"] is False
    assert values["allow_owner_eviction"] is False
    assert "require_live_runtime_absent" not in values
