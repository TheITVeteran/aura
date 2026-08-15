from __future__ import annotations

import json

from core.brain.llm.latent_cortex.semantic_neural_decode_context import (
    execute_semantic_neural_decode_state,
    render_semantic_neural_answer,
)
from core.learning.frontier_process_supervision import frontier_process_task_battery
from tools.run_semantic_neural_decode_canary import (
    _append_journal_event,
    _arm_order,
    _grade,
    _lane_kwargs,
    _resident_manifest_identity,
    _state_prefill,
    _summary,
    _wire_prefill,
)


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


def test_semantic_neural_decode_resident_manifest_selects_measured_model(tmp_path):
    model = tmp_path / "resident"
    model.mkdir()
    manifest = tmp_path / "active.json"
    manifest.write_text(
        json.dumps(
            {
                "active_model_path": str(model),
                "base_model": "base",
                "fused_at": 1,
                "schema_version": 2,
                "tag": "test",
            }
        ),
        encoding="utf-8",
    )
    identity = _resident_manifest_identity(manifest, model.resolve())
    assert identity["active_model_path"] == str(model.resolve())
    assert identity["schema_version"] == 2


def test_semantic_neural_decode_uses_training_task_grade_contract():
    class Task:
        @staticmethod
        def grade(response):
            return {"correct": response == "right", "parsed": {"value": response}}

    assert _grade(Task(), "right") == (True, True)
    assert _grade(Task(), "wrong") == (False, True)


def test_semantic_neural_decode_prefill_contains_only_public_syntax():
    class Tokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            assert add_special_tokens is False
            return list(text.encode("ascii"))

    tokenizer = Tokenizer()
    for family, expected in {
        "frontier_coding": 'FINAL_ANSWER: {"returns":',
        "frontier_calibration": 'FINAL_ANSWER: {"choice":',
        "frontier_misleading_premise": 'FINAL_ANSWER: {"actual_score":',
    }.items():
        rendered = bytes(_wire_prefill(tokenizer, family)).decode("ascii")
        assert rendered == expected
        assert not any(character.isdigit() for character in rendered)


def test_semantic_decode_journal_is_fsynced_and_receipt_chained(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.touch()
    first = _append_journal_event(
        path,
        {"event": "first", "value": 1},
        previous_receipt_sha256="0" * 64,
    )
    second = _append_journal_event(
        path,
        {"event": "second", "value": 2},
        previous_receipt_sha256=first,
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["receipt_sha256"] == first
    assert rows[1]["previous_receipt_sha256"] == first
    assert rows[1]["receipt_sha256"] == second


def test_semantic_state_prefill_is_state_grounded_and_leaves_model_suffix():
    class Tokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            assert add_special_tokens is False
            return list(text.encode("ascii"))

    task = frontier_process_task_battery(("coding",), (1,), 1, seed=1552)[0]
    state = execute_semantic_neural_decode_state(task.prompt, task.family)
    full = render_semantic_neural_answer(state).encode("ascii")
    prefix = bytes(_state_prefill(Tokenizer(), state, limit=64))
    assert full.startswith(prefix)
    assert len(prefix) == 64
    assert len(prefix) < len(full)
