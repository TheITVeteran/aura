from __future__ import annotations

import copy
from typing import Any

import pytest

from core.learning.resident_recurrent_sft_bootstrap_authority import sha256_json
from core.learning.resident_recurrent_sft_bootstrap_execution import (
    ResidentSFTBootstrapExecutionError,
    advance_sample_history,
    family_depth_balanced_order,
    initial_sample_history,
    project_example,
    project_rows,
    sampling_receipt,
    validate_family_depth_balanced_order,
)


class FakeTokenizer:
    eos_token_id = 3

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str:
        assert add_generation_prompt is True
        assert tokenize is False
        return f"<user>{messages[0]['content']}</user><assistant>"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) + 10 for character in text]

    def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        return "".join(chr(token - 10) for token in tokens)


def _row(
    task_id: str = "task.1",
    *,
    family: str = "logic",
    depth: int = 2,
    ordinal: int = 0,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "family": family,
        "depth": depth,
        "prompt": f"Solve {task_id}.",
        "answer": 'FINAL_ANSWER: {"value":1}',
        "ordinal": ordinal,
    }


def _projected_rows() -> list[dict[str, Any]]:
    rows = [
        _row("logic.1", family="logic", depth=1, ordinal=0),
        _row("logic.2", family="logic", depth=2, ordinal=1),
        _row("logic.3", family="logic", depth=1, ordinal=2),
        _row("code.1", family="code", depth=1, ordinal=3),
        _row("code.2", family="code", depth=3, ordinal=4),
        _row("math.1", family="math", depth=2, ordinal=5),
    ]
    return project_rows(rows, tokenizer=FakeTokenizer(), max_seq_length=512)


def test_projection_uses_live_chat_boundary_answer_only_and_eos() -> None:
    row = _row()

    projected = project_example(
        row,
        tokenizer=FakeTokenizer(),
        max_seq_length=512,
    )

    assert projected["task_id"] == "task.1"
    assert projected["answer_tokens"][-1] == FakeTokenizer.eos_token_id
    assert projected["bridge_tokens"] == []
    rendered = "".join(chr(token - 10) for token in projected["prompt_tokens"])
    assert row["prompt"] in rendered
    assert row["answer"] not in rendered
    assert projected["example_id"] == sha256_json(
        {
            key: value
            for key, value in projected.items()
            if key not in {"example_id", "prompt_tokens", "answer_tokens"}
        }
    )


def test_projection_rejects_answer_round_trip_and_sequence_overflow() -> None:
    class BrokenDecode(FakeTokenizer):
        def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
            return "different"

    with pytest.raises(ResidentSFTBootstrapExecutionError, match="round_trip"):
        project_example(_row(), tokenizer=BrokenDecode(), max_seq_length=512)

    with pytest.raises(ResidentSFTBootstrapExecutionError, match="sequence_budget"):
        project_example(_row(), tokenizer=FakeTokenizer(), max_seq_length=32)


def test_projection_rejects_answer_leak_from_template() -> None:
    class LeakingTokenizer(FakeTokenizer):
        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            add_generation_prompt: bool,
            tokenize: bool,
        ) -> str:
            return super().apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
            ) + 'FINAL_ANSWER: {"value":1}'

    with pytest.raises(ResidentSFTBootstrapExecutionError, match="prompt_render"):
        project_example(_row(), tokenizer=LeakingTokenizer(), max_seq_length=512)


def test_projected_identity_is_input_sensitive_and_unique() -> None:
    first = project_example(_row(), tokenizer=FakeTokenizer(), max_seq_length=512)
    changed_row = _row()
    changed_row["prompt"] = "A different task."
    changed = project_example(changed_row, tokenizer=FakeTokenizer(), max_seq_length=512)
    assert first["example_id"] != changed["example_id"]

    with pytest.raises(ResidentSFTBootstrapExecutionError, match="identity_duplicate"):
        project_rows(
            [_row(), copy.deepcopy(_row())],
            tokenizer=FakeTokenizer(),
            max_seq_length=512,
        )


def test_family_depth_schedule_is_deterministic_and_without_replacement() -> None:
    rows = _projected_rows()
    first = family_depth_balanced_order(rows, seed=31, epoch=0)
    replay = family_depth_balanced_order(rows, seed=31, epoch=0)

    assert first == replay
    assert sorted(first) == list(range(len(rows)))
    assert len(first) == len(set(first))
    validate_family_depth_balanced_order(rows, first, seed=31, epoch=0)


def test_family_depth_schedule_changes_with_seed_or_epoch() -> None:
    rows = _projected_rows()
    baseline = family_depth_balanced_order(rows, seed=31, epoch=0)
    alternatives = {
        tuple(family_depth_balanced_order(rows, seed=seed, epoch=epoch))
        for seed, epoch in ((32, 0), (31, 1), (33, 2), (34, 3))
    }
    assert len({tuple(baseline), *alternatives}) > 1


def test_family_depth_schedule_rejects_order_drift() -> None:
    rows = _projected_rows()
    order = family_depth_balanced_order(rows, seed=31, epoch=0)
    order[0], order[1] = order[1], order[0]

    with pytest.raises(ResidentSFTBootstrapExecutionError, match="order_drift"):
        validate_family_depth_balanced_order(rows, order, seed=31, epoch=0)


def test_sampling_receipt_binds_strata_and_order() -> None:
    rows = _projected_rows()
    order = family_depth_balanced_order(rows, seed=31, epoch=0)

    receipt = sampling_receipt(rows, order, seed=31, epoch=0)

    assert receipt["all_rows_once"] is True
    assert receipt["family_counts"] == {"code": 2, "logic": 3, "math": 1}
    assert receipt["family_depth_counts"] == {
        "code:1": 1,
        "code:3": 1,
        "logic:1": 2,
        "logic:2": 1,
        "math:2": 1,
    }


def test_sample_history_is_chained_and_position_sensitive() -> None:
    rows = _projected_rows()
    start = initial_sample_history()
    first = advance_sample_history(
        start,
        example_id=rows[0]["example_id"],
        step=1,
        epoch=0,
        cursor=1,
    )
    second = advance_sample_history(
        first,
        example_id=rows[1]["example_id"],
        step=2,
        epoch=0,
        cursor=2,
    )

    assert first != start
    assert second != first
    assert second != advance_sample_history(
        first,
        example_id=rows[1]["example_id"],
        step=2,
        epoch=0,
        cursor=3,
    )
