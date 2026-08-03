from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import pytest

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.learning.recurrence_native_objective_v5 import (
    GeneratedRollinSelectionConfig,
)
from core.learning.resident_recurrent_sft_bootstrap_authority import (
    OBJECTIVE_NAME_V2,
    TRAINER_CONFIG_SCHEMA_V2,
    ResidentSFTBootstrapConfig,
    sha256_json,
)
from core.learning.resident_recurrent_sft_bootstrap_execution import (
    ResidentSFTBootstrapExecutionError,
    adapter_topology_sha256,
    advance_sample_history,
    execution_spec_for_projected_row,
    family_depth_balanced_order,
    initial_sample_history,
    project_example,
    project_rows,
    sampling_receipt,
    validate_family_depth_balanced_order,
)
from tools import train_resident_recurrent_sft_bootstrap as trainer


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


@pytest.mark.parametrize("depth", [2, 4, 8])
def test_projected_depth_selects_actual_recurrent_graph(depth: int) -> None:
    projected = project_example(
        _row(depth=depth),
        tokenizer=FakeTokenizer(),
        max_seq_length=512,
    )
    base = RLCExecutionSpec(recurrent_steps=4)

    executed = execution_spec_for_projected_row(projected, base_spec=base)

    assert executed.recurrent_steps == depth
    assert executed.sha256 == base.with_depth(depth).sha256
    assert {
        key: value
        for key, value in executed.to_dict().items()
        if key != "recurrent_steps"
    } == {
        key: value
        for key, value in base.to_dict().items()
        if key != "recurrent_steps"
    }


def test_projected_depth_binding_rejects_missing_or_invalid_depth() -> None:
    base = RLCExecutionSpec(recurrent_steps=4)
    with pytest.raises(ResidentSFTBootstrapExecutionError, match="projected_depth_invalid"):
        execution_spec_for_projected_row({}, base_spec=base)
    with pytest.raises(ResidentSFTBootstrapExecutionError, match="projected_depth_invalid"):
        execution_spec_for_projected_row({"depth": True}, base_spec=base)


def test_validation_executes_and_receipts_each_projected_depth(monkeypatch: Any) -> None:
    rows = _projected_rows()[:3]
    for index, depth in enumerate((2, 4, 8)):
        rows[index]["depth"] = depth
    observed: list[int] = []

    def fake_loss(
        _model: Any,
        prompt_tokens: list[int],
        answer_tokens: list[int],
        *,
        spec: RLCExecutionSpec,
        bridge_tokens: list[int],
        branch_indices: tuple[int, ...],
    ) -> SimpleNamespace:
        assert bridge_tokens == []
        assert branch_indices == (0, 1)
        observed.append(spec.recurrent_steps)
        return SimpleNamespace(
            execution_spec_sha256=spec.sha256,
            prompt_tokens_sha256=sha256_json(prompt_tokens),
            answer_tokens_sha256=sha256_json(answer_tokens),
            value=float(spec.recurrent_steps),
            branch_values=(float(spec.recurrent_steps),) * 2,
            answer_token_count=len(answer_tokens),
        )

    monkeypatch.setattr(trainer, "cached_supervised_live_path_loss", fake_loss)
    summary = trainer._validation_summary(
        object(),
        rows,
        spec=RLCExecutionSpec(recurrent_steps=4),
        config=SimpleNamespace(seed=17, validation_examples=3, branch_indices=(0, 1)),
    )

    assert sorted(observed) == [2, 4, 8]
    assert summary["executed_depths"] == [2, 4, 8]
    assert len(summary["row_execution_spec_sha256s"]) == 3
    assert {
        record["requested_recurrent_depth"] for record in summary["records"]
    } == {2, 4, 8}
    assert all(
        record["requested_recurrent_depth"] == record["executed_recurrent_depth"]
        for record in summary["records"]
    )


def test_validation_uses_bound_generated_rollin_objective(monkeypatch: Any) -> None:
    row = _projected_rows()[0]
    observed: list[tuple[int, int]] = []

    def fake_generated_loss(
        _model: Any,
        prompt_tokens: list[int],
        answer_tokens: list[int],
        *,
        spec: RLCExecutionSpec,
        base_seed: int,
        config: GeneratedRollinSelectionConfig,
        bridge_tokens: list[int],
        branch_indices: tuple[int, ...],
    ) -> SimpleNamespace:
        assert config == GeneratedRollinSelectionConfig()
        assert bridge_tokens == []
        assert branch_indices == (0, 1)
        observed.append((base_seed, spec.recurrent_steps))
        return SimpleNamespace(
            execution_spec_sha256=spec.sha256,
            prompt_tokens_sha256=sha256_json(prompt_tokens),
            answer_tokens_sha256=sha256_json(answer_tokens),
            value=0.75,
            branch_values=(0.5, 1.0),
            branch_weights=(0.8, 0.2),
            answer_token_count=len(answer_tokens),
            receipt=lambda: {"receipt_sha256": "a" * 64},
        )

    monkeypatch.setattr(
        trainer,
        "generated_rollin_live_path_loss",
        fake_generated_loss,
    )
    monkeypatch.setattr(
        trainer,
        "validate_generated_rollin_receipt",
        lambda value: value,
    )
    config = ResidentSFTBootstrapConfig(
        seed=17,
        schema=TRAINER_CONFIG_SCHEMA_V2,
        objective=OBJECTIVE_NAME_V2,
        generated_rollin=GeneratedRollinSelectionConfig(),
        validation_examples=1,
    )
    summary = trainer._validation_summary(
        object(),
        [row],
        spec=RLCExecutionSpec(recurrent_steps=4),
        config=config,
    )

    assert len(observed) == 1
    assert observed[0][0] == summary["records"][0]["rollin_base_seed"]
    assert summary["objective"] == OBJECTIVE_NAME_V2
    assert summary["records"][0]["branch_weights"] == [0.8, 0.2]
    assert summary["records"][0]["objective_receipt_sha256"] == "a" * 64
    assert summary["records"][0]["objective_receipt"] == {
        "receipt_sha256": "a" * 64
    }


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


def test_adapter_topology_digest_ignores_values_but_not_shape_or_name() -> None:
    first = {
        "layer.lora_a": mx.zeros((2, 3)),
        "layer.lora_b": mx.ones((3, 4)),
    }
    changed_values = {
        "layer.lora_a": mx.ones((2, 3)),
        "layer.lora_b": mx.zeros((3, 4)),
    }
    changed_shape = {
        "layer.lora_a": mx.ones((3, 2)),
        "layer.lora_b": mx.zeros((3, 4)),
    }

    assert adapter_topology_sha256(first) == adapter_topology_sha256(changed_values)
    assert adapter_topology_sha256(first) != adapter_topology_sha256(changed_shape)
