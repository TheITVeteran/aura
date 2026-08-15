"""Contracts for teacher-free broad recurrent process decoding."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from core.brain.llm import unified_recurrent_transfer_decode as transfer  # noqa: E402
from core.learning.unified_intrinsic_objective import (  # noqa: E402
    UnifiedIntrinsicTrainingSpec,
)
from core.learning.unified_intrinsic_recurrence import (  # noqa: E402
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
)


def _controller() -> UnifiedRecurrentController:
    return UnifiedRecurrentController(
        UnifiedRecurrenceConfig(hidden_size=8, correction_rank=2)
    )


def _spec() -> UnifiedIntrinsicTrainingSpec:
    return UnifiedIntrinsicTrainingSpec(1, 2, (1, 2, 4), (8, 16))


def test_typed_process_decode_uses_slots_without_answer_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_logits(_model, tokens, plan, _controller, **kwargs):
        calls.append({"tokens": tokens.tolist(), "depth": plan.iterations, **kwargs})
        token_id = 7 if len(calls) == 1 else 9
        logits = mx.full((1, int(tokens.shape[1]) + 5, 16), -100.0)
        logits = logits.at[0, -1, token_id].add(200.0)
        return logits, SimpleNamespace()

    monkeypatch.setattr(transfer, "unified_recurrent_logits", fake_logits)
    generated, stopped, _latency = transfer.decode_typed_process_tokens(
        object(),
        _controller(),
        _spec(),
        (1, 2, 3),
        recurrence_depth=4,
        eos_token_id=None,
        max_tokens=4,
        typed_action_lesion=True,
        transition_processor_lesion=True,
        transition_history_lesion=True,
        completion_check=lambda values: len(values) == 2,
    )

    assert generated == (7, 9)
    assert stopped is True
    assert [row["tokens"] for row in calls] == [[[1, 2, 3]], [[1, 2, 3, 7]]]
    assert all(row["depth"] == 4 for row in calls)
    assert all(row["state_slot_start"] == 3 for row in calls)
    assert all(row["answer_digit_pointer_enabled"] is False for row in calls)
    assert all(row["typed_action_lesion"] is True for row in calls)
    assert all(row["transition_processor_lesion"] is True for row in calls)
    assert all(row["transition_history_lesion"] is True for row in calls)
    assert all(row["process_tape_lesion"] is False for row in calls)


def test_typed_process_decode_rejects_untyped_or_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions are invalid"):
        transfer.decode_typed_process_tokens(
            object(),
            object(),  # type: ignore[arg-type]
            _spec(),
            (1,),
            recurrence_depth=4,
            eos_token_id=None,
            max_tokens=1,
        )


def test_base_decode_conditions_on_and_returns_matched_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def generate_step(tokens, model, *, max_tokens, sampler):
        observed.update(
            tokens=tokens.tolist(),
            model=model,
            max_tokens=max_tokens,
            sampled=int(sampler(SimpleNamespace()).item()),
        )
        yield 7, None
        yield 8, None

    class ArgmaxResult:
        @staticmethod
        def item() -> int:
            return 5

    monkeypatch.setattr(
        transfer.mx,
        "argmax",
        lambda _values, axis=-1: ArgmaxResult(),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.generate",
        SimpleNamespace(generate_step=generate_step),
    )
    progress: list[int] = []
    seen: list[tuple[int, ...]] = []
    model = object()

    generated, stopped, _latency_ms = transfer.decode_base_greedy_tokens(
        model,
        (1, 2),
        eos_token_id=None,
        max_tokens=2,
        prefill_tokens=(3, 4),
        completion_check=lambda values: seen.append(values) or values[-1] == 8,
        progress=progress.append,
    )

    assert observed == {
        "tokens": [1, 2, 3, 4],
        "model": model,
        "max_tokens": 2,
        "sampled": 5,
    }
    assert generated == (3, 4, 7, 8)
    assert stopped is True
    assert progress == [1, 2]
    assert seen == [(3, 4, 7), (3, 4, 7, 8)]


def test_base_decode_rejects_invalid_prefill_tokens() -> None:
    with pytest.raises(ValueError, match="dimensions are invalid"):
        transfer.decode_base_greedy_tokens(
            object(),
            (1,),
            eos_token_id=None,
            max_tokens=1,
            prefill_tokens=(-1,),
        )
