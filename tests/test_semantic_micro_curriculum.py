"""Contracts for answer-blind semantic primitive acquisition."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from core.learning.recurrent_action_schema import (  # noqa: E402
    SEMANTIC_MICRO_OPCODES,
)
from core.learning.semantic_micro_curriculum import (  # noqa: E402
    semantic_micro_batch,
    semantic_micro_batch_receipt,
)
from core.learning.unified_intrinsic_objective import (  # noqa: E402
    unified_semantic_micro_primitive_loss,
)
from core.learning.unified_intrinsic_recurrence import (  # noqa: E402
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
)


def _controller() -> UnifiedRecurrentController:
    return UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=64,
            correction_rank=33,
            state_slots=11,
        )
    )


def test_semantic_micro_batch_is_balanced_replayable_and_seed_separated() -> None:
    width = len(SEMANTIC_MICRO_OPCODES) * 2
    first = semantic_micro_batch(seed=71, batch_size=width, batch_index=0)
    replay = semantic_micro_batch(seed=71, batch_size=width, batch_index=0)
    heldout = semantic_micro_batch(seed=72, batch_size=width, batch_index=0)

    assert first == replay
    assert {example.opcode for example in first} == SEMANTIC_MICRO_OPCODES
    assert {example.example_sha256 for example in first}.isdisjoint(
        {example.example_sha256 for example in heldout}
    )
    receipt = semantic_micro_batch_receipt(first)
    assert receipt["answers_present"] is False
    assert receipt["expected_states_present"] is False


def test_semantic_micro_objective_uses_executable_noninvalid_targets() -> None:
    examples = semantic_micro_batch(
        seed=73,
        batch_size=len(SEMANTIC_MICRO_OPCODES),
        batch_index=0,
    )

    loss, metrics = unified_semantic_micro_primitive_loss(
        _controller(),
        examples,
    )
    mx.eval(loss)

    assert float(loss.item()) > 0.0
    assert metrics["examples"] == len(SEMANTIC_MICRO_OPCODES)
    assert metrics["opcodes"] == sorted(SEMANTIC_MICRO_OPCODES)
    assert metrics["writable_registers"] > 0
    assert metrics["microcode_available_to_treatment"] is False
