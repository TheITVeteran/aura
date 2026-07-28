"""Every instruction that spends compute must count against the ceiling.

CP126 bd211d76: only window ops counted, so exchange, savepoint and
verify_probe were free — and verify_probe decodes text and invokes the
episode verifier, by far the most expensive instruction in the set. A
schedule could sit under a layer-repeat ceiling while doing a great deal of
real work, which is how a budget stops bounding anything.
"""
from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.schedules import (
    MAX_TOTAL_LAYER_REPEATS,
    NON_WINDOW_LAYER_EQUIVALENTS,
    VERIFY_PROBE_LAYER_EQUIVALENT,
    LayerSchedule,
    StageOp,
    op_layer_equivalents,
)


def test_a_window_op_costs_its_layer_applications():
    assert op_layer_equivalents(StageOp(0, 8, 3)) == 24


@pytest.mark.parametrize("kind", ["exchange", "savepoint", "verify_probe"])
def test_every_non_window_op_has_a_declared_cost(kind):
    assert op_layer_equivalents(StageOp(kind=kind)) > 0
    assert NON_WINDOW_LAYER_EQUIVALENTS[kind] > 0


def test_a_verifier_probe_is_the_expensive_instruction():
    """It decodes and runs a verifier; costing it like a savepoint is why the
    ceiling could be evaded."""
    assert VERIFY_PROBE_LAYER_EQUIVALENT > NON_WINDOW_LAYER_EQUIVALENTS["savepoint"]
    assert VERIFY_PROBE_LAYER_EQUIVALENT > NON_WINDOW_LAYER_EQUIVALENTS["exchange"]


def test_the_estimate_counts_what_the_exact_window_total_does_not():
    schedule = LayerSchedule(
        ops=(StageOp(0, 4, 1), StageOp(kind="savepoint"), StageOp(kind="verify_probe")),
        name="s",
    )

    assert schedule.total_layer_repeats == 4          # exact, windows only
    assert schedule.estimated_layer_equivalents > 4   # whole program


def test_a_schedule_under_the_window_ceiling_is_caught_on_real_cost():
    """The evasion this closes: legal on windows, enormous on probes."""
    ops = (StageOp(0, 64, 63),) + tuple(
        StageOp(kind="verify_probe") for _ in range(4)
    )
    schedule = LayerSchedule(ops=ops, name="evasive")

    assert schedule.total_layer_repeats <= MAX_TOTAL_LAYER_REPEATS
    problems = schedule.validate(prelude_end=0, coda_start=64)

    assert any("layer-equivalents exceeds" in p for p in problems)


def test_an_ordinary_schedule_is_unaffected():
    schedule = LayerSchedule(
        ops=(StageOp(0, 4, 1), StageOp(kind="savepoint"), StageOp(kind="verify_probe")),
        name="ok",
    )

    assert schedule.validate(prelude_end=0, coda_start=4) == []


def test_the_two_totals_are_named_so_they_cannot_be_confused():
    schedule = LayerSchedule(ops=(StageOp(0, 4, 1),), name="s")

    assert schedule.total_layer_repeats == schedule.estimated_layer_equivalents == 4


def test_a_malformed_window_costs_nothing_rather_than_raising():
    assert op_layer_equivalents(StageOp("a", "b", 1)) == 0
    assert op_layer_equivalents(StageOp(4, 4, 1)) == 0
    assert op_layer_equivalents(StageOp(0, 4, 0)) == 0
