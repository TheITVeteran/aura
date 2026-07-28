"""A schedule's hash must name exactly one executable behaviour.

CP126 2e5d5cd6: execution kept the raw StageOp.alpha while canonical
serialization rounded it to six places before hashing. Two schedules whose
alphas differed past the sixth place ran differently but shared one hash —
and that hash keys receipts, the score cache, and promotion evidence, so one
schedule could inherit another's measured results.
"""
from __future__ import annotations

import math

import pytest

from core.brain.llm.latent_cortex.schedules import LayerSchedule, StageOp


def _sched(alpha):
    return LayerSchedule(ops=(StageOp(0, 10, 2, alpha=alpha),), name="s")


def test_the_executed_alpha_is_the_hashed_alpha():
    op = StageOp(0, 10, 2, alpha=0.12345678)

    assert op.alpha == round(0.12345678, StageOp.ALPHA_QUANTUM_PLACES)
    assert op.to_dict()["alpha"] == op.alpha


def test_alphas_differing_past_the_quantum_no_longer_collide():
    """They now collapse to ONE behaviour rather than two behaviours sharing
    one identity — distinctness is resolved, not hidden."""
    a = _sched(0.12345678)
    b = _sched(0.12345612)

    assert (a.schedule_hash == b.schedule_hash) == (a.ops[0].alpha == b.ops[0].alpha)


def test_behaviourally_distinct_alphas_keep_distinct_hashes():
    a = _sched(0.123457)
    b = _sched(0.123456)

    assert a.ops[0].alpha != b.ops[0].alpha
    assert a.schedule_hash != b.schedule_hash


def test_identical_alphas_share_a_hash():
    assert _sched(0.5).schedule_hash == _sched(0.5).schedule_hash


def test_an_exact_alpha_is_untouched():
    assert StageOp(0, 10, 2, alpha=0.5).alpha == 0.5
    assert StageOp(0, 10, 2, alpha=0.25).alpha == 0.25


def test_no_alpha_stays_none():
    assert StageOp(0, 10, 2).alpha is None


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
        pytest.param(10**10_000, id="unrepresentable"),
    ],
)
def test_an_unusable_alpha_is_left_for_validate_to_report(bad):
    """Quantization must not change WHERE bad programs are caught.

    Direct construction is permissive by design so validate() can report an
    unexecutable program rather than the caller crashing on it; from_dict is
    the strict gate for untrusted input. Quantizing had to preserve that.
    """
    op = StageOp(0, 10, 2, alpha=bad)          # constructs
    assert op.alpha is not None
    schedule = LayerSchedule(ops=(op,), name="s")
    assert schedule.validate(prelude_end=0, coda_start=10)  # reports problems


@pytest.mark.parametrize(
    "bad",
    [pytest.param(float("nan"), id="nan"), pytest.param(10**10_000, id="unrepresentable")],
)
def test_untrusted_input_is_still_refused_strictly(bad):
    with pytest.raises(ValueError):
        LayerSchedule.from_dict({"ops": [{"start": 0, "end": 10, "alpha": bad}]})


def test_round_tripping_preserves_identity():
    original = _sched(0.98765432)
    restored = LayerSchedule.from_dict(original.to_dict())

    assert restored.schedule_hash == original.schedule_hash
    assert restored.ops[0].alpha == original.ops[0].alpha


def test_existing_window_hashes_are_unchanged():
    """Schedules with no alpha override must keep their library hashes."""
    plain = LayerSchedule(ops=(StageOp(0, 10, 2),), name="p")

    assert plain.to_dict()["ops"][0] == {"start": 0, "end": 10, "repeats": 2}
