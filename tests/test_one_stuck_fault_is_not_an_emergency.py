"""One problem repeating is one problem.

Survival threat summed every degradation record in the window, so a single
stuck fault could saturate it alone. Measured live 2026-07-27: an empty draft
recorded once per turn by one fail-closed subsystem drove deg_threat to 1.00 on
a host at 4% memory pressure. existential_threat is max(memory, degradation) by
design, so it pinned too, which tripped the Ulysses covenant's "no heavy compute
while survival is threatened", which refused every build the owner asked for.
The runtime was healthy and reporting an emergency.

Fixing that one subsystem removed that one instance. The measurement was still
wrong: any recurring fault could do it again, and the next one would look just
as much like a real cascade.

A fault repeating IS worse than a fault happening once, so repeats still count —
with diminishing weight, so N repetitions of one problem can never outweigh N
genuinely different ones. The escalation governor already caps the *rate* of
re-escalation for the same reason; this applies the same principle to the
measurement that gates survival.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from core.consciousness.existential_stakes import ExistentialStakes


def _record(subsystem: str, error_type: str, *, severity: str = "critical", age: float = 0.0):
    return SimpleNamespace(
        subsystem=subsystem,
        error_type=error_type,
        error_message=f"{subsystem}:{error_type}",
        severity=severity,
        timestamp=time.time() - age,
    )


def _weigh(records) -> float:
    """Reproduce the aggregation the threat computation performs."""
    now = time.time()
    seen: dict[tuple[str, str], int] = {}
    total = 0.0
    for record in records:
        weight = ExistentialStakes._degradation_record_weight(record, now=now)
        if weight <= 0.0:
            continue
        signature = (
            str(getattr(record, "subsystem", "") or "unknown"),
            str(getattr(record, "error_type", "") or "")
            or str(getattr(record, "error_message", "") or "")[:80],
        )
        repeat = seen.get(signature, 0)
        seen[signature] = repeat + 1
        total += weight / (1.0 + repeat) ** 2
    return total


def test_one_fault_repeating_weighs_less_than_that_many_distinct_faults() -> None:
    same = [_record("personality_engine", "RuntimeError") for _ in range(10)]
    different = [_record(f"subsystem_{i}", "RuntimeError") for i in range(10)]
    assert _weigh(same) < _weigh(different)


def test_a_repeating_fault_cannot_saturate_survival_threat_alone() -> None:
    """The live failure: one subsystem, one message, once per turn, forever."""
    from core.consciousness.existential_stakes import DEGRADATION_THREAT_DENOMINATOR

    stuck = [_record("personality_engine", "RuntimeError") for _ in range(40)]
    assert _weigh(stuck) < DEGRADATION_THREAT_DENOMINATOR, (
        "a single stuck fault reached the saturation point on its own"
    )


def test_a_genuine_cascade_still_registers() -> None:
    """Tolerance is not blindness — many different things breaking is a cascade."""
    from core.consciousness.existential_stakes import DEGRADATION_THREAT_DENOMINATOR

    cascade = [_record(f"subsystem_{i}", "RuntimeError") for i in range(12)]
    assert _weigh(cascade) >= DEGRADATION_THREAT_DENOMINATOR


def test_the_first_occurrence_keeps_its_full_weight() -> None:
    """A new fault is never discounted; only its repeats are."""
    one = _weigh([_record("a", "X")])
    assert one > 0.0
    assert _weigh([_record("a", "X"), _record("a", "X")]) == pytest.approx(one * 1.25, rel=0.01)


def test_different_errors_from_one_subsystem_are_different_problems() -> None:
    """A subsystem failing two ways is two things wrong, not one repeated."""
    two_ways = _weigh([_record("mlx_client", "OSError"), _record("mlx_client", "TimeoutError")])
    one_way = _weigh([_record("mlx_client", "OSError"), _record("mlx_client", "OSError")])
    assert two_ways > one_way


def test_stale_records_still_age_out() -> None:
    from core.consciousness.existential_stakes import DEGRADATION_THREAT_WINDOW_S

    assert _weigh([_record("a", "X", age=DEGRADATION_THREAT_WINDOW_S + 5)]) == 0.0


def test_the_aggregation_lives_in_the_threat_computation() -> None:
    """A comment is not a mechanism; the discount must be in the summation."""
    import inspect

    source = inspect.getsource(ExistentialStakes.update)
    assert "weight / (1.0 + repeat) ** 2" in source
