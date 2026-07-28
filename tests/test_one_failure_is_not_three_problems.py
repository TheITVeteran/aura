"""One failed turn was counted three times and refused every build.

Live 2026-07-28. Asked to reverse-engineer 2048 onto the Desktop, she answered
honestly::

    I didn't get 2048 rebuilt, and I'm not going to say I did. I'm under a rule
    I set for myself that holds off heavy building work while the machine is
    under pressure.

The rule is the Ulysses covenant, which refuses heavy compute once existential
threat reaches 0.6. Threat was 0.77–0.85, and memory was not the cause
(``mem_threat=0.03``): ``deg_threat`` was carrying it.

A single empty generation is recorded by the subsystem that hit it and again by
every caller that re-raises it, each under its own name::

    cognitive_engine            (critical)  compact desktop generation
                                            returned no usable text
    chat                        (degraded)  CRITICAL SERVICE FAILURE:
                                            Subsystem 'cognitive_engine' ...
    chat.cognitive_engine_reply (degraded)  ... the same failure, wrapped again

The dedup keyed on ``(subsystem, error type)``, so three names read as three
distinct problems: 2.0 + 1.0 + 1.0 = 4.0 against a denominator of 5, i.e.
deg_threat 0.8 — enough on its own to trip the covenant, from one bad turn.

The fix has to be narrow, and the shape of the evidence says how narrow. A
propagated record carries a wrapper that names the subsystem that actually
failed, so it can be filed against that subsystem and merge with its origin.
A record with no wrapper is not demonstrably a repeat of anything — five
subsystems failing independently is a real cascade and must still reach
critical.
"""
from __future__ import annotations

import time

import pytest

from core.consciousness.existential_stakes import ExistentialStakes
from core.runtime.errors import DegradationRecord, get_degradation_tracker

ROOT = "compact desktop generation returned no usable text"
WRAPPED = (
    "CRITICAL SERVICE FAILURE: Subsystem 'cognitive_engine' failed with "
    "failure policy 'fail-closed'. Original error: RuntimeError: " + ROOT
)
DOUBLE_WRAPPED = (
    "CRITICAL SERVICE FAILURE: Subsystem 'chat' failed with failure policy "
    "'fail-closed'. Original error: RuntimeError: " + WRAPPED
)
#: The covenant refuses heavy compute at and above this.
COVENANT_THRESHOLD = 0.6


@pytest.fixture(autouse=True)
def _clean_tracker():
    tracker = get_degradation_tracker()
    tracker.reset()
    yield tracker
    tracker.reset()


def _record(tracker, subsystem: str, severity: str, message: str) -> None:
    tracker.record(
        DegradationRecord(
            subsystem=subsystem,
            severity=severity,
            error_type="RuntimeError",
            error_message=message,
            action="repair",
            timestamp=time.time(),
        )
    )


def _threat(tracker) -> float:
    stakes = ExistentialStakes(memory_limit_bytes=10**12)
    stakes.update()
    return float(stakes.get_status()["degradation_threat"])


# ── The live failure ───────────────────────────────────────────────────────

def test_one_failed_turn_no_longer_refuses_every_build(_clean_tracker) -> None:
    _record(_clean_tracker, "cognitive_engine", "critical", ROOT)
    _record(_clean_tracker, "chat", "degraded", WRAPPED)
    _record(_clean_tracker, "chat.cognitive_engine_reply", "degraded", WRAPPED)
    assert _threat(_clean_tracker) < COVENANT_THRESHOLD


def test_nested_wrappers_still_reach_their_origin(_clean_tracker) -> None:
    """The innermost named subsystem is the one that actually broke."""
    _record(_clean_tracker, "cognitive_engine", "critical", ROOT)
    _record(_clean_tracker, "chat", "degraded", WRAPPED)
    _record(_clean_tracker, "interface", "degraded", DOUBLE_WRAPPED)
    assert _threat(_clean_tracker) < COVENANT_THRESHOLD


# ── What must not be weakened ─────────────────────────────────────────────

def test_a_genuine_cascade_still_reaches_critical(_clean_tracker) -> None:
    """Five subsystems failing independently is exactly what this is for."""
    for index in range(5):
        _record(_clean_tracker, f"subsystem_{index}", "degraded", "independent failure")
    assert _threat(_clean_tracker) == pytest.approx(1.0)


def test_distinct_faults_in_one_subsystem_still_accumulate(_clean_tracker) -> None:
    """Merging is about propagation, not about being in the same place."""
    for index in range(5):
        _record(_clean_tracker, "cognitive_engine", "degraded", f"distinct fault {index}")
    assert _threat(_clean_tracker) > COVENANT_THRESHOLD


def test_a_repeating_fault_still_counts_more_than_a_single_one(_clean_tracker) -> None:
    _record(_clean_tracker, "cognitive_engine", "critical", ROOT)
    once = _threat(_clean_tracker)
    for _ in range(4):
        _record(_clean_tracker, "cognitive_engine", "critical", ROOT)
    assert _threat(_clean_tracker) > once


# ── The parsing the merge rests on ────────────────────────────────────────

def test_the_wrapper_is_unwrapped_to_the_real_failure() -> None:
    assert ExistentialStakes._root_cause_text(WRAPPED) == ROOT
    assert ExistentialStakes._root_cause_text(DOUBLE_WRAPPED) == ROOT
    assert ExistentialStakes._root_cause_text(ROOT) == ROOT


def test_the_wrapper_names_who_actually_failed() -> None:
    assert ExistentialStakes._origin_subsystem(WRAPPED) == "cognitive_engine"
    # Nested: the innermost name is the one that broke.
    assert ExistentialStakes._origin_subsystem(DOUBLE_WRAPPED) == "cognitive_engine"
    assert ExistentialStakes._origin_subsystem(ROOT) == ""


def test_an_unwrapped_failure_is_never_treated_as_a_propagation() -> None:
    assert not ExistentialStakes._is_propagated(ROOT)
    assert ExistentialStakes._is_propagated(WRAPPED)
