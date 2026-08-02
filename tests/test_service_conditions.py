"""Service conditions and finalizers: 'alive' and 'ready' are different claims.

Clean-room adoption of the Kubernetes condition convention and finalizer
guarantee. Aura's control plane already reconciled desired against observed
state with a generation counter; what it lacked was the part that makes a
status readable — independent named claims, and a guarantee that "stopped"
means cleanup actually finished.
"""
from __future__ import annotations

import pytest

from core.runtime.service_conditions import (
    ConditionSet,
    ConditionStatus,
    ConditionType,
    FinalizerSet,
)

pytestmark = pytest.mark.unit


# ── independent claims a single enum cannot express ────────────────────────


def test_ready_but_saturated_is_expressible():
    """A model can be loaded and usable while refusing new foreground work.
    One collapsed observed_state cannot say that."""
    conditions = ConditionSet()
    conditions.set(ConditionType.ALIVE, ConditionStatus.TRUE)
    conditions.set(ConditionType.READY, ConditionStatus.TRUE)
    conditions.set(ConditionType.ACCEPTING_WORK, ConditionStatus.FALSE,
                   reason="queue_full")

    assert conditions.summary() == "ready_but_saturated"
    assert conditions.is_true(ConditionType.READY)
    assert not conditions.is_true(ConditionType.ACCEPTING_WORK)


def test_degraded_while_recovering_is_not_down():
    """Degraded is running-but-not-fully-capable. A consumer that reads it as
    down takes a working service offline."""
    conditions = ConditionSet()
    conditions.set(ConditionType.ALIVE, ConditionStatus.TRUE)
    conditions.set(ConditionType.READY, ConditionStatus.TRUE)
    conditions.set(ConditionType.RECOVERING, ConditionStatus.TRUE)

    assert conditions.summary() == "recovering"
    assert conditions.is_true(ConditionType.ALIVE)


def test_unknown_is_not_a_failure():
    """UNKNOWN means the probe could not run. Collapsing it into FALSE turns
    'we could not check' into 'it is broken', which triggers recovery for a
    healthy service."""
    conditions = ConditionSet()
    conditions.set(ConditionType.ALIVE, ConditionStatus.UNKNOWN,
                   reason="probe_timeout")

    assert conditions.is_true(ConditionType.ALIVE) is False
    assert conditions.get(ConditionType.ALIVE).status is ConditionStatus.UNKNOWN


# ── transition times answer "how long has this been true?" ─────────────────


def test_reasserting_the_same_status_preserves_the_transition_time():
    """That question is what decides whether something is stuck, so a repeated
    observation must not reset the clock."""
    conditions = ConditionSet()
    first = conditions.set(ConditionType.ALIVE, ConditionStatus.TRUE, now=1000.0)
    again = conditions.set(ConditionType.ALIVE, ConditionStatus.TRUE, now=2000.0)

    assert again.last_transition_at == first.last_transition_at == 1000.0
    assert conditions.duration(ConditionType.ALIVE, now=2000.0) == 1000.0


def test_a_real_status_change_moves_the_transition_time():
    conditions = ConditionSet()
    conditions.set(ConditionType.ALIVE, ConditionStatus.TRUE, now=1000.0)
    changed = conditions.set(ConditionType.ALIVE, ConditionStatus.FALSE, now=2000.0)

    assert changed.last_transition_at == 2000.0


# ── generations stop a stale status being read as current ──────────────────


def test_conditions_go_stale_when_the_desired_configuration_changes():
    """A condition behind the object's generation describes a configuration
    that no longer exists."""
    conditions = ConditionSet()
    conditions.set(ConditionType.READY, ConditionStatus.TRUE)
    assert conditions.stale() == []

    conditions.bump_generation()

    stale = conditions.stale()
    assert [c.type for c in stale] == [ConditionType.READY]
    assert conditions.to_dict()["stale_conditions"] == ["ready"]


def test_reobserving_at_the_new_generation_clears_staleness():
    conditions = ConditionSet()
    conditions.set(ConditionType.READY, ConditionStatus.TRUE)
    conditions.bump_generation()
    conditions.set(ConditionType.READY, ConditionStatus.TRUE)

    assert conditions.stale() == []


# ── finalizers: stopped means cleaned up ───────────────────────────────────


def test_a_service_is_not_clear_until_its_cleanups_run():
    """'Stopped' is a receipt, not a hope. A lane declared cold while its
    worker still held 20GB is exactly this bug."""
    ran = []
    finalizers = FinalizerSet()
    finalizers.add("release_lease", lambda: ran.append("lease"))
    finalizers.add("reap_children", lambda: ran.append("children"))

    assert finalizers.is_clear is False
    assert set(finalizers.pending) == {"release_lease", "reap_children"}

    outcomes = finalizers.run_all()

    assert finalizers.is_clear is True
    assert set(ran) == {"lease", "children"}
    assert all(o.completed for o in outcomes)


def test_one_failing_cleanup_does_not_strand_the_others():
    """A released lease is worth having even if a temp file could not be
    removed."""
    ran = []
    finalizers = FinalizerSet()

    def _explode():
        raise OSError("device busy")

    finalizers.add("first", lambda: ran.append("first"))
    finalizers.add("broken", _explode)
    finalizers.add("last", lambda: ran.append("last"))

    outcomes = finalizers.run_all()

    assert set(ran) == {"first", "last"}
    assert {o.name for o in outcomes if o.completed} == {"first", "last"}
    failed = next(o for o in outcomes if not o.completed)
    assert "device busy" in failed.error


def test_a_failed_finalizer_is_retained_so_shutdown_stays_visibly_incomplete():
    """Dropping it would make the shutdown LOOK clean, which is the failure
    mode this whole mechanism exists to prevent."""
    finalizers = FinalizerSet()

    def _explode():
        raise RuntimeError("still holding the handle")

    finalizers.add("stubborn", _explode)
    finalizers.run_all()

    assert finalizers.is_clear is False
    assert "stubborn" in finalizers.pending


def test_a_retried_finalizer_can_succeed_later():
    attempts = {"n": 0}
    finalizers = FinalizerSet()

    def _flaky():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OSError("busy")

    finalizers.add("flaky", _flaky)
    finalizers.run_all()
    assert finalizers.is_clear is False

    finalizers.run_all()
    assert finalizers.is_clear is True


# ── wired into the live control plane ──────────────────────────────────────


def test_control_plane_observations_derive_conditions_from_reconciliation():
    """Conditions must be populated by real reconciliation, not be a parallel
    surface someone has to remember to update."""
    from core.runtime.control_plane import (
        DesiredServiceState,
        ObservedServiceState,
        RuntimeControlPlane,
        ServiceObservation,
    )

    observation = ServiceObservation(
        name="cortex", desired_state=DesiredServiceState.RUNNING
    )
    RuntimeControlPlane._transition(
        observation, ObservedServiceState.DEGRADED, "memory_pressure"
    )

    conditions = observation.condition_set()
    assert conditions.is_true(ConditionType.ALIVE)
    assert conditions.is_true(ConditionType.READY)
    assert conditions.is_true(ConditionType.DEGRADED)
    assert conditions.summary() == "degraded"
    assert observation.to_dict()["conditions"]["summary"] == "degraded"


def test_a_stopped_service_reports_not_alive():
    from core.runtime.control_plane import (
        DesiredServiceState,
        ObservedServiceState,
        RuntimeControlPlane,
        ServiceObservation,
    )

    observation = ServiceObservation(
        name="cortex", desired_state=DesiredServiceState.STOPPED
    )
    RuntimeControlPlane._transition(
        observation, ObservedServiceState.STOPPED, "shutdown"
    )

    conditions = observation.condition_set()
    assert not conditions.is_true(ConditionType.ALIVE)
    assert conditions.summary() == "down"
