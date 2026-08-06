"""tests/test_participation_probe.py — "online" and "never once ran" must differ.

This codebase's comments record the same defect several times over: steering
components instantiated and reporting online while their live-affect callbacks
were never passed into the token sentinel, so dozens of expected affect updates
never occurred; phi residual sampling producing zero samples across several
turns; a process boundary silently preventing the intended shared data path.
Every time, the health report said the component was fine.

It said so honestly. `liveness_check` names a READINESS predicate — `is_ready`,
`is_initialized` — and a component that is constructed, registered, healthy and
wired to nothing answers those correctly. The report was asking the only
question that could not detect the fault.

`participation_check` asks the other one. These tests pin the distinction, and
in particular pin that UNMEASURED is not reported as "never used" — trading a
false "everything is fine" for a false "everything is broken" gets the check
switched off faster than the original problem ever would.
"""

from __future__ import annotations

import pytest

from core.runtime.health_contract import (
    ServiceRequirement,
    ServiceStatus,
    ServiceTier,
    _read_participation,
)


def _requirement(**kwargs) -> ServiceRequirement:
    defaults = dict(
        name="Steering",
        container_key="affective_steering",
        tier=ServiceTier.IMPORTANT,
        description="Injects affect vectors during generation.",
    )
    defaults.update(kwargs)
    return ServiceRequirement(**defaults)


# ── the defect this exists to catch ──────────────────────────────────────


def test_a_ready_component_that_never_ran_is_idle_not_healthy():
    """The exact shape of the steering defect.

    Present, liveness green, zero invocations. Before this, that rendered
    identically to a component doing its job on every turn.
    """
    status = ServiceStatus(
        requirement=_requirement(participation_check="invocation_count"),
        present=True,
        liveness_ok=True,
        participation=0.0,
    )

    assert status.state == "idle"
    assert status.has_participated is False


def test_a_component_doing_work_is_participating():
    status = ServiceStatus(
        requirement=_requirement(participation_check="invocation_count"),
        present=True,
        liveness_ok=True,
        participation=57.0,
    )

    assert status.state == "participating"
    assert status.has_participated is True


def test_unmeasured_is_not_reported_as_never_used():
    """`None` is a real answer. Collapsing it into False is a new false alarm.

    Most services declare no participation probe. Reporting all of them as
    "never used" would bury the handful that genuinely are, which is the same
    failure as burying them under "healthy" — just louder.
    """
    status = ServiceStatus(
        requirement=_requirement(),  # no participation_check
        present=True,
        liveness_ok=True,
    )

    assert status.participation is None
    assert status.has_participated is None
    assert status.state == "unmeasured"


def test_absent_beats_every_other_state():
    status = ServiceStatus(
        requirement=_requirement(participation_check="invocation_count"),
        present=False,
        participation=99.0,
    )
    assert status.state == "absent"


# ── the probe reader ─────────────────────────────────────────────────────


class _Service:
    def __init__(self, value):
        self._value = value

    def invocation_count(self):
        if isinstance(self._value, BaseException):
            raise self._value
        return self._value


@pytest.mark.parametrize(
    "returned,expected",
    [
        (0, 0.0),
        (12, 12.0),
        (3.5, 3.5),
        (True, 1.0),
        (False, 0.0),
        # A probe that cannot answer must not be guessed at.
        ("many", None),
        (None, None),
    ],
)
def test_the_probe_reader_coerces_honestly(returned, expected):
    requirement = _requirement(participation_check="invocation_count")
    assert _read_participation(_Service(returned), requirement) == expected


@pytest.mark.parametrize(
    "exc", [RuntimeError("nope"), AttributeError("gone"), ValueError("bad"), OSError("io")]
)
def test_a_probe_that_raises_is_unmeasured_not_zero(exc):
    """A broken probe is not evidence that the component is idle.

    Reporting zero here would manufacture exactly the alarm this file exists to
    make trustworthy.
    """
    requirement = _requirement(participation_check="invocation_count")
    assert _read_participation(_Service(exc), requirement) is None


def test_a_missing_probe_method_is_unmeasured():
    requirement = _requirement(participation_check="does_not_exist")
    assert _read_participation(_Service(5), requirement) is None


def test_no_declared_probe_is_unmeasured():
    assert _read_participation(_Service(5), _requirement()) is None


def test_a_missing_service_is_unmeasured():
    requirement = _requirement(participation_check="invocation_count")
    assert _read_participation(None, requirement) is None


def test_reading_participation_never_raises():
    """This runs inside the health probe. An exception here takes down the
    report that exists to tell you what is wrong."""

    class Hostile:
        @property
        def invocation_count(self):
            raise RuntimeError("even attribute access explodes")

    requirement = _requirement(participation_check="invocation_count")
    assert _read_participation(Hostile(), requirement) is None
