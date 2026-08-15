"""Four things in the agency loop that measured the wrong quantity.

Social hunger and curiosity added a fixed increment per pulse, so the polling
rate decided how lonely and how curious she got: double the heartbeat and she
reaches SEEKING_CONTACT in half the wall time, having waited exactly as long.

Every heartbeat scheduled a pulse with no in-flight guard, and a pulse awaits
pathway hooks — so two interleave, both reading the same idle window and both
committing side effects against state the other is mutating.

Temporal greetings claimed overnight thought and watchkeeping from the clock
and an idle timer, with nothing recorded that could have made them false.

And sensory freshness was `min(since_last_visual, since_last_audio)` — the age
of the newest event of any kind — while the queue is emitted oldest first, so
one recent sound made an hours-old visual observation "fresh".
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from core.agency.agency_core import (
    _CURIOSITY_PER_S,
    _MAX_DYNAMICS_INTEGRATION_S,
    _OBSERVATION_STALE_S,
    _SOCIAL_HUNGER_PER_S,
    AgencyCore,
    AgencyState,
    EngagementMode,
    _observation_text,
    _observation_time,
)


def _core() -> AgencyCore:
    core = AgencyCore.__new__(AgencyCore)
    core.state = AgencyState()
    core.state.initiative_energy = 0.5
    core.state.social_hunger = 0.0
    core.state.curiosity_pressure = 0.0
    return core


def _drive_after(core: AgencyCore, *, seconds: float, pulses: int) -> tuple[float, float]:
    start = 1_000_000.0
    step = seconds / pulses
    core._update_social_dynamics(600.0, now=start)  # first call establishes the mark
    for i in range(1, pulses + 1):
        core._update_social_dynamics(600.0, now=start + i * step)
    return core.state.social_hunger, core.state.curiosity_pressure


def test_the_same_wall_time_produces_the_same_drives_at_any_poll_rate(monkeypatch):
    """Curiosity carries deliberate jitter, drawn once per update, so two
    polling rates never produce byte-identical curiosity and should not. What
    must not differ is the deterministic part: zero the jitter and the two
    rates have to agree exactly."""
    import core.runtime.managed_entropy as entropy_module

    monkeypatch.setattr(
        entropy_module,
        "get_managed_entropy",
        lambda: SimpleNamespace(get_curiosity_jitter=lambda intensity=1.0: 0.0),
    )

    slow = _drive_after(_core(), seconds=120.0, pulses=4)
    fast = _drive_after(_core(), seconds=120.0, pulses=120)

    assert slow[0] == pytest.approx(fast[0], rel=1e-9)
    assert slow[1] == pytest.approx(fast[1], rel=1e-9)


def test_the_jitter_is_scaled_by_elapsed_time_too():
    """Otherwise the noise itself reintroduces the poll-rate dependence: 120
    draws of a per-pulse jitter add thirty times more than four."""
    slow_total = 0.0
    fast_total = 0.0
    for _ in range(12):
        slow_total += _drive_after(_core(), seconds=120.0, pulses=4)[1]
        fast_total += _drive_after(_core(), seconds=120.0, pulses=120)[1]

    assert slow_total == pytest.approx(fast_total, rel=0.5), (
        f"slow={slow_total} fast={fast_total}"
    )


def test_more_wall_time_produces_more_pressure():
    short = _drive_after(_core(), seconds=60.0, pulses=10)
    long = _drive_after(_core(), seconds=600.0, pulses=10)

    assert long[0] > short[0]
    assert long[1] > short[1]


def test_the_first_call_integrates_nothing():
    core = _core()
    core._update_social_dynamics(600.0, now=1_000_000.0)

    assert core.state.social_hunger == 0.0
    assert core.state.curiosity_pressure == 0.0


def test_a_suspended_machine_does_not_wake_with_a_pinned_drive():
    """Six hours of real time is not six hours of idling she experienced."""
    core = _core()
    core._update_social_dynamics(600.0, now=1_000_000.0)
    core._update_social_dynamics(600.0, now=1_000_000.0 + 6 * 3600)

    ceiling = _SOCIAL_HUNGER_PER_S * 1.5 * _MAX_DYNAMICS_INTEGRATION_S
    assert core.state.social_hunger <= ceiling + 1e-9
    assert core.state.social_hunger < 1.0


def test_a_backwards_clock_integrates_nothing():
    core = _core()
    core._update_social_dynamics(600.0, now=1_000_000.0)
    core._update_social_dynamics(600.0, now=999_000.0)

    assert core.state.social_hunger == 0.0


def test_the_rates_are_per_second():
    assert _SOCIAL_HUNGER_PER_S > 0.0
    assert _CURIOSITY_PER_S > 0.0


class _Task:
    def __init__(self, done: bool):
        self._done = done

    def done(self) -> bool:
        return self._done


def test_a_heartbeat_arriving_mid_pulse_is_dropped(monkeypatch):
    import core.agency.agency_core as mod

    scheduled: list[str] = []
    monkeypatch.setattr(
        mod, "_schedule_agency_task", lambda coro, **kw: (coro.close(), scheduled.append(kw.get("name")))[0] or _Task(False)
    )

    core = _core()
    core._pulse_task = _Task(False)
    core.heartbeat()

    assert scheduled == [], "a second pulse was scheduled over one in flight"


def test_a_heartbeat_after_a_finished_pulse_runs(monkeypatch):
    import core.agency.agency_core as mod

    scheduled: list[str] = []

    def _fake(coro, **kw):
        coro.close()
        scheduled.append(kw.get("name"))
        return _Task(False)

    monkeypatch.setattr(mod, "_schedule_agency_task", _fake)

    core = _core()
    core._pulse_task = _Task(True)
    core.heartbeat()

    assert scheduled == ["agency.heartbeat.pulse"]


def test_a_greeting_claims_overnight_work_only_when_there_was_some(monkeypatch):
    core = _core()
    core.state.last_self_initiated_contact = 0.0
    core._autonomous_actions_since_contact = 0
    monkeypatch.setattr(time, "localtime", lambda *_a: time.struct_time((2026, 8, 15, 7, 0, 0, 4, 227, 0)))

    for _ in range(50):
        action = AgencyCore._pathway_temporal_rhythm(core, now=100_000.0, idle_seconds=600.0)
        if action is None:
            continue
        message = action["message"]
        assert "overnight" not in message
        assert "on my own" not in message


def test_a_greeting_may_report_work_that_happened(monkeypatch):
    core = _core()
    core.state.last_self_initiated_contact = 0.0
    core._autonomous_actions_since_contact = 4
    monkeypatch.setattr(time, "localtime", lambda *_a: time.struct_time((2026, 8, 15, 7, 0, 0, 4, 227, 0)))

    seen = set()
    for _ in range(200):
        action = AgencyCore._pathway_temporal_rhythm(core, now=100_000.0, idle_seconds=600.0)
        if action:
            seen.add(action["message"])

    assert any("4 things" in m for m in seen), seen


def test_a_recent_sound_does_not_make_an_old_sighting_fresh():
    core = _core()
    now = 500_000.0
    core.state.last_observation_comment = 0.0
    core.state.last_visual_change = now - 7200
    core.state.last_audio_event = now - 1  # a sound just now
    core.state.unshared_observations = [{"text": "the cat moved", "at": now - 7200}]

    action = AgencyCore._pathway_sensory_reactivity(core, now=now, idle_seconds=600.0)

    assert action is None, "reacted to a two-hour-old sighting as if it just happened"
    assert core.state.unshared_observations == [], "the stale entry blocks the queue"


def test_a_genuinely_fresh_observation_is_reacted_to():
    core = _core()
    now = 500_000.0
    core.state.last_observation_comment = 0.0
    core.state.unshared_observations = [{"text": "you just walked in", "at": now - 2}]

    action = AgencyCore._pathway_sensory_reactivity(core, now=now, idle_seconds=600.0)

    assert action is not None
    assert action["message"] == "you just walked in"
    assert action["priority"] == 0.8


def test_an_observation_written_the_old_way_is_not_lost():
    """A bare string from a writer that has not been updated is still a real
    thing she saw."""
    assert _observation_text("a plain string") == "a plain string"
    assert _observation_time("a plain string", default=42.0) == 42.0
    assert _observation_time({"text": "x"}, default=42.0) == 42.0
    assert _observation_time({"text": "x", "at": 7.0}, default=42.0) == 7.0


def test_observations_are_timestamped_when_recorded():
    core = _core()
    core.on_visual_change("something moved")
    core.on_audio_event("a door")

    for entry in core.state.unshared_observations:
        assert isinstance(entry, dict)
        assert entry["at"] > 0.0
    assert _OBSERVATION_STALE_S > 0.0
    assert core.state.engagement_mode is EngagementMode.ATTENTIVE_IDLE
