from types import SimpleNamespace

from core.mind_tick import MindTick


def test_mind_tick_background_reasoning_pauses_on_event_loop_lag(monkeypatch):
    monkeypatch.setattr("core.runtime.background_policy.background_activity_reason", lambda *_args, **_kwargs: "")
    tick = MindTick.__new__(MindTick)
    tick.orchestrator = SimpleNamespace(
        _flow_controller=SimpleNamespace(
            snapshot=lambda _orch: SimpleNamespace(
                lag_seconds=0.22,
                load=0.20,
                overloaded=False,
                governor_mode="FULL",
            )
        ),
        _last_user_interaction_time=0.0,
    )

    reason = MindTick._background_reasoning_pause_reason(
        tick,
        SimpleNamespace(cognition=SimpleNamespace(current_objective="Hold continuity", active_goals=[])),
    )

    assert reason == "event_loop_lag"


def test_mind_tick_background_reasoning_requires_context(monkeypatch):
    monkeypatch.setattr("core.runtime.background_policy.background_activity_reason", lambda *_args, **_kwargs: "")
    tick = MindTick.__new__(MindTick)
    tick.orchestrator = SimpleNamespace(
        _flow_controller=SimpleNamespace(
            snapshot=lambda _orch: SimpleNamespace(
                lag_seconds=0.0,
                load=0.10,
                overloaded=False,
                governor_mode="FULL",
            )
        ),
        _last_user_interaction_time=0.0,
    )

    reason = MindTick._background_reasoning_pause_reason(
        tick,
        SimpleNamespace(cognition=SimpleNamespace(current_objective="", active_goals=[])),
    )

    assert reason == "no_reasoning_context"


def test_mind_tick_background_reasoning_requires_real_user_anchor(monkeypatch):
    seen = {}

    def _background_reason(*_args, **kwargs):
        seen.update(kwargs)
        return "no_user_anchor"

    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        _background_reason,
    )
    tick = MindTick.__new__(MindTick)
    tick.orchestrator = SimpleNamespace(_last_user_interaction_time=0.0)

    reason = MindTick._background_reasoning_pause_reason(
        tick,
        SimpleNamespace(
            cognition=SimpleNamespace(
                current_objective="Explore a standing objective",
                active_goals=[],
            )
        ),
    )

    assert reason == "no_user_anchor"
    assert seen["allow_no_user_anchor"] is False


def test_mind_tick_objective_attempts_are_singleflight_and_rate_limited():
    tick = MindTick.__new__(MindTick)
    tick._objective_attempt_key = ""
    tick._objective_attempt_inflight = ""
    tick._objective_next_attempt_at = 0.0

    assert tick._objective_attempt_defer_reason("Explore octopus cognition", now=100.0) == ""
    tick._begin_objective_attempt("Explore octopus cognition")
    assert (
        tick._objective_attempt_defer_reason("Explore octopus cognition", now=101.0)
        == "objective_inflight"
    )

    tick._finish_objective_attempt(
        "Explore octopus cognition",
        retry_after_s=60.0,
        now=101.0,
    )
    assert (
        tick._objective_attempt_defer_reason("Explore octopus cognition", now=160.9)
        == "objective_cooldown"
    )
    assert tick._objective_attempt_defer_reason("Explore octopus cognition", now=161.0) == ""
    assert tick._objective_attempt_defer_reason("Different objective", now=102.0) == ""
