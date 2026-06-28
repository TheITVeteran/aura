from __future__ import annotations

from types import SimpleNamespace

from core.runtime.timescale_bridge import (
    TimescaleBridge,
    render_timescale_prompt_block,
)


def _frame(
    *,
    at: float,
    app: str = "Aura Zenith",
    window: str = "Chat",
    cpu: float = 12.0,
    memory: float = 55.0,
    idle: float = 0.0,
    novelty: float = 0.1,
    threat: float = 0.0,
    social: float = 0.5,
):
    return SimpleNamespace(
        timestamp=at,
        screen=SimpleNamespace(
            active_app=app,
            window_title=window,
            screen_changed=novelty > 0.2,
        ),
        system=SimpleNamespace(
            cpu_percent=cpu,
            memory_percent=memory,
            thermal_pressure=threat,
        ),
        user=SimpleNamespace(idle_seconds=idle),
        audio=SimpleNamespace(voice_activity=False),
        novelty_score=lambda: novelty,
        threat_score=lambda: threat,
        social_signal=lambda: social,
    )


def test_timescale_bridge_reconciles_idle_without_inventing_events():
    bridge = TimescaleBridge(sample_interval_s=0, idle_anchor_threshold_s=300)
    bridge.ingest_perceptual_frame(
        _frame(at=1000.0, app="Finder", window="Downloads", idle=600.0),
    )
    bridge.reconcile_foreground_turn("first turn", now=1010.0)

    bridge.ingest_perceptual_frame(
        _frame(at=1800.0, app="Chrome", window="Search", cpu=42.0, memory=70.0, idle=700.0),
    )
    reconciliation = bridge.reconcile_foreground_turn("You with me?", now=1900.0)

    data = reconciliation.to_dict()
    assert data["user_returned_after_idle"] is True
    assert data["foreground_anchor_required"] is True
    assert data["narrative_drift_risk"] >= 0.35
    assert "Chrome" in data["observed_apps"]
    assert any("Do not invent events" in directive for directive in data["directives"])


def test_timescale_bridge_prompt_block_is_compact_grounding():
    bridge = TimescaleBridge(sample_interval_s=0, idle_anchor_threshold_s=60)
    bridge.ingest_perceptual_frame(_frame(at=10.0, app="Notes", window="Aura Journal"))
    bridge.reconcile_foreground_turn("hello", now=20.0)
    reconciliation = bridge.reconcile_foreground_turn("back now", now=120.0)

    block = render_timescale_prompt_block(reconciliation)

    assert "TIMESCALE RECONCILIATION" in block
    assert "Do not invent events" in block
    assert "do not recite it as telemetry" in block


def test_timescale_bridge_status_reports_observations_and_last_reconciliation():
    bridge = TimescaleBridge(sample_interval_s=0, idle_anchor_threshold_s=60)
    bridge.ingest_perceptual_frame(_frame(at=1.0, app="Terminal", window="Aura logs"))
    bridge.reconcile_foreground_turn("status", now=10.0)

    status = bridge.get_status()

    assert status["running"] is True
    assert status["observations"] == 1
    assert status["latest_observation"]["active_app"] == "Terminal"
    assert status["last_reconciliation"]["summary"]
