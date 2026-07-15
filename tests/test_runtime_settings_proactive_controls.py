from __future__ import annotations

import json

import pytest

from core.autonomy import proactive_communication
from core.autonomy.proactive_communication import (
    EmotionalState,
    InterruptionUrgency,
    ProactiveCommunicationManager,
    ProactiveMessage,
)
from core.runtime import runtime_settings


@pytest.fixture(autouse=True)
def _isolated_runtime_settings(tmp_path, monkeypatch):
    path = tmp_path / "runtime.json"
    monkeypatch.setenv("AURA_SETTINGS_PATH", str(path))
    monkeypatch.setattr(
        proactive_communication,
        "_proactivity_suppressed_now",
        lambda _now=None: False,
    )
    runtime_settings.clear_runtime_settings_cache()
    yield path
    runtime_settings.clear_runtime_settings_cache()


def _write_settings(path, values):
    path.write_text(json.dumps(values), encoding="utf-8")
    runtime_settings.clear_runtime_settings_cache()


def test_proactive_modes_have_distinct_operational_policies(
    _isolated_runtime_settings,
):
    policies = {}
    manager = ProactiveCommunicationManager()
    for mode in ("minimal", "balanced", "frequent"):
        _write_settings(
            _isolated_runtime_settings,
            {"autonomy.proactive_messaging": mode},
        )
        status = manager.get_status()
        policies[mode] = (
            status["daily_message_limit"],
            status["minimum_interval_s"],
            status["idle_multiplier"],
        )

    assert policies["minimal"][0] < policies["balanced"][0] < policies["frequent"][0]
    assert policies["minimal"][1] > policies["balanced"][1] > policies["frequent"][1]
    assert policies["minimal"][2] > policies["balanced"][2] > policies["frequent"][2]


def test_proactive_daily_counters_reset_at_local_day_boundary():
    manager = ProactiveCommunicationManager()
    manager.messages_sent_today = 9
    manager.ordinary_messages_sent_today = 7
    manager._counter_day = "2000-01-01"

    manager._reset_daily_counters(1_800_000_000.0)

    assert manager.messages_sent_today == 0
    assert manager.ordinary_messages_sent_today == 0
    assert manager._counter_day != "2000-01-01"


@pytest.mark.asyncio
async def test_proactive_scheduler_releases_only_one_ready_normal_message(
    _isolated_runtime_settings,
    monkeypatch,
):
    _write_settings(
        _isolated_runtime_settings,
        {
            "autonomy.actions_enabled": True,
            "autonomy.proactive_messaging": "frequent",
        },
    )
    now = 1_800_000_000.0
    manager = ProactiveCommunicationManager()
    manager.last_interaction_time = now - 600.0
    manager.pending_messages.extend(
        [
            ProactiveMessage(
                "Medium priority",
                EmotionalState.CURIOUS,
                InterruptionUrgency.MEDIUM,
                timestamp=now - 20.0,
            ),
            ProactiveMessage(
                "High priority",
                EmotionalState.CONCERNED,
                InterruptionUrgency.HIGH,
                timestamp=now - 10.0,
            ),
        ]
    )
    delivered = []

    async def _deliver(message):
        delivered.append(message.content)
        return True

    monkeypatch.setattr(manager, "_send_msg", _deliver)

    await manager._process_pending(now)

    assert delivered == ["High priority"]
    assert [message.content for message in manager.pending_messages] == [
        "Medium priority"
    ]


@pytest.mark.asyncio
async def test_critical_proactive_message_bypasses_ordinary_limits(
    _isolated_runtime_settings,
    monkeypatch,
):
    _write_settings(
        _isolated_runtime_settings,
        {
            "autonomy.actions_enabled": True,
            "autonomy.proactive_messaging": "minimal",
        },
    )
    now = 1_800_000_000.0
    manager = ProactiveCommunicationManager()
    manager.last_interaction_time = now
    manager.ordinary_messages_sent_today = manager._cadence_policy().daily_limit
    manager.unanswered_count = manager.max_unanswered
    manager.pending_messages.extend(
        [
            ProactiveMessage(
                "Ordinary update",
                EmotionalState.CURIOUS,
                InterruptionUrgency.HIGH,
            ),
            ProactiveMessage(
                "Critical security alert",
                EmotionalState.CONCERNED,
                InterruptionUrgency.CRITICAL,
            ),
        ]
    )
    delivered = []

    async def _deliver(message):
        delivered.append(message.content)
        return True

    monkeypatch.setattr(manager, "_send_msg", _deliver)

    await manager._process_pending(now)

    assert delivered == ["Critical security alert"]
    assert [message.content for message in manager.pending_messages] == [
        "Ordinary update"
    ]


@pytest.mark.asyncio
async def test_failed_proactive_delivery_is_retained_with_backoff(
    _isolated_runtime_settings,
    monkeypatch,
):
    _write_settings(
        _isolated_runtime_settings,
        {
            "autonomy.actions_enabled": True,
            "autonomy.proactive_messaging": "balanced",
        },
    )
    now = 1_800_000_000.0
    manager = ProactiveCommunicationManager()
    manager.last_interaction_time = now - 600.0
    message = ProactiveMessage(
        "Retry this governed delivery",
        EmotionalState.CURIOUS,
        InterruptionUrgency.HIGH,
    )
    manager.pending_messages.append(message)

    async def _fail(_message):
        return False

    monkeypatch.setattr(manager, "_send_msg", _fail)

    await manager._process_pending(now)

    assert list(manager.pending_messages) == [message]
    assert message.delivery_attempts == 1
    assert message.next_attempt_at > now
