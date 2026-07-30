from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.governance.disclosure_policy import DisclosurePolicy, SocialContext
from core.ops import hypervisor as hypervisor_module
from core.ops.hypervisor import Hypervisor
from core.resilience.resource_governor import ResourceGovernor
from core.runtime import background_policy


def test_background_uptime_uses_current_process_incarnation(monkeypatch) -> None:
    monkeypatch.setattr(background_policy, "_PROCESS_STARTED_AT", 990.0)
    monkeypatch.setattr(background_policy.time, "time", lambda: 1_000.0)
    restored = SimpleNamespace(start_time=100.0, status=SimpleNamespace(start_time=100.0))

    assert background_policy._runtime_uptime_seconds(restored) == 10.0


def test_resource_governor_ignores_facade_without_mutable_belief_store(monkeypatch) -> None:
    facade = SimpleNamespace(query=lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: facade if name == "world_model" else default,
    )

    assert ResourceGovernor()._trim_world_model() == 0


def test_disclosure_policy_import_path_is_unambiguous() -> None:
    policy = DisclosurePolicy(trusted_set={"Bryan"})
    assert policy.decide(
        SocialContext(
            is_trusted_channel=True,
            user="Bryan",
            is_public=False,
            direct_identity_question=False,
            risk_of_harm_high=False,
        )
    ) == "disclose"


@pytest.mark.asyncio
async def test_hypervisor_uses_monotonic_clock_across_wall_clock_jump(monkeypatch) -> None:
    hypervisor = Hypervisor(lag_threshold_s=1.5)
    hypervisor._running = True
    hypervisor._start_time = 100.0
    monotonic_values = iter((50.0, 51.0))

    async def one_sleep(_seconds):
        hypervisor._running = False

    monkeypatch.setattr(hypervisor_module.asyncio, "sleep", one_sleep)
    monkeypatch.setattr(hypervisor_module, "_monotonic_now", lambda: next(monotonic_values))
    monkeypatch.setattr(hypervisor_module.time, "time", lambda: 1_000.0)
    monkeypatch.setattr(hypervisor, "_active_runtime_reason", lambda: "")

    await hypervisor._watchdog_loop()

    assert hypervisor._last_lag == 0.0
    assert hypervisor._severe_lag_streak == 0


@pytest.mark.asyncio
async def test_hypervisor_classifies_boot_grace_lag_as_startup_telemetry(
    monkeypatch, caplog
) -> None:
    hypervisor = Hypervisor(lag_threshold_s=1.5)
    hypervisor._running = True
    hypervisor._start_time = 995.0
    monotonic_values = iter((50.0, 58.0))

    async def one_sleep(_seconds):
        hypervisor._running = False

    monkeypatch.setattr(hypervisor_module.asyncio, "sleep", one_sleep)
    monkeypatch.setattr(hypervisor_module, "_monotonic_now", lambda: next(monotonic_values))
    monkeypatch.setattr(hypervisor_module.time, "time", lambda: 1_000.0)
    monkeypatch.setattr(hypervisor, "_active_runtime_reason", lambda: "")

    with caplog.at_level("INFO", logger="Aura.Hypervisor"):
        await hypervisor._watchdog_loop()

    assert hypervisor._last_lag == 7.0
    assert hypervisor._severe_lag_streak == 0
    assert "retained as startup telemetry" in caplog.text
    assert not any(record.levelname == "WARNING" for record in caplog.records)
