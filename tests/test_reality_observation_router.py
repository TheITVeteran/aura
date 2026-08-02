from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.perception.multimodal_sync import Modality
from core.reality_reach.contracts import (
    ChannelDeclaration,
    ChannelKind,
    CouplingClass,
    EvidenceLevel,
    NumericDomain,
    RealityLayer,
)
from core.reality_reach.live import ChannelReading, ReadingStatus, RealityReachService
from core.reality_reach.observation_router import (
    ObservationSubscription,
    RealityObservationRouter,
)


def _declaration(
    channel_id: str,
    *,
    tags: tuple[str, ...] = (),
    sample_rate_hz: float = 1.0,
) -> ChannelDeclaration:
    return ChannelDeclaration(
        channel_id=channel_id,
        kind=ChannelKind.SENSOR,
        observable=f"observable.{channel_id.rsplit('.', 1)[-1]}",
        unit="percent",
        domain=NumericDomain(0.0, 100.0),
        coupling=CouplingClass.NETWORK,
        reality_layers=(RealityLayer.EFFECTIVE,),
        evidence_level=EvidenceLevel.P2,
        owner="tests",
        resolution=0.1,
        sample_rate_hz=sample_rate_hz,
        max_latency_s=1.0,
        stale_after_s=10.0,
        reference_id=f"reference.{channel_id.rsplit('.', 1)[-1]}",
        compliance_tags=tags,
        coupling_validated=True,
    )


def _reading(
    channel_id: str,
    value: float | None,
    *,
    captured_at_ns: int | None = None,
    status: ReadingStatus = ReadingStatus.AVAILABLE,
) -> ChannelReading:
    return ChannelReading(
        channel_id=channel_id,
        value=value,
        unit="percent",
        captured_at_ns=captured_at_ns or time.time_ns(),
        status=status,
        source="test.sensor",
        uncertainty=0.1,
    )


class AsyncAdapter:
    def __init__(
        self,
        adapter_id: str,
        declaration: ChannelDeclaration,
        *,
        value: float = 1.0,
        delay_s: float = 0.0,
    ) -> None:
        self.adapter_id = adapter_id
        self.declaration = declaration
        self.value = value
        self.delay_s = delay_s
        self.refreshes = 0
        self.current = _reading(declaration.channel_id, value)

    def declarations(self) -> tuple[ChannelDeclaration, ...]:
        return (self.declaration,)

    def read(self) -> tuple[ChannelReading, ...]:
        return (self.current,)

    async def refresh_readback(self) -> ChannelReading:
        self.refreshes += 1
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        self.current = _reading(
            self.declaration.channel_id,
            self.value,
            captured_at_ns=time.time_ns() + self.refreshes,
        )
        return self.current


@pytest.mark.asyncio
async def test_router_is_bounded_and_evicts_lower_salience_for_alarm() -> None:
    router = RealityObservationRouter(RealityReachService(), queue_limit=8)
    for index in range(8):
        receipt = await router.submit(
            _declaration(f"test.sensor_{index}"),
            _reading(f"test.sensor_{index}", float(index)),
            adapter_id=f"test.adapter_{index}",
        )
        assert receipt.accepted is True

    alarm = await router.submit(
        _declaration("test.alarm", tags=("alarm",)),
        _reading("test.alarm", 1.0),
        adapter_id="test.alarm_adapter",
    )

    assert alarm.accepted is True
    assert alarm.evicted_observation_id
    assert router.status()["queue_depth"] == 8
    assert router.status()["overflow_drops"] == 1
    assert "test.alarm" in router.latest()


@pytest.mark.asyncio
async def test_router_applies_deadband_and_temporary_focus(monkeypatch) -> None:
    import core.reality_reach.observation_router as router_module
    clock = {"value": 100.0}
    monkeypatch.setattr(router_module.time, "monotonic", lambda: clock["value"])
    router = RealityObservationRouter(RealityReachService())
    router.configure_subscription(
        ObservationSubscription(
            subscription_id="test.deadband",
            selector="test.temperature",
            max_rate_hz=20.0,
            min_delta=2.0,
            min_salience=0.0,
        )
    )
    first = await router.submit(
        _declaration("test.temperature"),
        _reading("test.temperature", 20.0),
        adapter_id="test.thermometer",
    )
    clock["value"] += 1.0
    below_delta = await router.submit(
        _declaration("test.temperature"),
        _reading("test.temperature", 20.5, captured_at_ns=time.time_ns() + 1),
        adapter_id="test.thermometer",
    )
    focus = router.focus("test.temperature", duration_s=5.0, max_rate_hz=4.0)

    assert first.accepted is True
    assert below_delta.accepted is False
    assert below_delta.reason == "below_min_delta"
    assert focus.selector == "test.temperature"
    assert focus.expires_monotonic == pytest.approx(106.0)


@pytest.mark.asyncio
async def test_router_delivers_bounded_claims_to_perception_and_cognition(
    monkeypatch,
) -> None:
    events = []
    cognitive: list[tuple[tuple, dict]] = []
    synchronizer = SimpleNamespace(ingest=events.append)
    advanced = SimpleNamespace(
        observe_state=lambda *args, **kwargs: cognitive.append((args, kwargs))
    )
    services = {
        "multimodal_synchronizer": synchronizer,
        "advanced_cognition": advanced,
    }
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: services.get(name, default)),
        raising=False,
    )
    router = RealityObservationRouter(
        RealityReachService(),
        poll_interval_s=60.0,
        max_delivery_rate_hz=100.0,
    )
    await router.start()
    try:
        receipt = await router.submit(
            _declaration("test.room_temperature"),
            _reading("test.room_temperature", 24.5),
            adapter_id="test.room_sensor",
        )
        assert receipt.accepted is True
        for _ in range(50):
            if events and cognitive:
                break
            await asyncio.sleep(0.01)
    finally:
        await router.stop()

    assert len(events) == 1
    assert events[0].modality == Modality.DEVICE
    assert events[0].privacy.raw_retained is False
    assert cognitive[0][0][0] == "physical_environment"
    payload = cognitive[0][0][1]
    assert payload["channel_id"] == "test.room_temperature"
    assert payload["value"] == 24.5


@pytest.mark.asyncio
async def test_sampler_timeout_isolated_and_fresh_sampler_wins_cached_reading() -> None:
    service = RealityReachService()
    router = RealityObservationRouter(service, sampler_timeout_s=0.1)
    fast = AsyncAdapter("test.fast", _declaration("test.fast_value"), value=42.0)
    slow = AsyncAdapter(
        "test.slow",
        _declaration("test.slow_value"),
        value=7.0,
        delay_s=0.2,
    )
    service.register_adapter(fast)
    service.register_adapter(slow)
    router.register_sampler(fast)
    router.register_sampler(slow)

    accepted = await router.poll_once()

    assert accepted == 1
    assert router.status()["sampler_failures"] == 1
    assert router.latest()["test.fast_value"]["reading"]["value"] == 42.0
    assert "test.slow_value" not in router.latest()
    assert fast.refreshes == 1
    assert slow.refreshes == 1
