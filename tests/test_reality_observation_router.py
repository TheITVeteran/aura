from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.perception.multimodal_sync import Modality
from core.reality_reach import digital_twin as digital_twin_module
from core.reality_reach.contracts import (
    ChannelDeclaration,
    ChannelKind,
    CouplingClass,
    EvidenceLevel,
    NumericDomain,
    RealityLayer,
)
from core.reality_reach.digital_twin import (
    RealityDigitalTwinGraph,
    TwinDisposition,
    TwinReceipt,
)
from core.reality_reach.historian import (
    HistorianCorruptionError,
    HistorianHeadPage,
    HistorianHeadSnapshot,
    RealityHistorian,
)
from core.reality_reach.live import (
    AdapterInventoryEntry,
    ChannelReading,
    ReadingStatus,
    RealityReachService,
)
from core.reality_reach.observation_router import (
    ObservationSubscription,
    RealityObservation,
    RealityObservationRouter,
)
from core.runtime.audit_chain import canonical_json, sha256_hex


class _MemoryKeychainBackend:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, account: str) -> str | None:
        return self.values.get((service, account))

    def set_password(self, service: str, account: str, password: str) -> bool:
        self.values[(service, account)] = password
        return True


@pytest.fixture(autouse=True)
def _isolated_digital_twin_keychain(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _MemoryKeychainBackend()
    monkeypatch.setattr(
        digital_twin_module,
        "require_keychain_backend",
        lambda: backend,
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
    source_epoch: str = "",
    source_sequence: int = 0,
    source_quality: str = "",
) -> ChannelReading:
    return ChannelReading(
        channel_id=channel_id,
        value=value,
        unit="percent",
        captured_at_ns=captured_at_ns or time.time_ns(),
        status=status,
        source="test.sensor",
        uncertainty=0.1,
        source_epoch=source_epoch,
        source_sequence=source_sequence,
        source_quality=source_quality,
    )


class AsyncAdapter:
    def __init__(
        self,
        adapter_id: str,
        declaration: ChannelDeclaration,
        *,
        value: float = 1.0,
        delay_s: float = 0.0,
        physical_identity_sha256: str = "",
    ) -> None:
        self.adapter_id = adapter_id
        self.physical_identity_sha256 = physical_identity_sha256
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


def _accepting_synchronizer(events: list) -> SimpleNamespace:
    def _ingest(event):
        events.append(event)
        return SimpleNamespace(
            event_id=event.event_id,
            accepted=True,
            reason="accepted",
        )

    return SimpleNamespace(
        ingest=_ingest,
        get_status=lambda: {
            "accepted_events": len(events),
            "rejected_events": 0,
            "queue_overflow_drops": 0,
            "late_events": 0,
            "fusions": 0,
            "queue_depths": {"proprioception": 0},
            "queue_limit": 64,
        },
    )


def _accepting_cognition(cognitive: list[tuple[tuple, dict]]) -> SimpleNamespace:
    def _observe(*args, **kwargs):
        cognitive.append((args, kwargs))
        return {"receipt_id": f"advanced.receipt.{len(cognitive)}"}

    return SimpleNamespace(
        observe_state=_observe,
        health_report=lambda: {"ok": True},
    )


def _digital_twin(tmp_path, *, name: str = "twin.sqlite3") -> RealityDigitalTwinGraph:
    return RealityDigitalTwinGraph(
        tmp_path / name,
        session_id=f"test.router.{name.removesuffix('.sqlite3')}",
    )


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
    synchronizer = _accepting_synchronizer(events)
    advanced = _accepting_cognition(cognitive)
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


@pytest.mark.asyncio
async def test_async_sampler_uses_canonical_normalization_and_preserves_lineage() -> None:
    service = RealityReachService(session_id="test.router.normalization")
    adapter = AsyncAdapter(
        "test.out_of_domain",
        _declaration("test.out_of_domain_value"),
        value=120.0,
    )
    service.register_adapter(adapter)
    router = RealityObservationRouter(service)
    router.register_sampler(adapter)

    accepted = await router.poll_once()

    assert accepted == 1
    reading = router.latest()["test.out_of_domain_value"]["reading"]
    assert reading["status"] == "degraded"
    assert reading["error"] == "reading_outside_declared_domain"
    assert reading["session_id"] == "test.router.normalization"
    assert reading["sequence"] == 2
    assert reading["ingested_at_ns"] > 0
    assert reading["ingested_monotonic_ns"] > 0


@pytest.mark.asyncio
async def test_historian_records_unattended_reading_without_forcing_attention(
    tmp_path,
) -> None:
    historian = RealityHistorian(tmp_path / "history.sqlite3")
    router = RealityObservationRouter(RealityReachService(), historian=historian)
    router.pause_attention()

    receipt = await router.submit(
        _declaration("test.unattended"),
        _reading("test.unattended", 42.0),
        adapter_id="test.unattended_adapter",
    )

    assert receipt.accepted is False
    assert receipt.reason == "not_subscribed"
    assert historian.status()["observation_count"] == 1
    assert historian.status()["delivery_counts"] == {}


@pytest.mark.asyncio
async def test_attention_pause_preserves_subscription_intent_and_queue_depth(
    tmp_path,
) -> None:
    historian = RealityHistorian(tmp_path / "history.sqlite3")
    router = RealityObservationRouter(RealityReachService(), historian=historian)
    router.configure_subscription(
        ObservationSubscription(
            subscription_id="test.intentionally_disabled",
            selector="test.disabled",
            enabled=False,
        )
    )
    first = await router.submit(
        _declaration("test.queued"),
        _reading("test.queued", 41.0),
        adapter_id="test.queued_adapter",
    )
    assert first.accepted is True
    assert first.queue_depth == 1

    router.pause_attention()
    focus = router.focus("test.paused_focus")
    paused = await router.submit(
        _declaration("test.paused_focus"),
        _reading("test.paused_focus", 42.0),
        adapter_id="test.paused_focus_adapter",
    )
    assert paused.reason == "not_subscribed"
    assert paused.queue_depth == 1
    assert router.status()["queue_depth"] == 1

    router.resume_attention()
    subscriptions = {
        item.subscription_id: item for item in router.subscriptions()
    }
    assert subscriptions["test.intentionally_disabled"].enabled is False
    assert subscriptions[focus.subscription_id].enabled is True


@pytest.mark.asyncio
async def test_durable_historian_claims_remain_bound_after_outer_digest_rewrite(
    tmp_path,
) -> None:
    path = tmp_path / "history.sqlite3"
    historian = RealityHistorian(path)
    router = RealityObservationRouter(RealityReachService(), historian=historian)
    receipt = await router.submit(
        _declaration("test.bound_evidence"),
        _reading("test.bound_evidence", 42.0),
        adapter_id="test.bound_evidence_adapter",
    )
    assert receipt.accepted is True

    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            "SELECT payload_json, sink_states_json FROM reality_deliveries "
            "WHERE observation_id=?",
            (receipt.observation_id,),
        ).fetchone()
        assert row is not None
        payload = json.loads(str(row[0]))
        sink_envelope = json.loads(str(row[1]))
        payload["historian"]["alarm_codes"] = ["source_order_gap"]
        binding_evidence = dict(payload["historian"])
        binding_evidence.pop("binding_sha256")
        base_payload = dict(payload)
        base_payload.pop("historian")
        payload["historian"]["binding_sha256"] = str(
            sha256_hex(
                canonical_json(
                    {
                        "observation": base_payload,
                        "historian": binding_evidence,
                    }
                )
            )
        )
        sink_envelope["payload_sha256"] = str(
            sha256_hex(canonical_json(payload))
        )
        connection.execute(
            "UPDATE reality_deliveries SET payload_json=?, sink_states_json=? "
            "WHERE observation_id=?",
            (
                canonical_json(payload).decode("utf-8"),
                canonical_json(sink_envelope).decode("utf-8"),
                receipt.observation_id,
            ),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(HistorianCorruptionError, match="differs from its record"):
        await historian.claim_delivery(receipt.observation_id)
    assert historian.status()["delivery_counts"] == {"quarantined": 1}


@pytest.mark.asyncio
async def test_acquisition_pause_stops_sampling_before_history(tmp_path) -> None:
    historian = RealityHistorian(tmp_path / "history.sqlite3")
    router = RealityObservationRouter(RealityReachService(), historian=historian)
    router.pause_acquisition()

    receipt = await router.submit(
        _declaration("test.acquisition_paused"),
        _reading("test.acquisition_paused", 42.0),
        adapter_id="test.acquisition_paused_adapter",
    )

    assert receipt.accepted is False
    assert receipt.reason == "acquisition_paused"
    assert historian.status()["observation_count"] == 0


@pytest.mark.asyncio
async def test_source_gap_bypasses_attention_rate_and_delta_filters(
    tmp_path,
) -> None:
    historian = RealityHistorian(tmp_path / "history.sqlite3")
    router = RealityObservationRouter(
        RealityReachService(),
        historian=historian,
    )
    declaration = _declaration("test.source_gap", sample_rate_hz=20.0)
    router.configure_subscription(
        ObservationSubscription(
            subscription_id="reality.strict",
            selector=declaration.channel_id,
            max_rate_hz=0.01,
            min_delta=100.0,
            min_salience=0.94,
        )
    )

    first_receipt = await router.submit(
        declaration,
        _reading(
            declaration.channel_id,
            20.0,
            captured_at_ns=1_000,
            source_epoch="sensor.boot.a",
            source_sequence=1,
        ),
        adapter_id="test.adapter",
    )
    gap_receipt = await router.submit(
        declaration,
        _reading(
            declaration.channel_id,
            20.0,
            captured_at_ns=2_000,
            source_epoch="sensor.boot.a",
            source_sequence=3,
        ),
        adapter_id="test.adapter",
    )

    assert first_receipt.accepted is False
    assert first_receipt.reason == "below_salience"
    assert gap_receipt.accepted is True
    assert gap_receipt.salience == 0.95
    assert (await historian.active_alarms())[0]["alarm_code"] == "source_order_gap"


@pytest.mark.asyncio
async def test_durable_outbox_restores_cognitive_delivery_after_router_restart(
    monkeypatch,
    tmp_path,
) -> None:
    events = []
    cognitive: list[tuple[tuple, dict]] = []
    services = {
        "multimodal_synchronizer": _accepting_synchronizer(events),
        "advanced_cognition": _accepting_cognition(cognitive),
    }
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: services.get(name, default)),
        raising=False,
    )
    path = tmp_path / "history.sqlite3"
    digital_twin = _digital_twin(tmp_path)
    first_historian = RealityHistorian(path)
    first_router = RealityObservationRouter(
        RealityReachService(),
        historian=first_historian,
        digital_twin=digital_twin,
    )
    queued = await first_router.submit(
        _declaration("test.restart_delivery"),
        _reading("test.restart_delivery", 55.0),
        adapter_id="test.restart_adapter",
    )
    assert queued.accepted is True
    assert first_historian.status()["delivery_counts"] == {"queued": 1}

    restored_historian = RealityHistorian(path)
    restored_router = RealityObservationRouter(
        RealityReachService(),
        historian=restored_historian,
        digital_twin=digital_twin,
        poll_interval_s=60.0,
        max_delivery_rate_hz=100.0,
    )
    await restored_router.start()
    try:
        for _ in range(100):
            if events and cognitive:
                break
            await asyncio.sleep(0.01)
    finally:
        await restored_router.stop()

    assert len(events) == 1
    assert events[0].event_id == queued.observation_id
    assert cognitive[0][0][1]["value"] == 55.0
    assert restored_historian.status()["delivery_counts"] == {"delivered": 1}


@pytest.mark.asyncio
async def test_failed_cognitive_delivery_retries_durably_after_restart(
    monkeypatch,
    tmp_path,
) -> None:
    clock = {"value": 100.0}
    path = tmp_path / "history.sqlite3"
    delivered: list[dict] = []
    events: list = []
    mode = {"fail": True}

    def _observe_state(_name, payload, **_kwargs):
        if mode["fail"]:
            raise RuntimeError("synthetic cognitive outage")
        delivered.append(payload)
        return {"receipt_id": "advanced.receipt.retry-proof"}

    services = {
        "multimodal_synchronizer": _accepting_synchronizer(events),
        "advanced_cognition": SimpleNamespace(
            observe_state=_observe_state,
            health_report=lambda: {"ok": True},
        ),
    }
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: services.get(name, default)),
        raising=False,
    )
    first_historian = RealityHistorian(path, clock=lambda: clock["value"])
    digital_twin = _digital_twin(tmp_path)
    first_router = RealityObservationRouter(
        RealityReachService(),
        historian=first_historian,
        digital_twin=digital_twin,
        poll_interval_s=60.0,
    )
    queued = await first_router.submit(
        _declaration("test.retry_delivery"),
        _reading("test.retry_delivery", 61.0),
        adapter_id="test.retry_adapter",
    )
    assert queued.accepted is True
    await first_router.start()
    try:
        for _ in range(100):
            if first_router.status()["delivery_failures"]:
                break
            await asyncio.sleep(0.01)
    finally:
        await first_router.stop()

    assert first_router.status()["delivery_failures"] == 1
    assert first_historian.status()["delivery_counts"] == {"queued": 1}

    mode["fail"] = False
    clock["value"] = 102.0
    restored_historian = RealityHistorian(path, clock=lambda: clock["value"])
    restored_router = RealityObservationRouter(
        RealityReachService(),
        historian=restored_historian,
        digital_twin=digital_twin,
        poll_interval_s=60.0,
    )
    await restored_router.start()
    try:
        for _ in range(100):
            if delivered:
                break
            await asyncio.sleep(0.01)
    finally:
        await restored_router.stop()

    assert delivered[0]["channel_id"] == "test.retry_delivery"
    assert restored_historian.status()["delivery_counts"] == {"delivered": 1}


@pytest.mark.asyncio
async def test_historian_quality_and_order_evidence_reach_both_cognitive_sinks(
    monkeypatch,
    tmp_path,
) -> None:
    events: list = []
    cognitive: list[tuple[tuple, dict]] = []
    services = {
        "multimodal_synchronizer": _accepting_synchronizer(events),
        "advanced_cognition": _accepting_cognition(cognitive),
    }
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: services.get(name, default)),
        raising=False,
    )
    historian = RealityHistorian(tmp_path / "history.sqlite3")
    digital_twin = _digital_twin(tmp_path)
    router = RealityObservationRouter(
        RealityReachService(),
        historian=historian,
        digital_twin=digital_twin,
        poll_interval_s=60.0,
        max_delivery_rate_hz=100.0,
    )
    await router.start()
    try:
        receipt = await router.submit(
            _declaration("test.bad_quality"),
            _reading(
                "test.bad_quality",
                44.0,
                source_quality="bad",
            ),
            adapter_id="test.bad_quality_adapter",
        )
        assert receipt.accepted is True
        for _ in range(100):
            if events and cognitive:
                break
            await asyncio.sleep(0.01)
    finally:
        await router.stop()

    assert events[0].confidence == 0.0
    assert "historian_quality:bad" in events[0].quality_flags
    assert "historian_alarm:source_quality_bad" in events[0].quality_flags
    cognitive_historian = cognitive[0][0][1]["historian"]
    assert cognitive_historian["quality"] == "bad"
    assert cognitive_historian["alarm_codes"] == ["source_quality_bad"]


@pytest.mark.asyncio
async def test_rejected_sink_receipt_prevents_false_delivery_completion(
    monkeypatch,
    tmp_path,
) -> None:
    advanced_calls: list[tuple[tuple, dict]] = []

    def _reject(event):
        return SimpleNamespace(
            event_id=event.event_id,
            accepted=False,
            reason="synthetic_backpressure",
        )

    services = {
        "multimodal_synchronizer": SimpleNamespace(
            ingest=_reject,
            get_status=lambda: {
                "accepted_events": 0,
                "rejected_events": 1,
                "queue_overflow_drops": 0,
                "late_events": 0,
                "fusions": 0,
                "queue_depths": {"proprioception": 0},
                "queue_limit": 64,
            },
        ),
        "advanced_cognition": _accepting_cognition(advanced_calls),
    }
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: services.get(name, default)),
        raising=False,
    )
    historian = RealityHistorian(tmp_path / "history.sqlite3")
    digital_twin = _digital_twin(tmp_path)
    router = RealityObservationRouter(
        RealityReachService(),
        historian=historian,
        digital_twin=digital_twin,
        poll_interval_s=60.0,
    )
    await router.start()
    try:
        queued = await router.submit(
            _declaration("test.rejected_sink"),
            _reading("test.rejected_sink", 10.0),
            adapter_id="test.rejected_sink_adapter",
        )
        assert queued.accepted is True
        for _ in range(100):
            if router.status()["delivery_failures"]:
                break
            await asyncio.sleep(0.01)
    finally:
        await router.stop()

    assert advanced_calls == []
    assert historian.status()["delivery_counts"] == {"queued": 1}


@pytest.mark.asyncio
async def test_historian_database_failure_is_supervised_and_recovers(
    monkeypatch,
    tmp_path,
) -> None:
    events: list = []
    cognitive: list[tuple[tuple, dict]] = []
    services = {
        "multimodal_synchronizer": _accepting_synchronizer(events),
        "advanced_cognition": _accepting_cognition(cognitive),
    }
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: services.get(name, default)),
        raising=False,
    )
    historian = RealityHistorian(tmp_path / "history.sqlite3")
    digital_twin = _digital_twin(tmp_path)
    router = RealityObservationRouter(
        RealityReachService(),
        historian=historian,
        digital_twin=digital_twin,
        poll_interval_s=60.0,
    )
    queued = await router.submit(
        _declaration("test.database_recovery"),
        _reading("test.database_recovery", 33.0),
        adapter_id="test.database_recovery_adapter",
    )
    assert queued.accepted is True
    original_connect = historian._connect

    def _broken_connect():
        import sqlite3

        raise sqlite3.OperationalError("synthetic database outage")

    monkeypatch.setattr(historian, "_connect", _broken_connect)
    await router.start()
    try:
        for _ in range(100):
            if router.status()["historian_failures"]:
                break
            await asyncio.sleep(0.01)
        assert router.is_alive() is True
        assert router.is_ready() is False
        assert historian.health_snapshot()["ready"] is False

        monkeypatch.setattr(historian, "_connect", original_connect)
        for _ in range(200):
            if cognitive:
                break
            await asyncio.sleep(0.01)
    finally:
        await router.stop()

    assert len(events) == 1
    assert len(cognitive) == 1
    assert historian.status()["delivery_counts"] == {"delivered": 1}


@pytest.mark.asyncio
async def test_twin_delivery_fence_is_anchored_in_authoritative_observation_columns(
    tmp_path,
) -> None:
    path = tmp_path / "history.sqlite3"
    historian = RealityHistorian(path)
    digital_twin = _digital_twin(tmp_path)
    router = RealityObservationRouter(
        RealityReachService(),
        historian=historian,
        digital_twin=digital_twin,
    )
    receipt = await router.submit(
        _declaration("test.authoritative_twin_fence"),
        _reading("test.authoritative_twin_fence", 42.0),
        adapter_id="test.authoritative_twin_adapter",
    )
    assert receipt.accepted is True

    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            "SELECT d.payload_json, d.sink_states_json, o.twin_id, "
            "o.attachment_generation, o.attachment_bound_at_ns, o.topology_revision "
            "FROM reality_deliveries AS d JOIN reality_observations AS o "
            "ON o.record_id=d.record_id WHERE d.observation_id=?",
            (receipt.observation_id,),
        ).fetchone()
        assert row is not None
        payload = json.loads(str(row[0]))
        sink_envelope = json.loads(str(row[1]))
        assert row[2] == payload["twin_binding"]["twin_id"]
        assert row[3] == payload["twin_binding"]["attachment_generation"]
        assert row[4] == payload["twin_binding"]["attachment_bound_at_ns"]
        assert row[5] == payload["twin_binding"]["topology_revision"]

        payload["twin_binding"]["attachment_generation"] += 1
        twin_evidence = dict(payload["twin_binding"])
        twin_evidence.pop("binding_sha256")
        payload["twin_binding"]["binding_sha256"] = str(
            sha256_hex(
                canonical_json(
                    {
                        "observation_id": receipt.observation_id,
                        "binding": twin_evidence,
                    }
                )
            )
        )
        base_payload = dict(payload)
        base_payload.pop("historian")
        historian_evidence = dict(payload["historian"])
        historian_evidence.pop("binding_sha256")
        payload["historian"]["binding_sha256"] = str(
            sha256_hex(
                canonical_json(
                    {
                        "observation": base_payload,
                        "historian": historian_evidence,
                    }
                )
            )
        )
        sink_envelope["payload_sha256"] = str(
            sha256_hex(canonical_json(payload))
        )
        connection.execute(
            "UPDATE reality_deliveries SET payload_json=?, sink_states_json=? "
            "WHERE observation_id=?",
            (
                canonical_json(payload).decode("utf-8"),
                canonical_json(sink_envelope).decode("utf-8"),
                receipt.observation_id,
            ),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(HistorianCorruptionError, match="authoritative record"):
        await historian.claim_delivery(receipt.observation_id)
    assert historian.status()["delivery_counts"] == {"quarantined": 1}


@pytest.mark.asyncio
async def test_required_sink_registry_keeps_boot_degradation_unready_and_durable(
    monkeypatch,
    tmp_path,
) -> None:
    services = {
        "advanced_cognition": _accepting_cognition([]),
    }
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: services.get(name, default)),
        raising=False,
    )
    historian = RealityHistorian(tmp_path / "history.sqlite3")
    router = RealityObservationRouter(
        RealityReachService(),
        historian=historian,
        poll_interval_s=60.0,
    )
    await router.start()
    try:
        status = router.status()
        assert router.is_alive() is True
        assert router.is_ready() is False
        assert status["required_sinks"] == [
            "digital_twin",
            "multimodal",
            "advanced_cognition",
        ]
        assert status["required_sink_registry"]["digital_twin"] == {
            "dependency_key": "reality_digital_twin",
            "callable_attribute": "observe_observation",
            "health_attribute": "health_snapshot",
            "health_contract": "ready_flag",
            "configured": False,
            "ready": False,
            "reason": "dependency_unavailable",
        }
        assert status["required_sink_registry"]["multimodal"]["ready"] is False
        receipt = await router.submit(
            _declaration("test.boot_degraded"),
            _reading("test.boot_degraded", 37.0),
            adapter_id="test.boot_degraded_adapter",
        )
        assert receipt.accepted is True
        await asyncio.sleep(0.02)
    finally:
        await router.stop()

    sink_status = historian.status()["delivery_sink_status"]
    assert sink_status["digital_twin"]["pending"] == 1
    assert sink_status["multimodal"]["pending"] == 1
    assert sink_status["advanced_cognition"]["pending"] == 1


@pytest.mark.asyncio
async def test_legacy_head_backfill_is_receipted_idempotent_and_restart_safe(
    monkeypatch,
    tmp_path,
) -> None:
    services = {
        "multimodal_synchronizer": _accepting_synchronizer([]),
        "advanced_cognition": _accepting_cognition([]),
    }
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: services.get(name, default)),
        raising=False,
    )
    history_path = tmp_path / "history.sqlite3"
    legacy = RealityHistorian(history_path)
    declaration = _declaration("test.legacy_head")
    reading = _reading("test.legacy_head", 63.0, captured_at_ns=12_345)
    physical_identity = "sha256:" + "a" * 64
    source_service = RealityReachService()
    source_service.register_adapter(
        AsyncAdapter(
            "test.legacy_head_adapter",
            declaration,
            value=63.0,
            physical_identity_sha256=physical_identity,
        )
    )
    reading = source_service.ingest_sensor_readings(
        "test.legacy_head_adapter",
        (reading,),
    )[declaration.channel_id]
    admitted = await legacy.admit(
        declaration,
        reading,
        adapter_id="test.legacy_head_adapter",
    )
    assert admitted.accepted is True

    connection = sqlite3.connect(history_path)
    try:
        connection.execute("DROP TABLE reality_backfill_receipts")
        for column in (
            "topology_revision",
            "attachment_bound_at_ns",
            "attachment_generation",
            "twin_id",
        ):
            connection.execute(
                f"ALTER TABLE reality_observations DROP COLUMN {column}"
            )
        connection.execute(
            "UPDATE reality_historian_meta SET value='1' WHERE key='schema_version'"
        )
        connection.commit()
    finally:
        connection.close()

    historian = RealityHistorian(history_path)
    assert historian.status()["schema_version"] == 2
    digital_twin = _digital_twin(tmp_path)
    live_service = RealityReachService()
    live_service.register_adapter(
        AsyncAdapter(
            "test.legacy_head_adapter",
            declaration,
            value=63.0,
            physical_identity_sha256=physical_identity,
        )
    )
    original_record_receipt = historian.record_backfill_receipt

    async def _fail_receipt_once(**_kwargs):
        raise HistorianCorruptionError("synthetic receipt interruption")

    monkeypatch.setattr(historian, "record_backfill_receipt", _fail_receipt_once)
    interrupted = RealityObservationRouter(
        live_service,
        historian=historian,
        digital_twin=digital_twin,
        poll_interval_s=60.0,
    )
    await interrupted.start()
    try:
        assert interrupted.status()["twin_backfill_ready"] is False
        assert interrupted.status()["twin_backfill_failures"] == 1
    finally:
        await interrupted.stop()
    assert historian.status()["backfill_receipt_count"] == 0

    monkeypatch.setattr(
        historian,
        "record_backfill_receipt",
        original_record_receipt,
    )
    resumed = RealityObservationRouter(
        live_service,
        historian=historian,
        digital_twin=digital_twin,
        poll_interval_s=60.0,
    )
    await resumed.start()
    try:
        assert resumed.status()["twin_backfill_ready"] is True
        assert resumed.status()["twin_backfill_receipts"] == 1
    finally:
        await resumed.stop()
    assert historian.status()["backfill_receipt_count"] == 1

    restarted_historian = RealityHistorian(history_path)
    restarted = RealityObservationRouter(
        live_service,
        historian=restarted_historian,
        digital_twin=digital_twin,
        poll_interval_s=60.0,
    )
    await restarted.start()
    try:
        assert restarted.status()["twin_backfill_ready"] is True
        assert restarted.status()["twin_backfill_receipts"] == 0
    finally:
        await restarted.stop()
    assert restarted_historian.status()["backfill_receipt_count"] == 1
    public_snapshot = digital_twin.snapshot()
    assert public_snapshot["properties"] == []
    twin_connection = sqlite3.connect(digital_twin.db_path)
    try:
        projected = twin_connection.execute(
            "SELECT historian_record_id, captured_at_ns, value_json "
            "FROM twin_properties"
        ).fetchone()
    finally:
        twin_connection.close()
    assert projected == (admitted.record_id, 12_345, "63.0")


@pytest.mark.asyncio
async def test_legacy_head_backfill_pages_beyond_prior_4096_ceiling(
    monkeypatch,
    tmp_path,
) -> None:
    total = 4_100
    channel_ids = tuple(f"test.page.{index:04d}" for index in range(total))
    snapshots = tuple(
        HistorianHeadSnapshot(
            record_id=f"record.page.{index:04d}",
            adapter_id="test.paged_adapter",
            adapter_identity_sha256="sha256:" + "5" * 64,
            adapter_registration_generation=7,
            adapter_identity_stable=False,
            channel_id=channel_id,
            declaration={},
            reading={},
            quality="measured",
            order_basis="captured_at_ns",
            order_gap=False,
            alarm_codes=(),
            recorded_at=1.0,
            source_sha256="sha256:" + "1" * 64,
        )
        for index, channel_id in enumerate(channel_ids)
    )
    recorded: set[str] = set()
    page_calls = 0
    historian = RealityHistorian(tmp_path / "paged-history.sqlite3")

    async def _page(*, after_channel_id="", limit=512, **_kwargs):
        nonlocal page_calls
        page_calls += 1
        remaining = tuple(item for item in snapshots if item.record_id not in recorded)
        if after_channel_id:
            remaining = tuple(
                item for item in remaining if item.channel_id > after_channel_id
            )
        selected = remaining[:limit]
        return HistorianHeadPage(
            snapshots=selected,
            next_channel_id=(selected[-1].channel_id if selected else after_channel_id),
            exhausted=len(remaining) <= limit,
            scanned=len(selected),
        )

    async def _record_receipt(**kwargs):
        recorded.add(str(kwargs["record_id"]))
        return "receipt.backfill"

    monkeypatch.setattr(historian, "legacy_twin_head_page", _page)
    monkeypatch.setattr(historian, "record_backfill_receipt", _record_receipt)
    service = RealityReachService()
    monkeypatch.setattr(
        service,
        "adapter_inventory",
        lambda: {
            "test.paged_adapter": AdapterInventoryEntry(
                adapter_id="test.paged_adapter",
                channel_ids=channel_ids,
                identity_sha256="sha256:" + "5" * 64,
                registration_generation=7,
                stable_identity=False,
            )
        },
    )
    digital_twin = _digital_twin(tmp_path, name="paged-twin.sqlite3")
    monkeypatch.setattr(digital_twin, "is_ready", lambda: True)
    monkeypatch.setattr(
        digital_twin,
        "binding_context",
        lambda *_args: {
            "twin_id": "twin.virtual",
            "attachment_generation": 1,
            "attachment_bound_at_ns": 1,
            "topology_revision": 1,
            "binding_sha256": "sha256:" + "2" * 64,
        },
    )
    monkeypatch.setattr(
        RealityObservationRouter,
        "_declaration_from_snapshot",
        staticmethod(lambda snapshot: _declaration(snapshot.channel_id)),
    )
    monkeypatch.setattr(
        RealityObservationRouter,
        "_backfill_observation",
        staticmethod(
            lambda snapshot, _binding: (
                SimpleNamespace(
                    observation_id=snapshot.record_id,
                    twin_id="twin.virtual",
                ),
                "sha256:" + "3" * 64,
            )
        ),
    )

    def _observe(observation):
        return TwinReceipt(
            receipt_id=f"receipt.{observation.observation_id}",
            twin_id=observation.twin_id,
            event_id=f"event.{observation.observation_id}",
            disposition=TwinDisposition.ACCEPTED,
            accepted=True,
            graph_version=1,
            state_sha256="sha256:" + "4" * 64,
        )

    monkeypatch.setattr(digital_twin, "observe_observation", _observe)
    router = RealityObservationRouter(
        service,
        historian=historian,
        digital_twin=digital_twin,
    )

    assert await router._backfill_legacy_twin_heads() == total
    assert len(recorded) == total
    assert page_calls == 10
    assert router.status()["twin_backfill_ready"] is True


@pytest.mark.asyncio
async def test_legacy_backfill_never_binds_retired_adapter(
    monkeypatch,
    tmp_path,
) -> None:
    historian = RealityHistorian(tmp_path / "retired-history.sqlite3")
    declaration = _declaration("test.retired")
    admitted = await historian.admit(
        declaration,
        _reading("test.retired", 9.0),
        adapter_id="test.retired_adapter",
    )
    assert admitted.accepted is True
    digital_twin = _digital_twin(tmp_path, name="retired-twin.sqlite3")
    binding_calls = 0

    def _unexpected_binding(*_args):
        nonlocal binding_calls
        binding_calls += 1
        raise AssertionError("retired adapter reached twin binding")

    monkeypatch.setattr(digital_twin, "binding_context", _unexpected_binding)
    router = RealityObservationRouter(
        RealityReachService(),
        historian=historian,
        digital_twin=digital_twin,
    )

    assert await router._backfill_legacy_twin_heads() == 0
    assert binding_calls == 0
    assert router.status()["twin_backfill_ready"] is True


@pytest.mark.asyncio
async def test_legacy_backfill_rejects_adapter_id_reused_by_different_device(
    monkeypatch,
    tmp_path,
) -> None:
    historian = RealityHistorian(tmp_path / "identity-reuse-history.sqlite3")
    declaration = _declaration("test.identity_reuse")
    device_a = RealityReachService()
    device_a.register_adapter(
        AsyncAdapter(
            "test.reused_adapter",
            declaration,
            value=17.0,
            physical_identity_sha256="sha256:" + "a" * 64,
        )
    )
    reading_a = device_a.ingest_sensor_readings(
        "test.reused_adapter",
        (_reading(declaration.channel_id, 17.0),),
    )[declaration.channel_id]
    admitted = await historian.admit(
        declaration,
        reading_a,
        adapter_id="test.reused_adapter",
    )
    assert admitted.accepted is True

    replacement_service = RealityReachService()
    replacement_service.register_adapter(
        AsyncAdapter(
            "test.reused_adapter",
            declaration,
            value=29.0,
            physical_identity_sha256="sha256:" + "b" * 64,
        )
    )
    binding_calls = 0

    def _unexpected_binding(*_args):
        nonlocal binding_calls
        binding_calls += 1
        raise AssertionError("device A history reached device B twin binding")

    class _BackfillTwin(RealityDigitalTwinGraph):
        def __init__(self) -> None:
            pass

        def is_ready(self) -> bool:
            return True

        def binding_context(self, *_args):
            return _unexpected_binding(*_args)

    digital_twin = _BackfillTwin()
    router = RealityObservationRouter(
        replacement_service,
        historian=historian,
        digital_twin=digital_twin,
    )

    assert await router._backfill_legacy_twin_heads() == 0
    assert binding_calls == 0
    assert router._twin_backfill_ready is True


def test_required_sink_health_rejects_callable_but_unhealthy_dependencies(
    monkeypatch,
    tmp_path,
) -> None:
    multimodal = _accepting_synchronizer([])
    multimodal.get_status = lambda: {
        "accepted_events": 0,
        "rejected_events": 0,
        "queue_overflow_drops": 0,
        "late_events": 0,
        "fusions": 0,
        "queue_depths": {"device": 65},
        "queue_limit": 64,
    }
    advanced = _accepting_cognition([])
    advanced.health_report = lambda: {"ok": False, "reason": "degraded"}
    services = {
        "multimodal_synchronizer": multimodal,
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
        digital_twin=_digital_twin(tmp_path, name="sink-health-twin.sqlite3"),
    )

    status = router.status()["required_sink_registry"]
    assert status["digital_twin"]["ready"] is True
    assert status["multimodal"]["configured"] is True
    assert status["multimodal"]["ready"] is False
    assert status["multimodal"]["reason"] == "health_report_unready"
    assert status["advanced_cognition"]["configured"] is True
    assert status["advanced_cognition"]["ready"] is False
    assert status["advanced_cognition"]["reason"] == "health_report_unready"
    assert router.is_ready() is False


@pytest.mark.asyncio
async def test_observation_durable_codec_round_trips_and_rejects_tampering() -> None:
    router = RealityObservationRouter(RealityReachService())
    receipt = await router.submit(
        _declaration("test.codec"),
        _reading("test.codec", 12.0),
        adapter_id="test.codec_adapter",
    )
    payload = router.latest()["test.codec"]

    restored = RealityObservation.from_dict(payload)
    assert restored.observation_id == receipt.observation_id
    tampered = {**payload, "reading_sha256": "sha256:" + "0" * 64}
    with pytest.raises(ValueError, match="reading digest differs"):
        RealityObservation.from_dict(tampered)
    boolean_clock = {
        **payload,
        "reading": {**payload["reading"], "captured_at_ns": True},
    }
    with pytest.raises(TypeError, match="captured_at_ns must be an integer"):
        RealityObservation.from_dict(boolean_clock)
    boolean_receipt = {**payload, "received_monotonic_ns": True}
    with pytest.raises(TypeError, match="received_monotonic_ns must be an integer"):
        RealityObservation.from_dict(boolean_receipt)
    forged_historian = {
        **payload,
        "historian": {
            **payload["historian"],
            "quality": "bad",
            "reason": "accepted",
        },
    }
    with pytest.raises(ValueError, match="historian record binding"):
        RealityObservation.from_dict(forged_historian)
