from __future__ import annotations

import asyncio
import os
import sqlite3
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from core.reality_reach.contracts import (
    ChannelDeclaration,
    ChannelKind,
    CouplingClass,
    EvidenceLevel,
    NumericDomain,
    RealityLayer,
)
from core.reality_reach.historian import (
    HistorianCorruptionError,
    HistorianDisposition,
    ObservationQuality,
    RealityHistorian,
)
from core.reality_reach.live import ChannelReading, ReadingStatus


def _declaration(*, resolution: float = 1.0) -> ChannelDeclaration:
    return ChannelDeclaration(
        channel_id="test.room.temperature",
        kind=ChannelKind.SENSOR,
        observable="temperature",
        unit="celsius",
        domain=NumericDomain(-40.0, 125.0),
        coupling=CouplingClass.NETWORK,
        reality_layers=(RealityLayer.EFFECTIVE,),
        evidence_level=EvidenceLevel.P2,
        owner="tests.reality_historian",
        resolution=resolution,
        sample_rate_hz=1.0,
        max_latency_s=1.0,
        stale_after_s=30.0,
        reference_id="test.room.thermometer",
        coupling_validated=True,
    )


def _reading(
    value: float | None,
    *,
    sequence: int,
    captured_at_ns: int,
    status: ReadingStatus = ReadingStatus.AVAILABLE,
    session_id: str = "session-a",
    source_epoch: str = "",
    source_sequence: int = 0,
    source_event_id: str = "",
    source_quality: str = "",
) -> ChannelReading:
    return ChannelReading(
        channel_id="test.room.temperature",
        value=value,
        unit="celsius",
        captured_at_ns=captured_at_ns,
        status=status,
        source="test.thermometer",
        uncertainty=0.2,
        ingested_at_ns=captured_at_ns + 10,
        ingested_monotonic_ns=captured_at_ns + 20,
        session_id=session_id,
        sequence=sequence,
        source_epoch=source_epoch,
        source_sequence=source_sequence,
        source_event_id=source_event_id,
        source_quality=source_quality,
    )


@pytest.mark.asyncio
async def test_historian_enforces_source_order_and_source_deadband(tmp_path: Path) -> None:
    historian = RealityHistorian(tmp_path / "history.sqlite3", clock=lambda: 100.0)
    declaration = _declaration(resolution=1.0)

    first = await historian.admit(
        declaration,
        _reading(20.0, sequence=1, captured_at_ns=1_000),
        adapter_id="test.adapter",
    )
    deadband = await historian.admit(
        declaration,
        _reading(20.5, sequence=2, captured_at_ns=2_000),
        adapter_id="test.adapter",
    )
    regressed = await historian.admit(
        declaration,
        _reading(22.0, sequence=1, captured_at_ns=3_000),
        adapter_id="test.adapter",
    )
    conflict = await historian.admit(
        declaration,
        _reading(23.0, sequence=2, captured_at_ns=2_000),
        adapter_id="test.adapter",
    )

    assert first.disposition == HistorianDisposition.ACCEPTED
    assert first.quality == ObservationQuality.GOOD
    assert deadband.disposition == HistorianDisposition.DEADBAND
    assert deadband.record_id == first.record_id
    assert regressed.reason == "ingest_sequence_regressed"
    assert conflict.reason == "ingest_sequence_conflict"
    replay = await historian.replay_history(limit=50)
    assert replay["count"] == 1
    assert replay["records"][0]["value"] == 20.0
    quarantined = await historian.quarantine(limit=50)
    assert {item["reason"] for item in quarantined} == {
        "ingest_sequence_conflict",
        "ingest_sequence_regressed",
    }


@pytest.mark.asyncio
async def test_new_source_session_establishes_new_order_lineage(tmp_path: Path) -> None:
    historian = RealityHistorian(tmp_path / "history.sqlite3")
    declaration = _declaration()
    await historian.admit(
        declaration,
        _reading(20.0, sequence=9, captured_at_ns=1_000),
        adapter_id="test.adapter",
    )

    restarted = await historian.admit(
        declaration,
        _reading(
            22.0,
            sequence=1,
            captured_at_ns=2_000,
            session_id="session-b",
        ),
        adapter_id="test.adapter",
    )

    assert restarted.accepted is True
    assert (await historian.replay_history(limit=10))["count"] == 2


@pytest.mark.asyncio
async def test_alarm_journal_activates_acknowledges_changes_and_clears(
    tmp_path: Path,
) -> None:
    clock = {"value": 100.0}
    historian = RealityHistorian(
        tmp_path / "history.sqlite3",
        clock=lambda: clock["value"],
    )
    declaration = _declaration()
    await historian.admit(
        declaration,
        _reading(20.0, sequence=1, captured_at_ns=1_000),
        adapter_id="test.adapter",
    )
    clock["value"] += 1.0
    unavailable = await historian.admit(
        declaration,
        _reading(
            None,
            sequence=2,
            captured_at_ns=2_000,
            status=ReadingStatus.UNAVAILABLE,
        ),
        adapter_id="test.adapter",
    )

    alarms = await historian.active_alarms()
    assert unavailable.alarm_event_ids
    assert alarms[0]["alarm_code"] == "reading_unavailable"
    assert alarms[0]["severity"] == "high"
    assert alarms[0]["acknowledged"] is False

    acknowledged = await historian.acknowledge_alarm(
        "test.room.temperature",
        actor="aura",
    )
    assert acknowledged["event_id"].startswith("reality.alarm.")
    assert (await historian.active_alarms())[0]["acknowledged"] is True

    clock["value"] += 1.0
    changed = await historian.admit(
        declaration,
        _reading(
            20.0,
            sequence=3,
            captured_at_ns=3_000,
            status=ReadingStatus.DEGRADED,
        ),
        adapter_id="test.adapter",
    )
    assert len(changed.alarm_event_ids) == 2
    assert (await historian.active_alarms())[0]["alarm_code"] == "reading_degraded"

    clock["value"] += 1.0
    cleared = await historian.admit(
        declaration,
        _reading(21.5, sequence=4, captured_at_ns=4_000),
        adapter_id="test.adapter",
    )
    assert cleared.alarm_event_ids
    assert await historian.active_alarms() == ()


@pytest.mark.asyncio
async def test_store_and_forward_recovers_inflight_delivery_after_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.sqlite3"
    clock = {"value": 100.0}
    historian = RealityHistorian(path, clock=lambda: clock["value"])
    admitted = await historian.admit(
        _declaration(),
        _reading(20.0, sequence=1, captured_at_ns=1_000),
        adapter_id="test.adapter",
    )
    await historian.enqueue_delivery(
        observation_id="reality.obs.delivery-proof",
        record_id=admitted.record_id,
        payload={"observation_id": "reality.obs.delivery-proof", "value": 20.0},
    )
    claimed = await historian.claim_delivery("reality.obs.delivery-proof")
    assert claimed is not None
    assert claimed.attempts == 1

    clock["value"] += 31.0
    recovered = RealityHistorian(path, clock=lambda: clock["value"])
    due = await recovered.claim_due_deliveries(limit=10)
    assert len(due) == 1
    assert due[0].observation_id == "reality.obs.delivery-proof"
    assert due[0].attempts == 2
    await recovered.mark_delivered(
        due[0].observation_id,
        lease_token=due[0].lease_token,
    )

    status = recovered.status()
    assert status["delivery_counts"] == {"delivered": 1}
    assert status["recovered_inflight_total"] == 1
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_delivery_retry_exhaustion_is_terminal_and_visible(tmp_path: Path) -> None:
    clock = {"value": 100.0}
    historian = RealityHistorian(
        tmp_path / "history.sqlite3",
        clock=lambda: clock["value"],
        max_delivery_attempts=2,
    )
    admitted = await historian.admit(
        _declaration(),
        _reading(20.0, sequence=1, captured_at_ns=1_000),
        adapter_id="test.adapter",
    )
    await historian.enqueue_delivery(
        observation_id="reality.obs.retry-proof",
        record_id=admitted.record_id,
        payload={"observation_id": "reality.obs.retry-proof"},
    )

    first_claim = await historian.claim_delivery("reality.obs.retry-proof")
    assert first_claim is not None
    assert (
        await historian.mark_delivery_failed(
            "reality.obs.retry-proof",
            error_code="synthetic_failure",
            lease_token=first_claim.lease_token,
        )
        == "queued"
    )
    clock["value"] += 2.0
    second_claim = await historian.claim_delivery("reality.obs.retry-proof")
    assert second_claim is not None
    assert (
        await historian.mark_delivery_failed(
            "reality.obs.retry-proof",
            error_code="synthetic_failure",
            lease_token=second_claim.lease_token,
        )
        == "quarantined"
    )
    assert historian.status()["delivery_counts"] == {"quarantined": 1}


@pytest.mark.asyncio
async def test_malformed_delivery_is_quarantined_before_a_lease_can_strand_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.sqlite3"
    historian = RealityHistorian(path, max_delivery_attempts=2)
    admitted = await historian.admit(
        _declaration(),
        _reading(20.0, sequence=1, captured_at_ns=1_000),
        adapter_id="test.adapter",
    )
    await historian.enqueue_delivery(
        observation_id="reality.obs.corrupt-proof",
        record_id=admitted.record_id,
        payload={"observation_id": "reality.obs.corrupt-proof"},
    )
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "UPDATE reality_deliveries SET payload_json=? WHERE observation_id=?",
            ("{invalid-json", "reality.obs.corrupt-proof"),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(HistorianCorruptionError, match="payload is invalid"):
        await historian.claim_delivery("reality.obs.corrupt-proof")

    status = historian.status()
    assert status["delivery_counts"] == {"quarantined": 1}
    quarantined = await historian.quarantine()
    assert quarantined[0]["reason"] == "corrupt_delivery_payload"
    assert quarantined[0]["attempts"] == 1


@pytest.mark.asyncio
async def test_history_retention_and_replay_are_bounded(tmp_path: Path) -> None:
    historian = RealityHistorian(
        tmp_path / "history.sqlite3",
        max_records=8,
    )
    declaration = _declaration(resolution=0.0)
    for sequence in range(1, 13):
        admission = await historian.admit(
            declaration,
            _reading(float(sequence), sequence=sequence, captured_at_ns=sequence * 1_000),
            adapter_id="test.adapter",
        )
        assert admission.accepted is True

    status = historian.status()
    assert status["observation_count"] == 8
    first_page = await historian.replay_history(limit=3)
    second_page = await historian.replay_history(
        before_row_id=first_page["next_before_row_id"],
        limit=3,
    )
    assert first_page["count"] == 3
    assert second_page["count"] == 3
    assert set(item["record_id"] for item in first_page["records"]).isdisjoint(
        item["record_id"] for item in second_page["records"]
    )
    assert max(item["value"] for item in first_page["records"]) == 12.0


@pytest.mark.asyncio
async def test_delivery_payload_is_bounded_and_source_is_not_executed(tmp_path: Path) -> None:
    historian = RealityHistorian(tmp_path / "history.sqlite3")
    admitted = await historian.admit(
        _declaration(),
        _reading(20.0, sequence=1, captured_at_ns=1_000),
        adapter_id="test.adapter",
    )

    with pytest.raises(ValueError, match="bounded contract"):
        await historian.enqueue_delivery(
            observation_id="reality.obs.oversized",
            record_id=admitted.record_id,
            payload={"untrusted": "x" * (129 * 1024)},
        )


def test_existing_foreign_database_and_unsafe_path_are_refused(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign.sqlite3"
    connection = sqlite3.connect(foreign)
    try:
        connection.execute("CREATE TABLE unrelated(value TEXT)")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(HistorianCorruptionError, match="schema identity"):
        RealityHistorian(foreign)

    if hasattr(os, "symlink"):
        target = tmp_path / "target.sqlite3"
        link = tmp_path / "link.sqlite3"
        target.touch(mode=0o600)
        link.symlink_to(target)
        with pytest.raises(HistorianCorruptionError, match="symlink"):
            RealityHistorian(link)


def test_schema_and_monotonic_counter_tampering_are_refused_on_reopen(
    tmp_path: Path,
) -> None:
    schema_path = tmp_path / "schema.sqlite3"
    RealityHistorian(schema_path)
    connection = sqlite3.connect(schema_path)
    try:
        connection.execute(
            "ALTER TABLE reality_observations ADD COLUMN injected_payload TEXT"
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(HistorianCorruptionError, match="table contract differs"):
        RealityHistorian(schema_path)

    counter_path = tmp_path / "counter.sqlite3"
    RealityHistorian(counter_path)
    connection = sqlite3.connect(counter_path)
    try:
        connection.execute(
            "UPDATE reality_historian_meta SET value='rewound' "
            "WHERE key='observations_pruned_total'"
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(HistorianCorruptionError, match="counter is invalid"):
        RealityHistorian(counter_path)

    type_path = tmp_path / "column-type.sqlite3"
    RealityHistorian(type_path)
    connection = sqlite3.connect(type_path)
    try:
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='reality_observations'"
        ).fetchone()
        assert row is not None
        forged_sql = str(row[0]).replace(
            "captured_at_ns INTEGER NOT NULL",
            "captured_at_ns TEXT NOT NULL",
            1,
        )
        connection.execute("PRAGMA writable_schema=ON")
        connection.execute(
            "UPDATE sqlite_master SET sql=? WHERE type='table' "
            "AND name='reality_observations'",
            (forged_sql,),
        )
        schema_version = int(
            connection.execute("PRAGMA schema_version").fetchone()[0]
        )
        connection.execute(f"PRAGMA schema_version={schema_version + 1}")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(HistorianCorruptionError, match="table contract differs"):
        RealityHistorian(type_path)


@pytest.mark.asyncio
async def test_source_sequence_gap_is_uncertain_alarm_and_next_event_clears(
    tmp_path: Path,
) -> None:
    historian = RealityHistorian(tmp_path / "history.sqlite3")
    declaration = _declaration(resolution=0.0)
    await historian.admit(
        declaration,
        _reading(
            20.0,
            sequence=1,
            captured_at_ns=1_000,
            source_epoch="source.boot.a",
            source_sequence=1,
        ),
        adapter_id="test.adapter",
    )
    gap = await historian.admit(
        declaration,
        _reading(
            22.0,
            sequence=2,
            captured_at_ns=2_000,
            source_epoch="source.boot.a",
            source_sequence=3,
        ),
        adapter_id="test.adapter",
    )

    assert gap.reason == "accepted_with_source_gap"
    assert gap.quality == ObservationQuality.UNCERTAIN
    assert (await historian.active_alarms())[0]["alarm_code"] == "source_order_gap"

    restored = await historian.admit(
        declaration,
        _reading(
            23.0,
            sequence=3,
            captured_at_ns=3_000,
            source_epoch="source.boot.a",
            source_sequence=4,
        ),
        adapter_id="test.adapter",
    )
    assert restored.quality == ObservationQuality.GOOD
    assert await historian.active_alarms() == ()
    assert [event["state"] for event in await historian.alarm_history(limit=3)] == [
        "cleared",
        "active",
    ]


@pytest.mark.asyncio
async def test_equal_source_timestamps_preserve_distinct_events_as_uncertain(
    tmp_path: Path,
) -> None:
    historian = RealityHistorian(tmp_path / "history.sqlite3")
    declaration = _declaration(resolution=0.0)
    first = await historian.admit(
        declaration,
        _reading(
            20.0,
            sequence=1,
            captured_at_ns=1_000,
            source_epoch="source.boot.a",
            source_event_id="source.event.1",
        ),
        adapter_id="test.adapter",
    )
    tied = await historian.admit(
        declaration,
        _reading(
            21.0,
            sequence=2,
            captured_at_ns=1_000,
            source_epoch="source.boot.a",
            source_event_id="source.event.2",
        ),
        adapter_id="test.adapter",
    )

    assert first.accepted is True
    assert tied.accepted is True
    assert tied.quality == ObservationQuality.UNCERTAIN
    history = await historian.replay_history(limit=10)
    assert [record["value"] for record in history["records"]] == [20.0, 21.0]
    assert history["records"][-1]["order_basis"] == "source_event_time_tie"
    assert history["records"][-1]["order_gap"] is True
    assert (await historian.active_alarms())[0]["alarm_code"] == "source_order_gap"


@pytest.mark.asyncio
async def test_source_epoch_without_event_id_still_uses_source_time_ordering(
    tmp_path: Path,
) -> None:
    historian = RealityHistorian(tmp_path / "history.sqlite3")
    declaration = _declaration(resolution=0.0)
    await historian.admit(
        declaration,
        _reading(
            20.0,
            sequence=1,
            captured_at_ns=2_000,
            source_epoch="source.boot.a",
        ),
        adapter_id="test.adapter",
    )
    regressed = await historian.admit(
        declaration,
        _reading(
            21.0,
            sequence=2,
            captured_at_ns=1_000,
            source_epoch="source.boot.a",
        ),
        adapter_id="test.adapter",
    )

    assert regressed.disposition == HistorianDisposition.QUARANTINED
    assert regressed.reason == "source_event_time_regressed"


@pytest.mark.asyncio
async def test_physical_failure_outranks_order_gap_and_native_quality_is_visible(
    tmp_path: Path,
) -> None:
    historian = RealityHistorian(tmp_path / "history.sqlite3")
    declaration = _declaration(resolution=0.0)
    await historian.admit(
        declaration,
        _reading(
            20.0,
            sequence=1,
            captured_at_ns=1_000,
            source_epoch="source.boot.a",
            source_sequence=1,
        ),
        adapter_id="test.adapter",
    )
    unavailable = await historian.admit(
        declaration,
        _reading(
            None,
            sequence=2,
            captured_at_ns=2_000,
            status=ReadingStatus.UNAVAILABLE,
            source_epoch="source.boot.a",
            source_sequence=3,
        ),
        adapter_id="test.adapter",
    )

    assert unavailable.quality == ObservationQuality.BAD
    alarm = (await historian.active_alarms())[0]
    assert alarm["alarm_code"] == "reading_unavailable"
    assert alarm["severity"] == "high"

    native_bad = await historian.admit(
        declaration,
        _reading(
            21.0,
            sequence=3,
            captured_at_ns=3_000,
            source_epoch="source.boot.a",
            source_sequence=4,
            source_quality="bad",
        ),
        adapter_id="test.adapter",
    )
    assert native_bad.quality == ObservationQuality.BAD
    alarm = (await historian.active_alarms())[0]
    assert alarm["alarm_code"] == "source_quality_bad"
    assert alarm["severity"] == "high"


@pytest.mark.asyncio
async def test_deadband_accumulates_and_max_silence_forces_a_record(tmp_path: Path) -> None:
    clock = {"value": 100.0}
    historian = RealityHistorian(
        tmp_path / "history.sqlite3",
        clock=lambda: clock["value"],
        max_silence_s=10.0,
    )
    declaration = _declaration(resolution=1.0)
    await historian.admit(
        declaration,
        _reading(20.0, sequence=1, captured_at_ns=1_000),
        adapter_id="test.adapter",
    )
    clock["value"] = 105.0
    below = await historian.admit(
        declaration,
        _reading(20.4, sequence=2, captured_at_ns=2_000),
        adapter_id="test.adapter",
    )
    clock["value"] = 111.0
    heartbeat = await historian.admit(
        declaration,
        _reading(20.4, sequence=3, captured_at_ns=3_000),
        adapter_id="test.adapter",
    )

    assert below.disposition == HistorianDisposition.DEADBAND
    assert heartbeat.disposition == HistorianDisposition.ACCEPTED
    assert (await historian.replay_history(limit=10))["count"] == 2


@pytest.mark.asyncio
async def test_history_and_outbox_commit_atomically_on_collision(tmp_path: Path) -> None:
    historian = RealityHistorian(tmp_path / "history.sqlite3")
    declaration = _declaration(resolution=0.0)
    first = await historian.admit(
        declaration,
        _reading(20.0, sequence=1, captured_at_ns=1_000),
        adapter_id="test.adapter",
        delivery_observation_id="reality.obs.atomic-proof",
        delivery_payload={"observation_id": "reality.obs.atomic-proof", "value": 20.0},
    )
    assert first.delivery_observation_id == "reality.obs.atomic-proof"
    assert historian.status()["delivery_counts"] == {"queued": 1}

    with pytest.raises(HistorianCorruptionError, match="idempotency key conflicts"):
        await historian.admit(
            declaration,
            _reading(25.0, sequence=2, captured_at_ns=2_000),
            adapter_id="test.adapter",
            delivery_observation_id="reality.obs.atomic-proof",
            delivery_payload={
                "observation_id": "reality.obs.atomic-proof",
                "value": 25.0,
            },
        )

    assert historian.status()["observation_count"] == 1
    recovered = await historian.admit(
        declaration,
        _reading(25.0, sequence=2, captured_at_ns=2_000),
        adapter_id="test.adapter",
        delivery_observation_id="reality.obs.atomic-proof-2",
        delivery_payload={"observation_id": "reality.obs.atomic-proof-2"},
    )
    assert recovered.accepted is True


@pytest.mark.asyncio
async def test_active_delivery_lease_cannot_be_stolen_or_forged(tmp_path: Path) -> None:
    path = tmp_path / "history.sqlite3"
    clock = {"value": 100.0}
    first = RealityHistorian(path, clock=lambda: clock["value"], delivery_lease_s=10.0)
    admission = await first.admit(
        _declaration(),
        _reading(20.0, sequence=1, captured_at_ns=1_000),
        adapter_id="test.adapter",
    )
    await first.enqueue_delivery(
        observation_id="reality.obs.lease-proof",
        record_id=admission.record_id,
        payload={"observation_id": "reality.obs.lease-proof"},
    )
    claim = await first.claim_delivery("reality.obs.lease-proof")
    assert claim is not None

    second = RealityHistorian(path, clock=lambda: clock["value"], delivery_lease_s=10.0)
    assert await second.claim_due_deliveries(limit=1) == ()
    with pytest.raises(HistorianCorruptionError, match="lease token differs"):
        await first.mark_delivered(
            claim.observation_id,
            lease_token="forged-token",
        )

    clock["value"] = 111.0
    reclaimed = (await second.claim_due_deliveries(limit=1))[0]
    assert reclaimed.lease_token != claim.lease_token
    with pytest.raises(HistorianCorruptionError, match="lease token differs"):
        await first.mark_delivered(
            claim.observation_id,
            lease_token=claim.lease_token,
        )
    await second.mark_delivered(
        reclaimed.observation_id,
        lease_token=reclaimed.lease_token,
    )


@pytest.mark.asyncio
async def test_active_outbox_blocks_eviction_and_capacity_counters_are_durable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.sqlite3"
    historian = RealityHistorian(path, max_records=8)
    declaration = _declaration(resolution=0.0)
    for sequence in range(1, 9):
        channel_id = f"test.capacity.sensor_{sequence}"
        admission = await historian.admit(
            replace(
                declaration,
                channel_id=channel_id,
                reference_id=f"test.capacity.reference_{sequence}",
            ),
            replace(
                _reading(
                    float(sequence),
                    sequence=sequence,
                    captured_at_ns=sequence * 1_000,
                ),
                channel_id=channel_id,
            ),
            adapter_id=f"test.adapter_{sequence}",
            delivery_observation_id=f"reality.obs.capacity-{sequence}",
            delivery_payload={"sequence": sequence},
        )
        assert admission.accepted is True

    ninth_declaration = replace(
        declaration,
        channel_id="test.capacity.sensor_9",
        reference_id="test.capacity.reference_9",
    )
    ninth_reading = replace(
        _reading(9.0, sequence=9, captured_at_ns=9_000),
        channel_id="test.capacity.sensor_9",
    )
    refused = await historian.admit(
        ninth_declaration,
        ninth_reading,
        adapter_id="test.adapter_9",
    )
    assert refused.disposition == HistorianDisposition.CAPACITY_EXHAUSTED
    assert historian.status()["capacity_refusals_total"] == 1

    claim = await historian.claim_delivery("reality.obs.capacity-1")
    assert claim is not None
    await historian.mark_delivered(
        claim.observation_id,
        lease_token=claim.lease_token,
    )
    admitted = await historian.admit(
        ninth_declaration,
        ninth_reading,
        adapter_id="test.adapter_9",
    )
    assert admitted.accepted is True
    status = RealityHistorian(path, max_records=8).status()
    assert status["observation_count"] == 8
    assert status["capacity_refusals_total"] == 1
    assert status["observations_pruned_total"] == 1
    assert status["terminal_deliveries_pruned_total"] == 1


@pytest.mark.asyncio
async def test_same_channel_replacement_reclaims_capacity_in_one_transaction(
    tmp_path: Path,
) -> None:
    historian = RealityHistorian(tmp_path / "history.sqlite3", max_records=8)
    base = _declaration(resolution=0.0)
    for index in range(1, 9):
        channel_id = f"test.atomic_capacity.sensor_{index}"
        admission = await historian.admit(
            replace(
                base,
                channel_id=channel_id,
                reference_id=f"test.atomic_capacity.reference_{index}",
            ),
            replace(
                _reading(float(index), sequence=1, captured_at_ns=index * 1_000),
                channel_id=channel_id,
            ),
            adapter_id=f"test.atomic_capacity.adapter_{index}",
            delivery_observation_id=f"reality.obs.atomic-capacity-{index}",
            delivery_payload={"index": index},
            delivery_queue_limit=8,
        )
        assert admission.delivery_accepted is True

    replacement = await historian.admit(
        replace(
            base,
            channel_id="test.atomic_capacity.sensor_1",
            reference_id="test.atomic_capacity.reference_1",
        ),
        replace(
            _reading(101.0, sequence=2, captured_at_ns=10_000),
            channel_id="test.atomic_capacity.sensor_1",
        ),
        adapter_id="test.atomic_capacity.adapter_1",
        delivery_observation_id="reality.obs.atomic-capacity-replacement",
        delivery_payload={"index": 101},
        delivery_queue_limit=8,
    )

    assert replacement.accepted is True
    assert replacement.delivery_accepted is True
    assert replacement.superseded_delivery_ids == (
        "reality.obs.atomic-capacity-1",
    )
    status = historian.status()
    assert status["observation_count"] == 8
    assert status["delivery_counts"] == {"queued": 8}
    assert status["observations_pruned_total"] == 1


@pytest.mark.asyncio
async def test_alarm_and_quarantine_pruning_are_bounded_and_accounted(
    tmp_path: Path,
) -> None:
    historian = RealityHistorian(
        tmp_path / "history.sqlite3",
        max_alarm_events=8,
        max_quarantine=8,
    )
    declaration = _declaration(resolution=0.0)
    await historian.admit(
        declaration,
        _reading(20.0, sequence=100, captured_at_ns=100_000),
        adapter_id="test.adapter",
    )
    for sequence in range(1, 11):
        quarantined = await historian.admit(
            declaration,
            _reading(
                float(sequence),
                sequence=sequence,
                captured_at_ns=(100_000 + sequence),
            ),
            adapter_id="test.adapter",
        )
        assert quarantined.disposition == HistorianDisposition.QUARANTINED

    for offset in range(1, 12):
        sequence = 100 + offset
        unavailable = offset % 2 == 1
        admitted = await historian.admit(
            declaration,
            _reading(
                None if unavailable else 20.0 + offset,
                sequence=sequence,
                captured_at_ns=sequence * 1_000,
                status=(
                    ReadingStatus.UNAVAILABLE
                    if unavailable
                    else ReadingStatus.AVAILABLE
                ),
            ),
            adapter_id="test.adapter",
        )
        assert admitted.accepted is True

    status = historian.status()
    assert status["quarantine_count"] == 8
    assert status["quarantine_pruned_total"] == 2
    assert status["alarm_events_pruned_total"] == 3
    assert len(await historian.alarm_history(limit=100)) == 8


@pytest.mark.asyncio
async def test_independent_historian_instances_serialize_concurrent_writers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.sqlite3"
    first = RealityHistorian(path, busy_timeout_s=5.0)
    second = RealityHistorian(path, busy_timeout_s=5.0)

    async def _write(index: int) -> None:
        channel_id = f"test.concurrent.sensor_{index}"
        declaration = replace(
            _declaration(resolution=0.0),
            channel_id=channel_id,
            reference_id=f"reference.concurrent.sensor_{index}",
        )
        reading = replace(
            _reading(
                float(index),
                sequence=1,
                captured_at_ns=1_000 + index,
            ),
            channel_id=channel_id,
        )
        historian = first if index % 2 else second
        admission = await historian.admit(
            declaration,
            reading,
            adapter_id=f"test.adapter_{index}",
        )
        assert admission.accepted is True

    await asyncio.gather(*(_write(index) for index in range(32)))

    assert first.status()["observation_count"] == 32
    assert (await second.replay_history(limit=100))["count"] == 32


@pytest.mark.asyncio
async def test_delivery_queue_coalescing_and_quality_binding_are_atomic(
    tmp_path: Path,
) -> None:
    historian = RealityHistorian(tmp_path / "history.sqlite3")
    declaration = _declaration(resolution=0.0)
    first = await historian.admit(
        declaration,
        _reading(
            20.0,
            sequence=1,
            captured_at_ns=1_000,
            source_epoch="source.boot.a",
            source_sequence=1,
        ),
        adapter_id="test.adapter",
        delivery_observation_id="reality.obs.atomic-first",
        delivery_payload={"observation_id": "reality.obs.atomic-first"},
        delivery_queue_limit=8,
        delivery_salience=0.2,
        delivery_required_sinks=("multimodal", "advanced_cognition"),
    )
    second = await historian.admit(
        declaration,
        _reading(
            21.0,
            sequence=2,
            captured_at_ns=2_000,
            source_epoch="source.boot.a",
            source_sequence=3,
        ),
        adapter_id="test.adapter",
        delivery_observation_id="reality.obs.atomic-second",
        delivery_payload={"observation_id": "reality.obs.atomic-second"},
        delivery_queue_limit=8,
        delivery_salience=0.9,
        delivery_required_sinks=("multimodal", "advanced_cognition"),
    )

    assert first.delivery_accepted is True
    assert second.delivery_accepted is True
    assert second.superseded_delivery_ids == ("reality.obs.atomic-first",)
    assert historian.status()["delivery_counts"] == {"queued": 1, "superseded": 1}
    claim = await historian.claim_delivery("reality.obs.atomic-second")
    assert claim is not None
    historian_evidence = dict(claim.payload["historian"])
    binding_sha256 = historian_evidence.pop("binding_sha256")
    assert binding_sha256.startswith("sha256:")
    assert len(binding_sha256) == 71
    assert historian_evidence == {
        "alarm_codes": ["source_order_gap"],
        "order_basis": "source_sequence",
        "order_gap": True,
        "quality": "uncertain",
        "reason": "accepted_with_source_gap",
        "record_id": second.record_id,
        "schema": "aura.reality-historian-evidence.v1",
    }
    assert set(claim.sink_states) == {"advanced_cognition", "multimodal"}


@pytest.mark.asyncio
async def test_delivery_requires_and_persists_each_sink_receipt(tmp_path: Path) -> None:
    clock = {"value": 100.0}
    path = tmp_path / "history.sqlite3"
    historian = RealityHistorian(path, clock=lambda: clock["value"])
    admission = await historian.admit(
        _declaration(),
        _reading(20.0, sequence=1, captured_at_ns=1_000),
        adapter_id="test.adapter",
        delivery_observation_id="reality.obs.sink-proof",
        delivery_payload={"observation_id": "reality.obs.sink-proof"},
        delivery_required_sinks=("multimodal", "advanced_cognition"),
    )
    assert admission.delivery_accepted is True
    claim = await historian.claim_delivery("reality.obs.sink-proof")
    assert claim is not None
    with pytest.raises(HistorianCorruptionError, match="every required sink"):
        await historian.mark_delivered(
            claim.observation_id,
            lease_token=claim.lease_token,
        )
    await historian.mark_sink_delivered(
        claim.observation_id,
        sink="multimodal",
        receipt_id="multimodal.receipt.1",
        lease_token=claim.lease_token,
    )
    assert (
        await historian.mark_delivery_failed(
            claim.observation_id,
            error_code="advanced_cognition_unavailable",
            lease_token=claim.lease_token,
        )
        == "queued"
    )

    clock["value"] = 102.0
    restored = RealityHistorian(path, clock=lambda: clock["value"])
    retry = await restored.claim_delivery("reality.obs.sink-proof")
    assert retry is not None
    assert retry.sink_states["multimodal"] == {
        "state": "delivered",
        "receipt_id": "multimodal.receipt.1",
    }
    assert retry.sink_states["advanced_cognition"]["state"] == "pending"
    await restored.mark_sink_delivered(
        retry.observation_id,
        sink="advanced_cognition",
        receipt_id="advanced.receipt.1",
        lease_token=retry.lease_token,
    )
    await restored.mark_delivered(
        retry.observation_id,
        lease_token=retry.lease_token,
    )
    assert restored.status()["delivery_counts"] == {"delivered": 1}


@pytest.mark.asyncio
async def test_storage_headroom_refuses_before_unbounded_growth(tmp_path: Path) -> None:
    historian = RealityHistorian(
        tmp_path / "history.sqlite3",
        min_free_bytes=64 * 1024,
        disk_free_bytes=lambda _path: 1,
    )

    refused = await historian.admit(
        _declaration(),
        _reading(20.0, sequence=1, captured_at_ns=1_000),
        adapter_id="test.adapter",
    )

    assert refused.disposition == HistorianDisposition.CAPACITY_EXHAUSTED
    assert refused.reason == "historian_storage_budget_exhausted"
    status = historian.status()
    assert status["observation_count"] == 0
    assert status["capacity_refusals_total"] == 1
    assert set(status["storage_files"]) == {"database", "wal", "shm"}


@pytest.mark.asyncio
async def test_age_pruning_repairs_heads_and_alarm_references(tmp_path: Path) -> None:
    clock = {"value": 100.0}
    path = tmp_path / "history.sqlite3"
    historian = RealityHistorian(
        path,
        clock=lambda: clock["value"],
        max_age_s=60.0,
    )
    old = await historian.admit(
        _declaration(),
        _reading(
            None,
            sequence=1,
            captured_at_ns=1_000,
            status=ReadingStatus.UNAVAILABLE,
        ),
        adapter_id="test.adapter",
    )
    assert old.accepted is True
    clock["value"] = 161.0
    fresh = await historian.admit(
        _declaration(),
        _reading(21.0, sequence=2, captured_at_ns=2_000),
        adapter_id="test.adapter",
    )
    assert fresh.accepted is True

    connection = sqlite3.connect(path)
    try:
        dangling = connection.execute(
            "SELECT COUNT(*) FROM reality_alarm_events "
            "WHERE record_id=?",
            (old.record_id,),
        ).fetchone()[0]
        head = connection.execute(
            "SELECT last_stored_record_id FROM reality_channel_heads"
        ).fetchone()[0]
    finally:
        connection.close()
    assert dangling == 0
    assert head == fresh.record_id
    assert historian.status()["observations_pruned_total"] == 1


@pytest.mark.asyncio
async def test_idle_age_maintenance_resets_deadband_evidence_before_next_sample(
    tmp_path: Path,
) -> None:
    clock = {"value": 100.0}
    historian = RealityHistorian(
        tmp_path / "history.sqlite3",
        clock=lambda: clock["value"],
        max_age_s=60.0,
    )
    first = await historian.admit(
        _declaration(resolution=1.0),
        _reading(20.0, sequence=1, captured_at_ns=1_000),
        adapter_id="test.adapter",
    )
    assert first.accepted is True

    clock["value"] = 161.0
    await historian.maintain()
    assert historian.status()["observation_count"] == 0

    replacement = await historian.admit(
        _declaration(resolution=1.0),
        _reading(20.2, sequence=2, captured_at_ns=2_000),
        adapter_id="test.adapter",
    )
    assert replacement.disposition == HistorianDisposition.ACCEPTED
    assert replacement.record_id
    assert historian.status()["observation_count"] == 1


def test_index_contract_and_signed_integer_bounds_are_enforced(tmp_path: Path) -> None:
    path = tmp_path / "history.sqlite3"
    RealityHistorian(path)
    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP INDEX reality_deliveries_channel_state")
        connection.execute(
            "CREATE INDEX reality_deliveries_channel_state "
            "ON reality_deliveries(state, channel_id, created_at)"
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(HistorianCorruptionError, match="index contract differs"):
        RealityHistorian(path)

    with pytest.raises(ValueError, match="signed 64-bit"):
        _reading(
            20.0,
            sequence=(1 << 63),
            captured_at_ns=1_000,
        )
