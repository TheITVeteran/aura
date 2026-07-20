from __future__ import annotations

import os

import pytest

from core.runtime.resource_stage_guard import (
    ResourceStageGuardError,
    ack_path,
    publish_armed_ack,
    publish_compute_lease_ack,
    publish_compute_lease_request,
    publish_ready_marker,
    read_armed_ack,
    read_compute_lease_ack,
    read_compute_lease_request,
    read_ready_marker,
)


def test_resource_stage_handshake_is_hash_and_pid_bound(tmp_path):
    marker_path = tmp_path / "ready.json"
    _marker, marker_raw = publish_ready_marker(
        marker_path,
        target_pid=os.getpid(),
        trainer_sha256="a" * 64,
    )
    observed, observed_raw = read_ready_marker(
        marker_path,
        expected_target_pid=os.getpid(),
    )
    assert observed["trainer_sha256"] == "a" * 64
    assert observed_raw == marker_raw

    destination, acknowledgement, ack_raw = publish_armed_ack(
        marker_path,
        marker_raw=marker_raw,
        target_pid=os.getpid(),
        sentinel_pid=4242,
        startup_lethal_mb=73728.0,
        steady_lethal_mb=59392.0,
    )
    assert destination == ack_path(marker_path)
    verified, verified_raw = read_armed_ack(
        marker_path,
        marker_raw=marker_raw,
        expected_target_pid=os.getpid(),
        startup_lethal_mb=73728.0,
        steady_lethal_mb=59392.0,
    )
    assert verified == acknowledgement
    assert verified_raw == ack_raw


def test_resource_stage_documents_are_create_once(tmp_path):
    marker_path = tmp_path / "ready.json"
    _marker, marker_raw = publish_ready_marker(
        marker_path,
        target_pid=os.getpid(),
        trainer_sha256="b" * 64,
    )
    with pytest.raises(ResourceStageGuardError, match="already exists"):
        publish_ready_marker(
            marker_path,
            target_pid=os.getpid(),
            trainer_sha256="b" * 64,
        )

    publish_armed_ack(
        marker_path,
        marker_raw=marker_raw,
        target_pid=os.getpid(),
        sentinel_pid=4242,
        startup_lethal_mb=73728.0,
        steady_lethal_mb=59392.0,
    )
    with pytest.raises(ResourceStageGuardError, match="already exists"):
        publish_armed_ack(
            marker_path,
            marker_raw=marker_raw,
            target_pid=os.getpid(),
            sentinel_pid=4242,
            startup_lethal_mb=73728.0,
            steady_lethal_mb=59392.0,
        )


def test_resource_stage_ack_rejects_wrong_marker_or_limits(tmp_path):
    marker_path = tmp_path / "ready.json"
    _marker, marker_raw = publish_ready_marker(
        marker_path,
        target_pid=os.getpid(),
        trainer_sha256="c" * 64,
    )
    publish_armed_ack(
        marker_path,
        marker_raw=marker_raw,
        target_pid=os.getpid(),
        sentinel_pid=4242,
        startup_lethal_mb=73728.0,
        steady_lethal_mb=59392.0,
    )

    with pytest.raises(ResourceStageGuardError, match="contract is invalid"):
        read_armed_ack(
            marker_path,
            marker_raw=b"different marker\n",
            expected_target_pid=os.getpid(),
            startup_lethal_mb=73728.0,
            steady_lethal_mb=59392.0,
        )
    with pytest.raises(ResourceStageGuardError, match="contract is invalid"):
        read_armed_ack(
            marker_path,
            marker_raw=marker_raw,
            expected_target_pid=os.getpid(),
            startup_lethal_mb=73728.0,
            steady_lethal_mb=60000.0,
        )


def test_resource_stage_reader_rejects_noncanonical_or_duplicate_json(tmp_path):
    marker_path = tmp_path / "ready.json"
    marker_path.write_text(
        '{"schema":"aura.resource_stage.marker.v1","schema":"duplicate"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ResourceStageGuardError, match="duplicate JSON key"):
        read_ready_marker(marker_path)


def test_compute_lease_is_create_once_and_hash_chained(tmp_path):
    marker_path = tmp_path / "ready.json"
    _marker, marker_raw = publish_ready_marker(
        marker_path,
        target_pid=123,
        trainer_sha256="e" * 64,
    )
    _initial_path, _initial, initial_ack_raw = publish_armed_ack(
        marker_path,
        marker_raw=marker_raw,
        target_pid=123,
        sentinel_pid=456,
        startup_lethal_mb=73728.0,
        steady_lethal_mb=59392.0,
    )
    acquire_path, _request, acquire_raw = publish_compute_lease_request(
        marker_path,
        marker_raw=marker_raw,
        target_pid=123,
        sequence=1,
        workload="training_step",
        action="acquire",
        predecessor_ack_raw=initial_ack_raw,
    )
    _path, observed, observed_raw = read_compute_lease_request(
        marker_path,
        marker_raw=marker_raw,
        expected_target_pid=123,
        sequence=1,
        workload=None,
        action="acquire",
        predecessor_ack_raw=initial_ack_raw,
    )
    assert observed["workload"] == "training_step"
    assert observed_raw == acquire_raw
    _ack_path, _ack, acquire_ack_raw = publish_compute_lease_ack(
        acquire_path,
        request_raw=acquire_raw,
        target_pid=123,
        sentinel_pid=456,
        sequence=1,
        workload="training_step",
        action="acquire",
        active_lethal_mb=73728.0,
    )
    verified, _raw = read_compute_lease_ack(
        acquire_path,
        request_raw=acquire_raw,
        expected_target_pid=123,
        sequence=1,
        workload="training_step",
        action="acquire",
        active_lethal_mb=73728.0,
    )
    assert verified["stage"] == "compute_guard_armed"

    release_path, _release, release_raw = publish_compute_lease_request(
        marker_path,
        marker_raw=marker_raw,
        target_pid=123,
        sequence=1,
        workload="training_step",
        action="release",
        predecessor_ack_raw=acquire_ack_raw,
    )
    _path, _ack, release_ack_raw = publish_compute_lease_ack(
        release_path,
        request_raw=release_raw,
        target_pid=123,
        sentinel_pid=456,
        sequence=1,
        workload="training_step",
        action="release",
        active_lethal_mb=59392.0,
    )
    released, observed_release_ack = read_compute_lease_ack(
        release_path,
        request_raw=release_raw,
        expected_target_pid=123,
        sequence=1,
        workload="training_step",
        action="release",
        active_lethal_mb=59392.0,
    )
    assert released["stage"] == "steady_memory_guard_rearmed"
    assert observed_release_ack == release_ack_raw


def test_compute_lease_rejects_wrong_predecessor(tmp_path):
    marker_path = tmp_path / "ready.json"
    _marker, marker_raw = publish_ready_marker(
        marker_path,
        target_pid=123,
        trainer_sha256="f" * 64,
    )
    _initial_path, _initial, initial_ack_raw = publish_armed_ack(
        marker_path,
        marker_raw=marker_raw,
        target_pid=123,
        sentinel_pid=456,
        startup_lethal_mb=73728.0,
        steady_lethal_mb=59392.0,
    )
    publish_compute_lease_request(
        marker_path,
        marker_raw=marker_raw,
        target_pid=123,
        sequence=1,
        workload="training_step",
        action="acquire",
        predecessor_ack_raw=initial_ack_raw,
    )

    with pytest.raises(ResourceStageGuardError, match="contract is invalid"):
        read_compute_lease_request(
            marker_path,
            marker_raw=marker_raw,
            expected_target_pid=123,
            sequence=1,
            workload="training_step",
            action="acquire",
            predecessor_ack_raw=b"wrong acknowledgement",
        )
