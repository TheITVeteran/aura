from __future__ import annotations

import os

import pytest

from core.runtime.resource_stage_guard import (
    ResourceStageGuardError,
    ack_path,
    publish_armed_ack,
    publish_ready_marker,
    read_armed_ack,
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
