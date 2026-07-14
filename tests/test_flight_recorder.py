"""Black-box flight recorder (roadmap A5) — crash-survivable last moments.

The claims under test are the ones that matter when Aura dies hard:

  * every appended mind-moment survives SIGKILL (proven with a real killed
    subprocess, not a simulation);
  * torn or corrupt slots are skipped, never misread;
  * a clean shutdown produces no death report; an unclean one produces a
    grounded report whose narrative matches the recovered frames;
  * the death artifact is written through the governed async write lane and
    the incident narrator picks it up as receipt-backed evidence;
  * the continuity waking sequence receives the black-box note, superseding
    the continuity record's stale "graceful" optimism.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from core.runtime.flight_recorder import (
    _HEADER_SIZE,
    _SLOT_SIZE,
    FlightRecorder,
    inspect_ring_file,
    record_mind_moment,
    set_flight_recorder_for_test,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _isolate_recorder_singleton():
    set_flight_recorder_for_test(None)
    yield
    set_flight_recorder_for_test(None)


@pytest.fixture()
def recorder(tmp_path):
    instance = FlightRecorder(tmp_path / "flight", slot_count=64)
    yield instance
    instance.close()


def _write_frames(instance: FlightRecorder, count: int, *, start: int = 0) -> None:
    for index in range(start, start + count):
        assert instance.record_frame(
            tick=index,
            stage=f"stage_{index}",
            mode="focus",
            tick_duration_ms=float(index),
            consecutive_failures=index % 3,
        )


# ── ring + codec ───────────────────────────────────────────────────────


def test_frame_roundtrip_through_ring_file(recorder):
    assert recorder.start_sync() is None  # first boot: no previous ring
    _write_frames(recorder, 5)
    recorder.close()

    inspection = inspect_ring_file(recorder.ring_path)
    assert inspection is not None and inspection.readable
    assert not inspection.clean
    assert len(inspection.frames) == 5
    frame = inspection.frames[-1]
    assert frame.tick == 4
    assert frame.payload["stage"] == "stage_4"
    assert frame.payload["mode"] == "focus"
    assert frame.tick_duration_ms == pytest.approx(4.0)
    assert frame.consecutive_failures == 1
    assert frame.wall_ts == pytest.approx(time.time(), abs=30.0)
    assert inspection.pid == os.getpid()
    assert inspection.boot_id


def test_ring_wraparound_keeps_newest_frames(tmp_path):
    instance = FlightRecorder(tmp_path / "flight", slot_count=16)
    instance.start_sync()
    _write_frames(instance, 40)
    instance.close()

    inspection = inspect_ring_file(instance.ring_path)
    assert len(inspection.frames) == 16
    assert [frame.tick for frame in inspection.frames] == list(range(24, 40))
    assert inspection.frames[-1].payload["stage"] == "stage_39"


def test_ring_file_size_is_bounded(recorder):
    recorder.start_sync()
    expected = _HEADER_SIZE + 64 * _SLOT_SIZE
    _write_frames(recorder, 200)
    assert recorder.ring_path.stat().st_size == expected


def test_corrupt_slot_is_skipped_not_misread(recorder):
    recorder.start_sync()
    _write_frames(recorder, 6)
    recorder.close()

    raw = bytearray(recorder.ring_path.read_bytes())
    # Tear the third slot mid-payload the way a partial page flush would.
    offset = _HEADER_SIZE + 2 * _SLOT_SIZE + 40
    raw[offset : offset + 8] = b"\xff" * 8
    recorder.ring_path.write_bytes(bytes(raw))

    inspection = inspect_ring_file(recorder.ring_path)
    assert [frame.tick for frame in inspection.frames] == [0, 1, 3, 4, 5]


def test_oversized_payload_is_bounded_and_frame_still_valid(recorder):
    recorder.start_sync()
    assert recorder.record_frame(
        tick=1, stage="s", mode="m", extra={"blob": "x" * 4096}
    )
    recorder.close()
    inspection = inspect_ring_file(recorder.ring_path)
    assert len(inspection.frames) == 1
    assert inspection.frames[0].payload["stage"] == "s"
    assert "extra" not in inspection.frames[0].payload


# ── death detection ────────────────────────────────────────────────────


def test_clean_shutdown_produces_no_death_report(tmp_path):
    flight_dir = tmp_path / "flight"
    first = FlightRecorder(flight_dir, slot_count=64)
    first.start_sync()
    _write_frames(first, 3)
    assert first.mark_clean_shutdown("graceful")
    first.close()

    second = FlightRecorder(flight_dir, slot_count=64)
    assert second.start_sync() is None
    assert second.get_last_death_report() is None
    assert second.waking_note() == ""
    # The previous ring is preserved for deep forensics either way.
    assert (flight_dir / "flight_ring.prev").exists()
    second.close()


def test_unclean_shutdown_yields_grounded_death_report(tmp_path):
    flight_dir = tmp_path / "flight"
    first = FlightRecorder(flight_dir, slot_count=64)
    first.start_sync()
    _write_frames(first, 10)
    first.close()  # no clean marker: this is a death

    second = FlightRecorder(flight_dir, slot_count=64)
    report = second.start_sync()
    assert report is not None
    assert report["schema"] == "aura.flight_recorder.death.v1"
    assert report["frames_recovered"] == 10
    assert report["final_tick"] == 9
    assert report["final_stage"] == "stage_9"
    assert report["died_at"] == pytest.approx(time.time(), abs=30.0)
    assert report["uptime_s"] is not None and report["uptime_s"] >= 0.0
    assert len(report["last_frames"]) == 10
    assert report["last_frames"][-1]["stage"] == "stage_9"
    assert "went down hard" in report["narrative"]
    assert "stage_9" in report["narrative"]

    note = second.waking_note()
    assert note.startswith("Black-box record of the gap:")
    assert "went down hard" in note
    second.close()


def test_death_during_boot_is_reported_honestly(tmp_path):
    flight_dir = tmp_path / "flight"
    first = FlightRecorder(flight_dir, slot_count=64)
    first.start_sync()
    first.close()  # died before the first frame

    second = FlightRecorder(flight_dir, slot_count=64)
    report = second.start_sync()
    assert report is not None
    assert report["frames_recovered"] == 0
    assert "died during boot" in report["narrative"]
    second.close()


def test_second_runtime_is_refused_by_the_ring_lock(tmp_path):
    flight_dir = tmp_path / "flight"
    first = FlightRecorder(flight_dir, slot_count=64)
    first.start_sync()
    _write_frames(first, 2)

    second = FlightRecorder(flight_dir, slot_count=64)
    with pytest.raises(RuntimeError, match="locked by another runtime"):
        second.start_sync()
    # The intruder must not have disturbed the live ring.
    assert first.record_frame(tick=99, stage="still_alive")
    first.close()

    inspection = inspect_ring_file(first.ring_path)
    assert inspection.frames[-1].tick == 99


def test_failed_fresh_ring_open_restores_previous_ring_and_releases_lock(
    tmp_path,
    monkeypatch,
):
    flight_dir = tmp_path / "flight"
    first = FlightRecorder(flight_dir, slot_count=64)
    first.start_sync()
    _write_frames(first, 3)
    first.close()
    expected = (flight_dir / "flight_ring.bin").read_bytes()

    failed = FlightRecorder(flight_dir, slot_count=64)

    def _fail_open():
        raise OSError("injected ring-open failure")

    monkeypatch.setattr(failed, "_open_fresh_ring", _fail_open)
    with pytest.raises(OSError, match="injected ring-open failure"):
        failed.start_sync()

    assert failed.started is False
    assert failed._lock_file is None
    assert (flight_dir / "flight_ring.bin").read_bytes() == expected
    assert not (flight_dir / "flight_ring.prev").exists()

    survivor = FlightRecorder(flight_dir, slot_count=64)
    report = survivor.start_sync()
    assert report is not None
    assert report["frames_recovered"] == 3
    survivor.close()


# ── the SIGKILL proof ──────────────────────────────────────────────────


def test_frames_survive_sigkill(tmp_path):
    """The core crash-survivability claim, proven on a real killed process."""
    flight_dir = tmp_path / "flight"
    ready_path = tmp_path / "ready"
    script = tmp_path / "child.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import sys, time
            from pathlib import Path
            sys.path.insert(0, {str(REPO_ROOT)!r})
            from core.runtime.flight_recorder import FlightRecorder

            recorder = FlightRecorder(Path({str(flight_dir)!r}), slot_count=64)
            recorder.start_sync()
            for index in range(12):
                recorder.record_frame(
                    tick=index, stage=f"stage_{{index}}", mode="focus",
                    tick_duration_ms=1.0,
                )
            Path({str(ready_path)!r}).write_text("ready")
            time.sleep(60)
            """
        )
    )
    env = dict(os.environ)
    env["AURA_LOG_DIR"] = str(tmp_path / "logs")
    process = subprocess.Popen(
        [sys.executable, str(script)], cwd=str(tmp_path), env=env
    )
    try:
        deadline = time.monotonic() + 60.0
        while not ready_path.exists():
            assert process.poll() is None, "child died before writing frames"
            assert time.monotonic() < deadline, "child never became ready"
            time.sleep(0.05)
        os.kill(process.pid, signal.SIGKILL)
        process.wait(timeout=10.0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10.0)

    inspection = inspect_ring_file(flight_dir / "flight_ring.bin")
    assert inspection is not None and inspection.readable
    assert not inspection.clean, "SIGKILL must not look like a clean shutdown"
    assert len(inspection.frames) == 12
    assert inspection.frames[-1].payload["stage"] == "stage_11"

    # And the next life reads the death correctly.
    survivor = FlightRecorder(flight_dir, slot_count=64)
    report = survivor.start_sync()
    assert report is not None
    assert report["frames_recovered"] == 12
    assert report["previous_pid"] == process.pid
    survivor.close()


# ── governed artifact publication ──────────────────────────────────────


def test_start_publishes_governed_death_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep any relative paths inside the sandbox
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "strict")

    from core.governance_context import GovernanceViolationError
    from core.runtime.file_write_gateway import get_file_write_gateway

    # Control: with governance enforced, a naked gateway write is refused.
    with pytest.raises(GovernanceViolationError):
        get_file_write_gateway().write_text(
            tmp_path / "naked.txt", "refused", source="test_control"
        )

    flight_dir = tmp_path / "flight"
    first = FlightRecorder(flight_dir, slot_count=64)
    first.start_sync()
    _write_frames(first, 4)
    first.close()

    second = FlightRecorder(flight_dir, slot_count=64)
    report = asyncio.run(second.start())
    second.close()

    assert report is not None
    artifact_path = Path(report["artifact_path"])
    assert artifact_path.exists(), "death artifact must be written under governance"
    published = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert published["schema"] == "aura.flight_recorder.death.v1"
    assert published["frames_recovered"] == 4
    assert published["narrative"] == report["narrative"]


def test_record_mind_moment_is_inert_until_boot_starts_a_recorder(tmp_path):
    assert record_mind_moment(tick=1, stage="pre_boot") is False

    instance = FlightRecorder(tmp_path / "flight", slot_count=64)
    set_flight_recorder_for_test(instance)
    assert record_mind_moment(tick=1, stage="pre_boot") is False  # not started

    instance.start_sync()
    assert record_mind_moment(tick=2, stage="ticking", mode="focus") is True
    instance.close()

    inspection = inspect_ring_file(instance.ring_path)
    assert [frame.tick for frame in inspection.frames] == [2]


def test_disabled_by_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_FLIGHT_RECORDER", "0")
    instance = FlightRecorder(tmp_path / "flight", slot_count=64)
    assert instance.start_sync() is None
    assert not instance.started
    assert not (tmp_path / "flight" / "flight_ring.bin").exists()
    assert instance.record_frame(tick=1) is False


def test_flags_are_declared():
    from core.runtime.flags import declared_flags

    declared = declared_flags()
    assert "AURA_FLIGHT_RECORDER" in declared
    assert "AURA_FLIGHT_RECORDER_SLOTS" in declared
    assert declared["AURA_FLIGHT_RECORDER"].owner == "core.runtime.flight_recorder"


# ── narrator integration ───────────────────────────────────────────────


def _publish_report_for_narrator(root: Path, died_at: float) -> Path:
    flight_dir = root / "flight"
    flight_dir.mkdir(parents=True, exist_ok=True)
    artifact = flight_dir / f"death_{int(died_at)}.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": "aura.flight_recorder.death.v1",
                "generated_at": died_at + 4.0,
                "died_at": died_at,
                "uptime_s": 7212.0,
                "frames_recovered": 12,
                "final_tick": 4821,
                "final_stage": "llm_health",
                "final_rss_mb": 41210.0,
                "rss_delta_final_minute_mb": 812.0,
                "previous_boot_id": "cafe" * 8,
                "narrative": (
                    "The previous run went down hard; its last recorded moment "
                    "was 14:03:22, after 2.0h alive, in stage 'llm_health' "
                    "(tick 4821), with RSS 41210 MB and climbing (+812 MB over "
                    "the final minute)."
                ),
            }
        ),
        encoding="utf-8",
    )
    return artifact


def test_narrator_reads_the_black_box(tmp_path):
    from core.health.degraded_events import clear_degraded_events
    from core.observability.incident_narrator import IncidentNarrator

    clear_degraded_events()
    died_at = time.time() - 120.0
    artifact = _publish_report_for_narrator(tmp_path, died_at)

    narrator = IncidentNarrator(error_log_root=tmp_path)
    evidence = narrator.collect_window(minutes=60.0)
    black_box = [item for item in evidence if item.source == "flight_recorder"]
    assert len(black_box) == 1
    assert black_box[0].kind == "unclean_shutdown"
    assert black_box[0].severity == "critical"
    assert black_box[0].receipt == str(artifact)
    assert black_box[0].detail["final_stage"] == "llm_health"

    report = narrator.narrate(minutes=60.0)
    assert report["healthy"] is False
    headlines = [episode["headline"] for episode in report["episodes"]]
    assert any("hard death" in headline for headline in headlines)
    narratives = " ".join(episode["narrative"] for episode in report["episodes"])
    assert "clean-shutdown marker" in narratives  # the causal reading

    injection = narrator.get_context_injection("what happened while I was away?")
    assert "hard death" in injection
    assert str(artifact) in injection


def test_narrator_ignores_stale_deaths_outside_window(tmp_path):
    from core.observability.incident_narrator import IncidentNarrator

    _publish_report_for_narrator(tmp_path, time.time() - 7200.0)
    narrator = IncidentNarrator(error_log_root=tmp_path)
    evidence = narrator.collect_window(minutes=60.0)
    assert not [item for item in evidence if item.source == "flight_recorder"]


# ── continuity integration ─────────────────────────────────────────────


def test_waking_context_carries_the_black_box_note(tmp_path):
    from core.continuity import ContinuityEngine, ContinuityRecord

    flight_dir = tmp_path / "flight"
    first = FlightRecorder(flight_dir, slot_count=64)
    first.start_sync()
    _write_frames(first, 6)
    first.close()  # hard death

    second = FlightRecorder(flight_dir, slot_count=64)
    assert second.start_sync() is not None
    set_flight_recorder_for_test(second)

    engine = ContinuityEngine()
    engine._record = ContinuityRecord(
        last_shutdown=time.time() - 3600.0,
        last_shutdown_reason="graceful",  # stale optimism — written pre-death
        total_uptime_seconds=1000.0,
        session_count=7,
        last_conversation_summary="",
        identity_hash="",
    )
    engine._gap_seconds = 3600.0

    context = engine.get_waking_context()
    assert "Black-box record of the gap:" in context
    assert "went down hard" in context
    assert "ended gracefully" not in context  # the black box supersedes it
    second.close()


def test_waking_context_unchanged_after_clean_shutdown(tmp_path):
    from core.continuity import ContinuityEngine, ContinuityRecord

    flight_dir = tmp_path / "flight"
    first = FlightRecorder(flight_dir, slot_count=64)
    first.start_sync()
    first.mark_clean_shutdown("graceful")
    first.close()

    second = FlightRecorder(flight_dir, slot_count=64)
    assert second.start_sync() is None
    set_flight_recorder_for_test(second)

    engine = ContinuityEngine()
    engine._record = ContinuityRecord(
        last_shutdown=time.time() - 60.0,
        last_shutdown_reason="graceful",
        total_uptime_seconds=1000.0,
        session_count=7,
        last_conversation_summary="",
        identity_hash="",
    )
    engine._gap_seconds = 60.0

    context = engine.get_waking_context()
    assert "ended gracefully" in context
    assert "Black-box" not in context
    second.close()
