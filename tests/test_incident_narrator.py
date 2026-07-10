"""Incident Narrator — Aura's grounded self-knowledge of her own failures.

Every claim the narrator makes must trace to a receipt (file, event id,
timestamp). These tests fabricate real forensic artifacts and verify the
synthesis: parsing, correlation into episodes, causal readings, and the
conversation-lane injection gate.
"""
from __future__ import annotations

import time

import pytest

from core.observability.incident_narrator import (
    IncidentNarrator,
    _asks_about_incidents,
)


@pytest.fixture(autouse=True)
def _isolate_process_global_forensics(monkeypatch):
    """The narrator deliberately consults live process-global state beyond its
    file root (the degraded-events registry, the dropped-log counter) — right in
    production, but in a shared-process test chunk every earlier test that
    legitimately exercised a degradation path pollutes the window, so the
    "empty forensics" assertions flake (fail in-chunk, pass alone). Same class
    as the self-forensics order-dependence fixed in 460a7282: isolate the
    globals here, at the test root.
    """
    from core.health.degraded_events import clear_degraded_events

    clear_degraded_events()
    monkeypatch.setattr("core.logging_config.get_dropped_log_count", lambda: 0)
    yield
    clear_degraded_events()

_IDLE_STACK = '''  File "/opt/homebrew/lib/python3.12/threading.py", line 1032, in _bootstrap
    self._bootstrap_inner()
  File "/repo/core/runtime/runtime_hygiene.py", line 540, in _wrapped_run
    return original_run(*args, **kwargs)
'''

_CULPRIT_STACK = '''  File "/repo/interface/routes/system.py", line 2939, in api_health
    return JSONResponse(_json_safe(payload))
  File "/repo/core/memory/episodic_memory.py", line 1112, in get_summary
    return {
'''


def _write_stall_dump(root, at: float, elapsed: float = 5.1) -> None:
    stall_dir = root / "stalls"
    stall_dir.mkdir(parents=True, exist_ok=True)
    (stall_dir / f"stall_{int(at)}.txt").write_text(
        f"STALL DETECTED: {elapsed}s\n" + "=" * 40 + "\n"
        f"\nThread ID: 111\n{_IDLE_STACK}"
        f"\nThread ID: 222\n{_CULPRIT_STACK}",
        encoding="utf-8",
    )


def _write_sentinel_log(root, exit_at: float, start_at: float) -> None:
    mem_dir = root / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    exit_stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(exit_at))
    start_stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(start_at))
    (mem_dir / "sentinel.log").write_text(
        f"[{exit_stamp}] pid=1 exiting: target pid=2 vanished; capturing death syslog\n"
        f"[{start_stamp}] pid=3 armed: target pid=4 lethal_mb=43008 interval_s=0.5 ring=x\n",
        encoding="utf-8",
    )


def test_stall_dump_parsing_finds_culprit_frame(tmp_path):
    now = time.time()
    _write_stall_dump(tmp_path, now, elapsed=5.1)
    narrator = IncidentNarrator(error_log_root=tmp_path)

    items = narrator._collect_stall_dumps(cutoff=now - 60)
    assert len(items) == 1
    item = items[0]
    assert item.kind == "event_loop_stall"
    assert item.detail["elapsed_s"] == 5.1
    # The deepest in-repo, non-idle frame wins; idle thread scaffolding loses.
    assert "episodic_memory.py:1112" in item.detail["culprit_frame"]
    assert item.receipt.endswith(".txt")


def test_sentinel_log_yields_exit_and_start_events(tmp_path):
    now = time.time()
    _write_sentinel_log(tmp_path, exit_at=now - 30, start_at=now - 25)
    narrator = IncidentNarrator(error_log_root=tmp_path)

    items = narrator._collect_memory_sentinel(cutoff=now - 3600)
    kinds = [item.kind for item in items]
    assert "process_exit" in kinds
    assert "process_start" in kinds


def test_correlation_splits_episodes_on_time_gap(tmp_path):
    now = time.time()
    _write_stall_dump(tmp_path, now - 600, elapsed=6.0)   # ten minutes ago
    _write_stall_dump(tmp_path, now - 5, elapsed=5.0)     # just now
    narrator = IncidentNarrator(error_log_root=tmp_path)

    evidence = narrator._collect_stall_dumps(cutoff=now - 3600)
    evidence.sort(key=lambda item: item.at)
    episodes = narrator._correlate(evidence)
    assert len(episodes) == 2, "evidence >90s apart must form separate episodes"


def test_narrative_carries_receipts_and_causal_reading(tmp_path):
    now = time.time()
    _write_stall_dump(tmp_path, now - 10, elapsed=6.0)
    narrator = IncidentNarrator(error_log_root=tmp_path)

    report = narrator.narrate(minutes=30.0)
    assert report["episode_count"] >= 1
    episode = report["episodes"][0]
    assert "event loop froze" in episode["headline"]
    assert "receipt:" in episode["narrative"]
    assert "blocking work ran on the event loop" in episode["narrative"]
    assert report["healthy"] is False  # a stall is an error-severity episode


def test_degraded_events_are_collected_with_friendly_language(tmp_path):
    from core.health.degraded_events import (
        isolated_degraded_event_scope,
        record_degraded_event,
    )

    narrator = IncidentNarrator(error_log_root=tmp_path)
    with isolated_degraded_event_scope("narrator-test"):
        record_degraded_event(
            "mlx_client",
            "token_progress_stalled",
            detail="Qwen2.5-32B>45.0s",
            severity="error",
        )
        items = narrator._collect_degraded_events(cutoff=time.time() - 60)
    matching = [item for item in items if item.kind == "token_progress_stalled"]
    assert matching, items
    assert "stopped producing tokens" in matching[0].summary


def test_context_injection_gates_on_incident_questions(tmp_path):
    narrator = IncidentNarrator(error_log_root=tmp_path)

    assert narrator.get_context_injection("what's the weather like") == ""
    block = narrator.get_context_injection("hey, why were you slow just now?")
    assert "SYSTEM INCIDENT SELF-KNOWLEDGE" in block
    # With an empty forensic window, honesty is explicit — no invented outage.
    assert "did not register" in block


def test_context_injection_includes_real_episode(tmp_path):
    now = time.time()
    _write_stall_dump(tmp_path, now - 10, elapsed=6.0)
    narrator = IncidentNarrator(error_log_root=tmp_path)

    block = narrator.get_context_injection("what happened? did something break?")
    assert "event loop froze" in block
    assert "receipt:" in block
    assert "do not invent" in block


def test_incident_question_detector():
    assert _asks_about_incidents("Why did you restart yourself earlier?")
    assert _asks_about_incidents("give me a health report")
    assert not _asks_about_incidents("write me a poem about restarting a garden")
