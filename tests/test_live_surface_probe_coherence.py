"""Readiness-coherence contract for the live-surface probe.

The probe's job is catching user-facing readiness LIES. This pins the pure
coherence logic (the 'booting forever while conversation-ready' detector and
friends) so it keeps catching them without needing a live server.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_PROBE_PATH = Path(__file__).resolve().parents[1] / "tools" / "live_surface_probe.py"
_spec = importlib.util.spec_from_file_location("live_surface_probe", _PROBE_PATH)
probe_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe_mod)
readiness_incoherences = probe_mod.readiness_incoherences


def test_healthy_ready_payload_is_coherent():
    payload = {
        "status": "ready",
        "boot_phase": "kernel_ready",
        "conversation_ready": True,
        "ready": True,
        "runtime_age_s": 3600,
    }
    assert readiness_incoherences(payload) == []


def test_degraded_but_conversational_is_coherent():
    # The correct post-fix presentation of a degraded important service.
    payload = {
        "status": "degraded",
        "boot_phase": "conversation_operational",
        "conversation_ready": True,
        "ready": True,
        "runtime_age_s": 3300,
    }
    assert readiness_incoherences(payload) == []


def test_booting_forever_while_conversational_is_flagged():
    # The exact live bug: 6287s uptime, conversation ready, still "booting".
    payload = {
        "status": "booting",
        "boot_phase": "kernel_warming",
        "conversation_ready": True,
        "ready": False,
        "runtime_age_s": 6287,
    }
    problems = readiness_incoherences(payload)
    assert problems
    assert "booting" in problems[0]


def test_early_boot_is_not_flagged():
    # Genuinely still warming (under the 120s grace) — not a lie.
    payload = {
        "status": "booting",
        "boot_phase": "kernel_warming",
        "conversation_ready": True,
        "ready": False,
        "runtime_age_s": 30,
    }
    assert readiness_incoherences(payload) == []


def test_ready_without_lane_is_flagged():
    payload = {
        "status": "ready",
        "boot_phase": "kernel_ready",
        "conversation_ready": False,
        "ready": True,
        "runtime_age_s": 500,
    }
    problems = readiness_incoherences(payload)
    assert any("conversation_ready=false" in p for p in problems)


def test_ready_status_disagreeing_with_phase_is_flagged():
    payload = {
        "status": "ready",
        "boot_phase": "conversation_warming",
        "conversation_ready": True,
        "ready": True,
        "runtime_age_s": 500,
    }
    problems = readiness_incoherences(payload)
    assert any("phase=" in p for p in problems)
