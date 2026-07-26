"""Launcher contracts: the neuron mark, and a blocked boot that says so.

Two things the launcher must keep doing:

1. The boot mark is a NEURON (soma / dendrites / myelinated axon / travelling
   spikes) in a retro-arcade idiom — square "pixel" nodes and CRT scanlines —
   not the old orbital atom.
2. When a start is positively refused because another runtime holds the
   instance lock, the window shows THAT, instead of spinning on
   "Aura is waking up… waiting for boot health".
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SWIFT = (PROJECT_ROOT / "scripts" / "AuraLauncher.swift").read_text(encoding="utf-8")


# ── the neuron mark ────────────────────────────────────────────────────────


def test_mark_is_a_neuron_not_an_atom():
    assert "Aura's neuron mark" in SWIFT
    for part in ("soma", "dendrite", "axon", "myelin", "bouton"):
        assert part.lower() in SWIFT.lower(), f"neuron anatomy missing: {part}"


def test_orbital_atom_is_gone():
    # The old mark drove "electrons" around ellipses; none of that should remain.
    assert "Aura's orbital mark" not in SWIFT
    assert "electron" not in SWIFT.lower()
    assert 'forKey: "orbit"' not in SWIFT


def test_retro_arcade_idiom_is_present():
    # Square pixel nodes, stepped (discrete) animation, and CRT scanlines are
    # what make it read as a sprite rather than a diagram.
    assert "func pixel(" in SWIFT, "square pixel nodes are the arcade vocabulary"
    assert "calculationMode = .discrete" in SWIFT, "stepped motion, not smooth glide"
    assert "scanline" in SWIFT.lower()
    assert "lineDashPattern" in SWIFT, "myelin segments"


def test_spikes_ride_real_paths():
    # A spike must be bound to its fibre's path, not approximated with offsets.
    assert "CAKeyframeAnimation(keyPath: \"position\")" in SWIFT
    assert 'forKey: "spike"' in SWIFT


def test_mark_keeps_its_public_shape():
    # Call sites construct it by diameter; renaming/reshaping would break them.
    assert "private final class AuraSigilView: NSView" in SWIFT
    assert "init(diameter: CGFloat)" in SWIFT


# ── the blocked-boot notice ────────────────────────────────────────────────


def test_launcher_reads_the_boot_blocked_notice():
    assert "boot_blocked.json" in SWIFT
    assert "readBootBlockedNotice" in SWIFT


def test_blocked_boot_short_circuits_the_waking_up_screen():
    # The check must run BEFORE the "waiting for boot health" copy is rendered.
    pending = SWIFT.index("private func renderPendingLaunch")
    body = SWIFT[pending:pending + 900]
    assert "readBootBlockedNotice()" in body
    assert body.index("readBootBlockedNotice()") < body.index("Waiting for Aura to publish boot health")


def test_blocked_screen_shows_reason_and_remedy():
    start = SWIFT.index("private func renderBootBlocked")
    body = SWIFT[start:start + 700]
    assert "Another Aura is already running" in body
    assert "notice.reason" in body and "notice.remedy" in body
    assert ".rose" in body, "an instance conflict is a problem state, not progress"


def test_dead_holder_is_not_treated_as_a_live_blocker():
    start = SWIFT.index("private func readBootBlockedNotice")
    body = SWIFT[start:start + 1200]
    # kill(pid, 0) liveness probe: a notice about an exited process must clear.
    assert "kill(pid_t(pid), 0)" in body
