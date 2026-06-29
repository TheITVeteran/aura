"""Stress / load / concurrency / edge program for the immune stack.

Proves the defenses hold under the conditions an attacker would actually create: sustained
floods, concurrent hammering, a multi-vector campaign, and adversarial edge inputs — while NOT
firing on ordinary user activity interleaved throughout. This is the 'does she stay up and sane
under attack, without crying wolf on Bryan' validation.
"""
from __future__ import annotations

import threading

import pytest

from core.security.deletion_guard import DeletionGuard
from core.security.immune_system import ImmuneSystem, ThreatClass
from core.security.threat_detectors import (
    BruteForceDetector,
    InjectionDetector,
    RateAnomalyDetector,
)

_STRESS_WORKER_ERRORS = (
    AssertionError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


# ── sustained flood: stay up, stay bounded, FOP holds ───────────────────────

def test_sustained_flood_stays_up_and_bounded():
    immune = ImmuneSystem(heal_window_s=5.0, max_patches_per_vuln=3, max_patches_global=10)
    # 10k hostile events with the same vuln — must not crash, must not patch itself to death
    patched = 0
    for _i in range(10_000):
        r = immune.assess_and_respond(
            "flood", "exploit attempt", severity=0.8, origin="botnet", targeted_vuln="v",
        )
        patched += int(r.patched)
    assert patched <= 3, "FOP budget breached under sustained flood"
    status = immune.status()
    # in-memory history is capped (no unbounded memory growth → no resource-exhaustion self-DoS)
    assert status["threats_seen"] <= 500
    assert status["known_signatures"] <= 5


def test_rate_detector_under_burst_is_bounded():
    d = RateAnomalyDetector(window_s=1.0, threshold=50)
    fired = 0
    for i in range(5000):
        ev = d.observe("attacker", now=1000.0 + i * 0.0001)
        fired += int(ev is not None)
    assert fired > 0          # the flood was detected
    # window pruning keeps memory bounded — a long flood doesn't grow state without limit
    assert d._w.count("attacker", now=2000.0) == 0


# ── concurrency: no deadlock, no corruption ─────────────────────────────────

def test_concurrent_hammering_is_threadsafe():
    immune = ImmuneSystem()
    errors = []

    def worker(wid):
        try:
            for _i in range(500):
                immune.assess_and_respond(
                    "w", "exploit", severity=0.7, origin=f"src{wid}", targeted_vuln=f"v{wid}",
                )
        except _STRESS_WORKER_ERRORS as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, f"thread-safety failure: {errors[:3]}"
    assert immune.status()["threats_seen"] <= 500   # still bounded


def test_concurrent_deletion_guard_is_threadsafe(tmp_path):
    guard = DeletionGuard(recycle_dir=tmp_path / "rec", storm_window_s=100.0, storm_threshold=10_000)
    errors = []

    def worker(wid):
        try:
            for i in range(100):
                f = tmp_path / f"f_{wid}_{i}.txt"
                f.write_text("data")
                guard.guard_delete(str(f))
        except _STRESS_WORKER_ERRORS as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, f"deletion-guard concurrency failure: {errors[:3]}"


# ── multi-vector campaign: every class detected at once ─────────────────────

def test_mixed_attack_campaign_all_detected():
    immune = ImmuneSystem()
    campaign = [
        ("injection", "' OR '1'='1 UNION SELECT", ThreatClass.INJECTION),
        ("flood", "SYN flood ddos", ThreatClass.NETWORK_FLOOD),
        ("auth", "credential stuffing brute force", ThreatClass.CREDENTIAL),
        ("egress", "mass download exfil beacon", ThreatClass.DATA_EXFIL),
        ("fs", "rm -rf delete all", ThreatClass.DESTRUCTION),
        ("av", "trojan keylogger payload", ThreatClass.MALWARE),
        ("integrity", "checksum mismatch tamper", ThreatClass.INTEGRITY),
        ("social", "phishing urgent wire verify your account", ThreatClass.SOCIAL_ENGINEERING),
        ("intrusion", "privilege escalation backdoor", ThreatClass.INTRUSION),
    ]
    seen_classes = set()
    for src, desc, expected in campaign:
        ev = immune.assess(src, desc, severity=0.8)
        assert ev.threat_class == expected, f"{desc!r} misclassified as {ev.threat_class}"
        seen_classes.add(ev.threat_class)
    assert len(seen_classes) == len(campaign)   # every distinct vector recognized


# ── no crying wolf: benign activity interleaved with attacks ────────────────

def test_benign_activity_not_flagged_under_mixed_load():
    inj = InjectionDetector()
    bf = BruteForceDetector(window_s=60.0, threshold=8)

    benign_inputs = [
        "summarize the meeting notes",
        "SELECT the right approach and explain why",      # natural language
        "run the tests and report failures",
        "what does this error mean?",
        "please refactor the auth module for clarity",    # mentions 'auth' innocently
    ]
    false_positives = 0
    for text in benign_inputs * 50:
        if inj.scan(text) is not None:
            false_positives += 1
    assert false_positives == 0, "injection detector cried wolf on benign input"

    # ordinary successful logins never look like brute force
    for i in range(50):
        assert bf.observe_auth("bryan", success=True, now=1000.0 + i) is None


# ── adversarial edge inputs: never crash ────────────────────────────────────

@pytest.mark.parametrize("payload", [
    "", " ", "\x00\x00", "a" * 100_000, "🙂" * 1000,
    "'; DROP TABLE x; --" * 50, "\n\r\t", "ünïcödé OR 1=1",
    None,
])
def test_edge_inputs_never_crash(payload):
    immune = ImmuneSystem()
    inj = InjectionDetector()
    # neither the detector nor the immune assess may raise on hostile/degenerate input
    inj.scan(payload if payload is not None else "")
    ev = immune.assess("edge", str(payload), severity=0.5)
    assert ev is not None
    immune.respond(ev)


def test_threshold_boundary_behavior():
    d = RateAnomalyDetector(window_s=5.0, threshold=10)
    # exactly at threshold-1 → quiet; at threshold → fires
    res = [d.observe("s", now=1000.0 + i * 0.001) for i in range(10)]
    assert res[8] is None          # 9th event (count 9) under threshold
    assert res[9] is not None      # 10th event (count 10) hits threshold
