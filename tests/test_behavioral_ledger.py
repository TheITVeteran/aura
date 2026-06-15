"""tests/test_behavioral_ledger.py
======================================
The L3 behavioral output ledger must be append-only and tamper-evident: every
recorded run on the sealed held-out pack is hash-chained, so an edited score, a
reordered row, or a deleted entry is detectable, and the longitudinal summary is
computed only from real recorded entries (never synthesized).
"""
from __future__ import annotations

import json

import pytest

from core.evaluation.behavioral_ledger import BehavioralLedger, LedgerEntry


def _ledger(tmp_path):
    return BehavioralLedger(tmp_path / "ledger.jsonl")


def test_append_chains_entries(tmp_path):
    led = _ledger(tmp_path)
    e0 = led.append(pack_id="P", manifest_hash="M", task_count=50, condition="baseline",
                    score=0.3, passed=False, commit_sha="c0")
    e1 = led.append(pack_id="P", manifest_hash="M", task_count=50, condition="candidate",
                    score=0.9, passed=True, commit_sha="c0")
    assert e0.seq == 0 and e1.seq == 1
    assert e0.prev_hash == "GENESIS"
    assert e1.prev_hash == e0.entry_hash
    assert e1.entry_hash == e1.computed_hash()
    assert led.verify_chain() == (True, "ok")


def test_append_requires_sealed_pack_identity(tmp_path):
    led = _ledger(tmp_path)
    with pytest.raises(ValueError):
        led.append(pack_id="", manifest_hash="M", task_count=1, condition="x",
                   score=1.0, passed=True)
    with pytest.raises(ValueError):
        led.append(pack_id="P", manifest_hash="", task_count=1, condition="x",
                   score=1.0, passed=True)


def test_tampered_score_is_detected(tmp_path):
    led = _ledger(tmp_path)
    led.append(pack_id="P", manifest_hash="M", task_count=50, condition="candidate",
               score=0.5, passed=False, commit_sha="c0")
    led.append(pack_id="P", manifest_hash="M", task_count=50, condition="candidate",
               score=0.6, passed=False, commit_sha="c0")
    # Tamper: rewrite the first entry's score directly in the file.
    lines = led.path.read_text().splitlines()
    first = json.loads(lines[0])
    first["score"] = 0.99  # inflate the recorded score, keep the old hash
    lines[0] = json.dumps(first)
    led.path.write_text("\n".join(lines) + "\n")
    ok, detail = led.verify_chain()
    assert ok is False
    assert "hash mismatch" in detail


def test_deleted_entry_is_detected(tmp_path):
    led = _ledger(tmp_path)
    for s in (0.3, 0.4, 0.5):
        led.append(pack_id="P", manifest_hash="M", task_count=10, condition="candidate",
                   score=s, passed=False, commit_sha="c0")
    lines = led.path.read_text().splitlines()
    del lines[1]  # remove the middle entry
    led.path.write_text("\n".join(lines) + "\n")
    ok, detail = led.verify_chain()
    assert ok is False  # seq gap or broken link


def test_summary_uses_real_recorded_numbers(tmp_path):
    led = _ledger(tmp_path)
    led.append(pack_id="P", manifest_hash="M", task_count=50, condition="baseline",
               score=0.20, passed=False, commit_sha="c0")
    led.append(pack_id="P", manifest_hash="M", task_count=50, condition="candidate",
               score=0.80, passed=True, commit_sha="c0")
    led.append(pack_id="P", manifest_hash="M", task_count=50, condition="candidate",
               score=0.90, passed=True, commit_sha="c1")
    summary = led.summary()
    assert summary["total_runs"] == 3
    assert summary["chain_ok"] is True
    assert summary["held_out_integrity_ok"] is True
    cand = summary["conditions"]["candidate"]
    assert cand["runs"] == 2
    assert cand["mean_score"] == 0.85          # (0.80 + 0.90) / 2 — real
    assert cand["pass_rate"] == 1.0
    assert cand["score_trend"] == round(0.90 - 0.80, 4)
    assert summary["conditions"]["baseline"]["mean_score"] == 0.20


def test_changed_sealed_pack_flags_integrity(tmp_path):
    led = _ledger(tmp_path)
    led.append(pack_id="P", manifest_hash="M1", task_count=50, condition="candidate",
               score=0.8, passed=True, commit_sha="c0")
    led.append(pack_id="P", manifest_hash="M2", task_count=50, condition="candidate",
               score=0.99, passed=True, commit_sha="c0")
    summary = led.summary()
    # Same pack_id with two different manifests => the sealed set changed.
    assert summary["held_out_integrity_ok"] is False
    assert summary["chain_ok"] is True  # chain itself is intact


def test_entries_roundtrip(tmp_path):
    led = _ledger(tmp_path)
    led.append(pack_id="P", manifest_hash="M", task_count=7, condition="candidate",
               score=0.42, passed=True, commit_sha="c0")
    reloaded = led.entries()
    assert len(reloaded) == 1
    assert isinstance(reloaded[0], LedgerEntry)
    assert reloaded[0].score == 0.42
    assert reloaded[0].entry_hash == reloaded[0].computed_hash()


def test_record_bundle_to_ledger(tmp_path):
    from core.evaluation.behavioral_ledger import record_bundle_to_ledger

    class _Solver:
        def __init__(self, score):
            self.score = score

    class _Smoke:
        pack_id = "PACK123"
        manifest_hash = "MAN456"
        task_count = 50
        passed = True
        baseline = _Solver(0.3)
        candidate = _Solver(0.92)

    class _Bundle:
        smoke = _Smoke()

    recorded = record_bundle_to_ledger(_Bundle(), ledger_path=tmp_path / "l.jsonl", commit_sha="c9")
    assert [e.condition for e in recorded] == ["baseline", "candidate"]
    assert recorded[1].score == 0.92 and recorded[1].passed is True
    assert recorded[0].passed is False
    led = BehavioralLedger(tmp_path / "l.jsonl")
    assert led.verify_chain() == (True, "ok")
    assert led.summary()["held_out_integrity_ok"] is True
