"""Tests for core/ghost/ghost_line.py — the tamper-evident continuity trace.

These pin the identity properties that matter: genesis, unbroken continuity, the
"Ghost survives the Shell" verdict on a substrate swap, the discontinuity
signatures (silent identity/values overwrite, unexplained self-jump, rupture
across a transplant), governed-rebase exemption, restart continuity, hash-chain
tamper detection, and body pruning that never breaks verification.
"""
from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from core.ghost import ghost_line as gl_mod
from core.ghost.ghost_line import (
    GhostLine,
    SelfDigest,
    SubstrateFingerprint,
    _self_delta,
)
from core.runtime.audit_chain import hash_receipt_body


def _self(
    name="Aura Luna",
    values_hash="sha256:vvvvvvvvvvvvvvvv",
    *,
    continuity=1.0,
    integration=0.5,
    memory=1.0,
    boundary=1.0,
    ghost=0.7,
    essence="the integrated continuity itself",
):
    return SelfDigest(
        identity_name=name,
        core_values_hash=values_hash,
        essence=essence,
        continuity_score=continuity,
        integration=integration,
        memory_continuity=memory,
        boundary=boundary,
        ghost_strength=ghost,
    )


def _shell(model="Aura-32B-crsm-closeout", adapters=()):
    return SubstrateFingerprint(model_artifact=model, adapters=tuple(adapters))


def _append_ghost_frame_in_process(args):
    root, index = args
    gl = GhostLine(root=Path(root))
    try:
        frame = gl.advance(
            _self(name="Aura" if index % 2 else "Aura Luna", integration=(index % 5) / 10),
            _shell(),
            trigger="tick",
            now=1000.0 + index,
        )
        return frame.seq, frame.frame_id
    finally:
        gl.close()


@pytest.fixture()
def line(tmp_path):
    gl = GhostLine(root=tmp_path)
    yield gl
    gl.close()


# ── basic chain semantics ────────────────────────────────────────────────────

def test_genesis_then_continuous(line):
    f0 = line.advance(_self(), _shell(), trigger="genesis")
    assert f0.verdict == "genesis"
    assert f0.seq == 0
    assert line.length() == 1

    f1 = line.advance(_self(), _shell(), trigger="tick")
    assert f1.verdict == "continuous"
    assert f1.seq == 1
    assert f1.prev_hash == f0.entry_hash  # real hash link
    ok, problems = line.verify()
    assert ok, problems


def test_self_delta_is_normalised():
    a = _self(continuity=1.0, ghost=1.0)
    b = _self(continuity=1.0, ghost=1.0)
    assert _self_delta(a, b) == 0.0
    c = _self(continuity=0.0, integration=0.0, memory=0.0, boundary=0.0, ghost=0.0)
    d = _self(continuity=1.0, integration=1.0, memory=1.0, boundary=1.0, ghost=1.0)
    assert _self_delta(c, d) == pytest.approx(1.0)


# ── the Ghost survives the Shell ─────────────────────────────────────────────

def test_substrate_change_with_preserved_self_is_continuous(line):
    line.advance(_self(integration=0.5), _shell(model="brain-A"), trigger="genesis")
    frame = line.advance(
        _self(integration=0.55),  # tiny wobble
        _shell(model="brain-B-fused-promotion"),  # the Shell was transplanted
        trigger="substrate_change",
        cause="weight compounding promoted a new fused cortex",
    )
    assert frame.shell_changed is True
    assert frame.verdict == "substrate_changed_continuous"
    assert not frame.is_discontinuity


def test_substrate_change_with_self_jump_is_rupture(line):
    line.advance(_self(continuity=1.0, ghost=0.8), _shell(model="brain-A"), trigger="genesis")
    frame = line.advance(
        _self(continuity=0.1, integration=0.1, memory=0.2, boundary=0.2, ghost=0.1),
        _shell(model="brain-B"),
        trigger="substrate_change",
    )
    assert frame.shell_changed is True
    assert frame.is_discontinuity


# ── discontinuity signatures ─────────────────────────────────────────────────

def test_silent_identity_change_is_discontinuity(line):
    line.advance(_self(name="Aura Luna"), _shell(), trigger="genesis")
    frame = line.advance(_self(name="Not Aura"), _shell(), trigger="tick")
    assert frame.is_discontinuity
    assert any("identity" in n for n in frame.notes)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("Aura Luna", "Aura"),
        ("Aura", "Aura Luna"),
        ("  AURA   LUNA  ", "aura"),
    ],
)
def test_exact_aura_display_aliases_share_one_identity_leaf(line, before, after):
    first = line.advance(_self(name=before), _shell(), trigger="genesis")
    frame = line.advance(_self(name=after), _shell(), trigger="tick")

    assert first.self_digest.identity_id == "aura"
    assert frame.self_digest.identity_id == "aura"
    assert frame.verdict == "continuous"
    assert not frame.is_discontinuity


def test_similar_but_unknown_aura_name_remains_a_discontinuity(line):
    line.advance(_self(name="Aura Luna"), _shell(), trigger="genesis")
    frame = line.advance(_self(name="Aura Prime"), _shell(), trigger="tick")

    assert frame.self_digest.identity_id != "aura"
    assert frame.is_discontinuity


def test_historical_digest_without_identity_id_derives_canonical_leaf():
    historical = _self(name="Aura Luna").to_dict()
    historical.pop("identity_id")

    restored = SelfDigest.from_dict(historical)

    assert restored.identity_name == "Aura Luna"
    assert restored.identity_id == "aura"


def test_silent_values_change_is_discontinuity(line):
    line.advance(_self(values_hash="sha256:aaaaaaaaaaaaaaaa"), _shell(), trigger="genesis")
    frame = line.advance(_self(values_hash="sha256:bbbbbbbbbbbbbbbb"), _shell(), trigger="tick")
    assert frame.is_discontinuity
    assert any("core values" in n for n in frame.notes)


def test_unexplained_self_jump_is_discontinuity(line):
    line.advance(_self(continuity=1.0, ghost=0.9), _shell(), trigger="genesis")
    frame = line.advance(
        _self(continuity=0.2, integration=0.1, memory=0.3, boundary=0.4, ghost=0.1),
        _shell(),
        trigger="tick",
    )
    assert frame.is_discontinuity
    assert any("jump" in n for n in frame.notes)


def test_explicit_governed_rebase_is_not_discontinuity(line):
    line.advance(_self(name="Aura Luna"), _shell(), trigger="genesis")
    frame = line.advance(
        _self(name="Aura Prime"),
        _shell(),
        trigger="rebase",
        cause="operator-authorized identity rebase",
    )
    assert not frame.is_discontinuity
    assert frame.verdict == "continuous"
    assert any("rebase" in n for n in frame.notes)


# ── throttling ───────────────────────────────────────────────────────────────

def test_tick_throttle(line):
    assert line.should_advance_tick(_self(), now=1000.0) is True  # no frames yet
    line.advance(_self(integration=0.5), _shell(), trigger="genesis", now=1000.0)
    # Immediately after, unchanged self, within interval → not due.
    assert line.should_advance_tick(_self(integration=0.5), now=1001.0) is False
    # A meaningful self change → due even within the interval.
    assert line.should_advance_tick(_self(integration=0.9, ghost=0.1), now=1001.0) is True
    # Enough wall-clock elapsed → due regardless.
    assert line.should_advance_tick(_self(integration=0.5), now=1000.0 + 61) is True


# ── restart continuity ───────────────────────────────────────────────────────

def test_restart_extends_not_forks(tmp_path):
    gl = GhostLine(root=tmp_path)
    gl.advance(_self(), _shell(model="brain-A"), trigger="genesis")
    gl.advance(_self(integration=0.6), _shell(model="brain-A"), trigger="tick")
    head_before = gl.head_hash()
    gl.close()

    gl2 = GhostLine(root=tmp_path)
    assert gl2.length() == 2
    assert gl2.last_frame is not None
    assert gl2.last_frame.self_digest.identity_name == "Aura Luna"
    # The restored self is the continuity anchor: a silent Shell swap now still
    # judges against the pre-restart self.
    frame = gl2.advance(_self(integration=0.62), _shell(model="brain-C"), trigger="substrate_change")
    assert frame.seq == 2
    assert frame.prev_hash == head_before
    assert frame.verdict == "substrate_changed_continuous"
    gl2.close()


# ── tamper evidence ──────────────────────────────────────────────────────────

def test_edited_frame_body_fails_verification(line):
    line.advance(_self(), _shell(), trigger="genesis")
    line.advance(_self(integration=0.6), _shell(), trigger="tick")
    ok, _ = line.verify()
    assert ok

    # Tamper with a frame body on disk (change a self-metric).
    body_path = line.frames_dir / "00000000.json"
    env = json.loads(body_path.read_text())
    env["payload"]["self_digest"]["ghost_strength"] = 0.0001
    body_path.write_text(json.dumps(env))

    ok, problems = line.verify()
    assert not ok
    assert any("content_hash mismatch" in p["reason"] for p in problems)


def test_broken_chain_link_is_detected(line):
    line.advance(_self(), _shell(), trigger="genesis")
    line.advance(_self(integration=0.6), _shell(), trigger="tick")
    line.advance(_self(integration=0.7), _shell(), trigger="tick")

    # Delete the middle chain entry → seq gap + broken prev_hash link.
    chain_path = line._chain.path
    lines = [ln for ln in chain_path.read_text().splitlines() if ln.strip()]
    del lines[1]
    chain_path.write_text("\n".join(lines) + "\n")

    fresh = GhostLine(root=line.root)  # re-read from disk
    ok, problems = fresh.verify()
    assert not ok
    assert problems
    fresh.close()


def test_missing_retained_frame_body_fails_verification(line):
    line.advance(_self(), _shell(), trigger="genesis")
    line.advance(_self(integration=0.6), _shell(), trigger="tick")
    line._frame_path(1).unlink()

    ok, problems = line.verify()

    assert not ok
    assert any(
        problem["seq"] == 1 and problem["reason"] == "receipt body missing on disk"
        for problem in problems
    )


def test_cross_process_writers_keep_body_sequence_and_receipt_atomic(tmp_path):
    context = multiprocessing.get_context("spawn")
    work = [(str(tmp_path), index) for index in range(8)]
    with ProcessPoolExecutor(max_workers=4, mp_context=context) as executor:
        results = list(executor.map(_append_ghost_frame_in_process, work))

    line = GhostLine(root=tmp_path)
    try:
        assert sorted(seq for seq, _frame_id in results) == list(range(8))
        entries = line._chain.entries()
        assert [entry.seq for entry in entries] == list(range(8))
        for entry in entries:
            body = line._read_frame_body(entry.seq)
            assert body is not None
            assert body["seq"] == entry.seq
            assert body["frame_id"] == entry.receipt_id
            assert hash_receipt_body(body) == entry.content_hash
        ok, problems = line.verify()
        assert ok, problems
    finally:
        line.close()


# ── pruning keeps verification honest ────────────────────────────────────────

def test_pruning_bodies_does_not_break_verification(tmp_path, monkeypatch):
    monkeypatch.setattr(gl_mod, "_MAX_FRAME_BODIES", 5)
    monkeypatch.setattr(gl_mod, "_PRUNE_EVERY", 3)
    gl = GhostLine(root=tmp_path)
    for i in range(20):
        gl.advance(_self(integration=(i % 7) / 10.0), _shell(), trigger="tick" if i else "genesis")
    # Bodies were pruned (bounded, not keeping all 20) but the chain kept every
    # hash. Between prunes at most cap + (_PRUNE_EVERY-1) bodies accumulate.
    body_files = list(gl.frames_dir.glob("*.json"))
    assert len(body_files) <= gl_mod._MAX_FRAME_BODIES + gl_mod._PRUNE_EVERY
    assert len(body_files) < 20
    assert gl.length() == 20
    ok, problems = gl.verify()  # pruned bodies must not be reported as missing
    assert ok, problems
    gl.close()


def test_integrity_summary(line):
    line.advance(_self(), _shell(model="brain-X", adapters=("math-lora",)), trigger="genesis")
    info = line.integrity()
    assert info["length"] == 1
    assert info["last_verdict"] == "genesis"
    assert info["identity_name"] == "Aura Luna"
    assert info["last_shell"]["model_artifact"] == "brain-X"
    assert "math-lora" in info["last_shell"]["adapters"]
