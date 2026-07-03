"""Affordance-judgment probe — scoring logic (hermetic, no live instance).

The probe measures #2: does Aura CHOOSE affordances well? Precision (no
gratuitous firing) and recall (fires when it should) are scored separately so
over-eager and under-eager judgment are distinguishable. These tests pin the
scoring math against a stubbed chat lane; the live run needs the instance.
"""
from __future__ import annotations

import importlib

probe = importlib.import_module("tools.affordance_judgment_probe")


def _payload(*affordances: str) -> dict:
    return {"data": {"affordances": [{"affordance": a} for a in affordances]}}


def test_fired_affordances_reads_the_wire():
    assert probe._fired_affordances(_payload("show_sketch", "request_media")) == {
        "show_sketch",
        "request_media",
    }
    assert probe._fired_affordances({"data": {}}) == set()
    assert probe._fired_affordances({}) == set()


def test_perfect_judgment_scores_f1_one(monkeypatch):
    # Every expected affordance fires; nothing fires when it shouldn't.
    def _fake_chat(message):
        for turn, expected in probe.SCENARIOS:
            if turn == message:
                return _payload(expected) if expected else _payload()
        return _payload()

    monkeypatch.setattr(probe, "_chat", _fake_chat)
    _redirect_out(monkeypatch)
    rc = probe.main()
    assert rc == 0  # F1 >= 0.6


def test_gratuitous_firing_tanks_precision(monkeypatch):
    # Fires show_sketch on EVERY turn — including the plain-conversation ones.
    monkeypatch.setattr(probe, "_chat", lambda _m: _payload("show_sketch"))
    _redirect_out(monkeypatch)
    rc = probe.main()
    # Over-eager judgment: correct only on the one scenario expecting show_sketch,
    # gratuitous everywhere else → weak.
    assert rc == 1


def test_never_firing_tanks_recall(monkeypatch):
    # Never fires anything: perfect on the None scenarios, misses every SHOULD.
    monkeypatch.setattr(probe, "_chat", lambda _m: _payload())
    _redirect_out(monkeypatch)
    rc = probe.main()
    assert rc == 1  # recall floored by the 5 missed SHOULD-fire scenarios


def test_runtime_unavailable_returns_2(monkeypatch):
    boom_calls = []

    def _boom(_m):
        boom_calls.append(1)
        raise OSError("connection refused")

    monkeypatch.setattr(probe, "_chat", _boom)
    _redirect_out(monkeypatch)
    assert probe.main() == 2


def _redirect_out(monkeypatch, tmp=None):
    import tempfile
    from pathlib import Path

    target = Path(tempfile.mkdtemp()) / "affordance_judgment.jsonl"
    monkeypatch.setattr(probe, "OUT", target)
