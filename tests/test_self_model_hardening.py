"""CP126 hardening contracts for core/self_model.py (the robustness subset).

Covers explicit recovery from malformed persisted state, per-snapshot recovery,
retention bounds, and advancing the version only after a durable write. The
concurrency/signing/governance findings (flush-erases-enqueues, unsigned
beliefs, lock-bypass) are intentionally deferred to a dedicated pass.
"""
from __future__ import annotations

import asyncio
import json

import core.self_model as sm
from core.self_model import SelfModel, SelfSnapshot


def _snap(sid: str, ts: float) -> dict:
    return {"id": sid, "ts": ts, "summary": "s", "beliefs": {}, "confidence": 0.5, "revision_note": None}


# ── 7995f890: malformed persisted state recovers, not crashes ──────────────


def test_load_recovers_from_malformed_json(tmp_path, monkeypatch):
    f = tmp_path / "self_model.json"
    f.write_text("{ this is not valid json ")
    monkeypatch.setattr(sm, "DATA_FILE", f)
    model = asyncio.run(SelfModel.load())
    assert isinstance(model, SelfModel) and model.version == 0  # fresh, not a crash


def test_load_drops_a_malformed_snapshot_keeps_the_rest(tmp_path, monkeypatch):
    f = tmp_path / "self_model.json"
    f.write_text(json.dumps({
        "id": "x", "version": 3,
        "snapshots": {"good": _snap("good", 1.0), "bad": {"missing": "fields"}},
    }))
    monkeypatch.setattr(sm, "DATA_FILE", f)
    model = asyncio.run(SelfModel.load())
    assert "good" in model.snapshots and "bad" not in model.snapshots
    assert model.version == 3


# ── ee5a8b71 + cec30c94: retention bounds ──────────────────────────────────


def test_load_trims_excess_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "_MAX_SNAPSHOTS", 5)
    snaps = {str(i): _snap(str(i), float(i)) for i in range(20)}
    f = tmp_path / "self_model.json"
    f.write_text(json.dumps({"id": "x", "snapshots": snaps}))
    monkeypatch.setattr(sm, "DATA_FILE", f)
    model = asyncio.run(SelfModel.load())
    assert len(model.snapshots) == 5
    # The most recent (highest ts) are kept.
    assert "19" in model.snapshots and "0" not in model.snapshots


def test_trim_retention_bounds_in_memory(monkeypatch):
    monkeypatch.setattr(sm, "_MAX_SNAPSHOTS", 3)
    monkeypatch.setattr(sm, "_MAX_PENDING", 2)
    model = SelfModel(id="x")
    for i in range(10):
        model.snapshots[str(i)] = SelfSnapshot(id=str(i), ts=float(i), summary="s", beliefs={}, confidence=0.5)
    model.pending_updates = [{"k": i} for i in range(10)]
    model._trim_retention()
    assert len(model.snapshots) == 3 and len(model.pending_updates) == 2
    assert model.pending_updates == [{"k": 8}, {"k": 9}]  # newest kept


# ── e64da5a4: version advances only after a durable write ──────────────────


def test_version_advances_only_on_successful_persist(tmp_path, monkeypatch):
    import core.runtime.atomic_writer as aw

    monkeypatch.setattr(sm, "DATA_FILE", tmp_path / "self_model.json")
    model = SelfModel(id="x")
    assert model.version == 0

    # A failing write must NOT advance the in-memory version.
    async def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(aw, "async_atomic_write_text", _boom)
    asyncio.run(model.persist())
    assert model.version == 0  # unchanged after a failed write

    # A successful write advances it.
    async def _ok(path, text, *a, **k):
        (tmp_path / "self_model.json").write_text(text)

    monkeypatch.setattr(aw, "async_atomic_write_text", _ok)
    asyncio.run(model.persist())
    assert model.version == 1
