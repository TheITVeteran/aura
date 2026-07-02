"""Gateway record index — the Will/recall hot path must not stall the event loop.

Covers: cold-start bounded scan, background warm build, incremental refresh,
immediate pickup of fresh writes, and scoring parity with the legacy scan
(term-hit fraction + pin boost, best score first).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from core.memory.gateway_record_index import GatewayRecordIndex


def _write_record(
    root: Path,
    name: str,
    content: str,
    *,
    written_at: float | None = None,
    pin: bool = False,
    subdir: str = "episodic",
) -> Path:
    target_dir = root / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "payload": {
            "content": content,
            "written_at": written_at or time.time(),
            "metadata": {"session_memory_pin": True} if pin else {},
        }
    }
    path = target_dir / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _wait_until_built(index: GatewayRecordIndex, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not index._built and time.monotonic() < deadline:
        time.sleep(0.01)
    assert index._built, "background index build did not complete"


def test_cold_search_returns_bounded_results_and_kicks_background_build(tmp_path):
    for i in range(10):
        _write_record(tmp_path, f"rec{i}", f"note number {i} about voltage plasticity")
    index = GatewayRecordIndex(tmp_path)

    t0 = time.monotonic()
    results = index.search("voltage plasticity", limit=3)
    elapsed = time.monotonic() - t0

    assert results, "cold search should still return best-effort matches"
    assert elapsed < 1.0, f"cold search must stay bounded, took {elapsed:.2f}s"
    _wait_until_built(index)


def test_warm_search_uses_index_without_reparsing(tmp_path, monkeypatch):
    _write_record(tmp_path, "a", "the sky was clear over the harbor")
    _write_record(tmp_path, "b", "reactor maintenance completed cleanly")
    index = GatewayRecordIndex(tmp_path)
    index.search("harbor", limit=2)
    _wait_until_built(index)

    import core.memory.gateway_record_index as mod

    def _explode(path, mtime):
        raise AssertionError("warm search must not re-parse record files")

    monkeypatch.setattr(mod, "_parse_record", _explode)
    results = index.search("reactor maintenance", limit=2)
    assert results
    assert results[0][1].content.startswith("reactor maintenance")


def test_fresh_write_is_visible_before_next_full_refresh(tmp_path):
    _write_record(tmp_path, "old", "ancient logbook entry about tides")
    index = GatewayRecordIndex(tmp_path)
    index.search("tides", limit=2)
    _wait_until_built(index)

    _write_record(tmp_path, "new", "brand new fact about the lighthouse keeper")
    results = index.search("lighthouse keeper", limit=2)

    assert results, "a just-written record must be recallable immediately"
    assert "lighthouse" in results[0][1].content


def test_pinned_records_outrank_equal_matches(tmp_path):
    _write_record(tmp_path, "plain", "the password hint is a favorite mountain")
    _write_record(tmp_path, "pinned", "the password hint is a favorite mountain", pin=True)
    index = GatewayRecordIndex(tmp_path)
    index.search("password hint", limit=2)
    _wait_until_built(index)

    # Partial-match query (2 of 4 terms hit) keeps base score below the 1.0
    # clamp so the pin boost is observable in the ranking.
    results = index.search("password hint glacier orchard", limit=2)
    assert len(results) == 2
    assert results[0][1].memory_id == "pinned"
    assert results[0][0] > results[1][0]


def test_facade_search_sync_routes_through_index(tmp_path, monkeypatch):
    _write_record(tmp_path, "fact", "aura fused the crsm delta into the active model")

    from core.memory.memory_facade import MemoryFacade

    class _Gateway:
        root = tmp_path

    monkeypatch.setattr(
        "core.memory.memory_write_gateway.get_memory_write_gateway",
        lambda: _Gateway(),
    )
    # Reset the process-wide index so this test owns the root binding.
    import core.memory.gateway_record_index as mod

    monkeypatch.setattr(mod, "_INDEX", None)

    facade = MemoryFacade()
    results = facade._search_gateway_records_sync("crsm delta fused", limit=3)
    assert results
    assert "crsm delta" in results[0]["content"]
    assert results[0]["metadata"].get("source_lane") or results[0]["id"] == "fact"
