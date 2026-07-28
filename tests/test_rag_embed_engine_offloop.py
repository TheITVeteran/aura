"""The dense embedder must never be built on the event loop."""
import asyncio, threading, time
import pytest
from core.memory import rag


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(rag, "_EMBED_ENGINE", None)
    monkeypatch.setattr(rag, "_EMBED_ENGINE_FAILED", False)
    monkeypatch.setattr(rag, "_ENGINE_WARM_INFLIGHT", False)
    monkeypatch.setattr(rag, "_semantic_enabled", lambda: True)


def test_on_the_loop_it_declines_and_warms_off_loop(monkeypatch):
    built = threading.Event()
    def _slow_build():
        time.sleep(0.05)
        built.set()
        return "engine"
    monkeypatch.setattr(rag, "_build_embed_engine", _slow_build)

    async def _go():
        t0 = time.monotonic()
        result = rag._get_embed_engine()
        return result, time.monotonic() - t0

    result, elapsed = asyncio.run(_go())
    assert result is None            # declined, did not block
    assert elapsed < 0.02            # the loop was never frozen
    assert built.wait(timeout=3.0)   # but it DID warm in the background


def test_declining_does_not_latch_failure(monkeypatch):
    monkeypatch.setattr(rag, "_build_embed_engine", lambda: "engine")
    asyncio.run(_call())
    assert rag._EMBED_ENGINE_FAILED is False


async def _call():
    return rag._get_embed_engine()


def test_off_loop_builds_inline(monkeypatch):
    monkeypatch.setattr(rag, "_build_embed_engine", lambda: "engine")
    assert rag._get_embed_engine() == "engine"


def test_concurrent_loop_callers_start_one_warm(monkeypatch):
    starts = []
    def _build():
        starts.append(1); time.sleep(0.05); return "e"
    monkeypatch.setattr(rag, "_build_embed_engine", _build)

    async def _go():
        return [rag._get_embed_engine() for _ in range(5)]

    asyncio.run(_go())
    time.sleep(0.2)
    assert len(starts) == 1


def test_a_real_failure_still_latches(monkeypatch):
    def _boom():
        rag._EMBED_ENGINE_FAILED = True
        return None
    monkeypatch.setattr(rag, "_build_embed_engine", _boom)
    assert rag._get_embed_engine() is None
