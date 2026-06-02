from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.mycelial_graph import MycelialNetwork


@pytest.mark.asyncio
async def test_mycelial_graph_rejects_cycles_and_preserves_existing_edges():
    net = MycelialNetwork()

    assert await net.add_edge("memory:a", "skill:b") is True
    assert await net.add_edge("skill:b", "memory:a") is False

    assert net.G.has_edge("memory:a", "skill:b")
    assert not net.G.has_edge("skill:b", "memory:a")


@pytest.mark.asyncio
async def test_mycelial_graph_returns_false_on_recoverable_graph_failure():
    net = MycelialNetwork()
    called = {"add_edge": False}

    def _raise_type_error(*_args, **_kwargs):
        called["add_edge"] = True
        raise TypeError("bad node")

    net.G = SimpleNamespace(
        add_edge=_raise_type_error,
        has_edge=lambda *_args, **_kwargs: False,
        remove_edge=lambda *_args, **_kwargs: None,
    )

    assert await net.add_edge("memory:a", "skill:b") is False
    assert called["add_edge"] is True
