"""Offline play asks her model as live play does, of one copy of it held by another process.

Offline 2026-10-11 every game played offline held no model at all, so everything her model reads live (a place's rules,
its screens, what she looked up) was read by her word patterns alone, and what was measured offline was not what plays
live.
"""
from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.unit


def test_what_she_asks_her_router_is_asked_of_the_model_held_elsewhere_and_its_failures_come_back_as_failures():
    from aiohttp import web

    from tools.her_model_server import _answer
    from tools.measure_on_archived_games import HerModelOverTheWire

    class _Router:
        async def generate_with_metadata(self, prompt, system_prompt=None, timeout=180.0, **kwargs):
            return {"ok": True, "text": f"read: {prompt}", "schema": kwargs.get("schema")}

        async def think(self, prompt=None, **kwargs):
            raise TimeoutError("too slow")

    async def run():
        async def ask(request):
            return web.json_response(await _answer(_Router(), await request.json()))

        app = web.Application()
        app.add_routes([web.post("/ask", ask)])
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            wire = HerModelOverTheWire(f"http://127.0.0.1:{port}")
            said = await wire.generate_with_metadata("the rules", None, 30.0, schema={"type": "object"},
                                                     _generation_metadata_sink={}, on_token=print)
            assert said == {"ok": True, "text": "read: the rules", "schema": {"type": "object"}}
            with pytest.raises(TimeoutError):
                await wire.think("anything")
        finally:
            await runner.cleanup()

    asyncio.run(run())
