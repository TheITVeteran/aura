#!/usr/bin/env python3
"""Her own model, held by one process and asked by others over a loopback socket.

Live, she reads a place's rules, its screens and what she looked up with her own model, and stands on the floor of her
word patterns where it is slow or away. A game played offline, one process a game, held no model at all (her model is
too large to load once a game), so everything her model would have read was read by the floor alone, and what was
measured offline was not what plays live. Here one process builds her router as any standalone script does
(core/brain/llm_health_router.py ``get_llm_router``) and answers ``generate_with_metadata``, ``generate`` and ``think``
for every game process that asks (tools/measure_on_archived_games.py ``--her-model``), each through the same decoding,
shapes and identity as live.

Only on the loopback interface; nothing is kept of what is asked.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("Aura.HerModelServer")

#: What may be asked of her router.
ASKED = ("generate_with_metadata", "generate", "think")


async def _answer(router: Any, body: dict[str, Any]) -> dict[str, Any]:
    how = str(body.get("how") or "")
    if how not in ASKED:
        return {"error": f"not something she is asked: {how!r}"}
    args = list(body.get("args") or [])
    kwargs = dict(body.get("kwargs") or {})
    try:
        said = await getattr(router, how)(*args, **kwargs)
    except Exception as why:  # noqa: BLE001 - every failure is the asker's to weigh, as it would be in-process
        return {"error": f"{type(why).__name__}: {str(why)[:400]}", "kind": type(why).__name__}
    return {"said": said}


async def serve(port: int) -> None:
    from aiohttp import web

    from core.brain.llm_health_router import get_llm_router

    router = get_llm_router()
    logger.info("her model is held here; warming it")
    warm = await router.generate_with_metadata("Answer with one word: ready.", None, 600.0, prefer_tier="primary",
                                               origin="her_model_server", purpose="warmup", internal_inference=True,
                                               max_tokens=8)
    logger.info("warm: %s", str((warm or {}).get("text") if isinstance(warm, dict) else warm)[:80])

    async def ask(request: Any) -> Any:
        body = await request.json()
        return web.json_response(await _answer(router, body), dumps=lambda o: json.dumps(o, default=str))

    async def health(_request: Any) -> Any:
        return web.json_response({"ok": True})

    app = web.Application(client_max_size=8 * 1024 * 1024)
    app.add_routes([web.post("/ask", ask), web.get("/health", health)])
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    logger.info("her model answers at http://127.0.0.1:%d", port)
    print(f"READY http://127.0.0.1:{port}", flush=True)
    await asyncio.Event().wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(serve(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
