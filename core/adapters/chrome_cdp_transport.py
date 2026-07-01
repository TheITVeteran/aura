"""Raw Chrome DevTools Protocol websocket transport.

This is the only sanctioned home for the raw CDP websocket send/receive pair.
Higher-level capabilities (visible web interlocutor, browser controllers) must
route their CDP traffic through :func:`cdp_call` so raw environment sinks stay
inside the approved adapter layer, per the no-raw-bypass final blocker.
"""
from __future__ import annotations

import json
import time
from typing import Any


def cdp_call(
    target_ws_url: str,
    method: str,
    params: dict[str, Any],
    *,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Execute one CDP JSON-RPC request and wait for its matching response.

    Raises RuntimeError for missing dependency or CDP-level errors and
    TimeoutError when the response never arrives inside ``timeout`` seconds.
    """
    try:
        import websocket
    except ImportError as exc:
        raise RuntimeError("websocket-client is required for Chrome CDP control") from exc
    ws = websocket.create_connection(target_ws_url, timeout=timeout)
    try:
        message_id = 1
        ws.send(json.dumps({"id": message_id, "method": method, "params": params}))
        deadline = time.time() + timeout
        while time.time() < deadline:
            raw = ws.recv()
            if not raw:
                continue
            data = json.loads(raw)
            if data.get("id") == message_id:
                if "error" in data:
                    raise RuntimeError(str(data["error"]))
                return {"result": data}
        raise TimeoutError(f"Timed out waiting for Chrome CDP method {method}")
    finally:
        ws.close()
