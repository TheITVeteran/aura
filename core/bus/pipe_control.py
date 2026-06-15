"""Canonical helpers for control-plane multiprocessing pipe messages."""

from __future__ import annotations

import json
import time
from typing import Any


def send_supervisor_stop(pipe: Any, actor_name: str) -> bool:
    """Send the cooperative supervisor stop frame over an existing pipe."""

    if pipe is None:
        return False
    write_endpoint = pipe[1] if isinstance(pipe, tuple) and len(pipe) >= 2 else pipe
    if getattr(write_endpoint, "closed", False):
        return False
    payload = {
        "type": "stop",
        "payload": {
            "source": "supervision_tree",
            "reason": "graceful_shutdown",
            "actor": actor_name,
        },
        "trace_id": f"supervisor-stop:{actor_name}:{time.time_ns()}",
    }
    write_endpoint.send(json.dumps(payload))
    return True
