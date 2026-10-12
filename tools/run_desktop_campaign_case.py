#!/usr/bin/env python3
"""Submit one bounded campaign case through the running desktop's durable chat path.

This client launches no runtime and changes no model or browser policy. A terminal
delivery remains unqualified until its task outcome is independently observed.
On a deadline it requests cancellation and waits for the durable terminal state.
An unresolved cancellation forbids advancing the campaign.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HEADERS = {"X-Aura-Surface": "desktop-ui", "X-Aura-Desktop-Request": "same-origin",
           "X-Aura-Require-CognitiveEngine": "true"}
TRANSPORT_ERRORS = (OSError, ValueError, httpx.RequestError)


async def run_case(client, *, message: str, session: str, key: str, seconds: float,
                   save, poll_s: float = 1., cancel_grace_s: float = 30.) -> dict:
    """One admission, scoped status reads and explicit cancellation; never inferred task success."""
    if not message.strip() or not math.isfinite(seconds) or not 0 < seconds <= 3600:
        raise ValueError("a nonempty request and a finite deadline of at most one hour are required")
    record = {"idempotency_key": key, "session_id": session, "message_sha256": hashlib.sha256(message.encode()).hexdigest(),
              "qualification": "unmeasured", "task_passed": None, "delivery_terminal": False,
              "safe_to_advance": False, "deadline_s": seconds, "errors": []}
    await save(record)
    posted = asyncio.create_task(client.post("/api/chat", json={"message": message, "session_id": session},
                                             headers={**HEADERS, "X-Idempotency-Key": key}, timeout=None))
    await asyncio.sleep(0)
    began = time.monotonic()
    cancelled_at = None
    try:
        while True:
            now = time.monotonic()
            if cancelled_at is None and now - began >= seconds:
                cancelled_at = now
                record["cancellation_requested"] = True
                await save(record)
                try:
                    response = await client.post(f"/api/chat/delivery/{key}/cancel", params={"session_id": session}, headers=HEADERS)
                    record["cancellation"] = {"http_status": response.status_code, "body": response.json()}
                except TRANSPORT_ERRORS as error:
                    record["errors"].append(f"cancellation: {type(error).__name__}: {error}")
                await save(record)
            if cancelled_at is not None and now - cancelled_at >= cancel_grace_s:
                record["state"] = "cancellation_unresolved"
                break
            try:
                response = await client.get(f"/api/chat/delivery/{key}", params={"session_id": session}, headers=HEADERS)
                status = response.json()
                if status.get("idempotency_key") not in (None, key):
                    raise ValueError("status belongs to another delivery")
                if response.status_code == 200 and status.get("terminal") is True:
                    if status.get("idempotency_key") != key or not isinstance(status.get("result"), dict):
                        raise ValueError("terminal delivery lacks its identity or durable result")
                    record.update(delivery_terminal=True, delivery=status,
                                  state="terminal_requires_independent_review")
                    break
                record["last_status"] = status
            except TRANSPORT_ERRORS as error:
                record["errors"] = (record["errors"] + [f"status: {type(error).__name__}: {error}"])[-20:]
            if posted.done() and "post_status" not in record:
                try:
                    posted_response = posted.result()
                    record["post_status"] = posted_response.status_code
                except TRANSPORT_ERRORS as error:
                    record["post_status"] = "transport_failed"
                    record["errors"].append(f"submit: {type(error).__name__}: {error}")
            await save(record)
            await asyncio.sleep(poll_s)
    finally:
        if not record["delivery_terminal"] and not record.get("cancellation_requested"):
            record["cancellation_requested"] = True
            record["state"] = "cancellation_unresolved"
            try:
                response = await client.post(f"/api/chat/delivery/{key}/cancel", params={"session_id": session}, headers=HEADERS)
                record["cancellation"] = {"http_status": response.status_code, "body": response.json()}
            except TRANSPORT_ERRORS as error:
                record["errors"].append(f"cancellation: {type(error).__name__}: {error}")
        if not posted.done():
            posted.cancel()
        with contextlib.suppress(asyncio.CancelledError, *TRANSPORT_ERRORS):
            await posted
        record["elapsed_s"] = round(time.monotonic() - began, 3)
        await save(record)
    return record


async def main() -> int:
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message-file", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--seconds", type=float, default=300.)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("use a new output path so earlier attempts remain intact")
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=10.) as client:
        health_response = await client.get("/api/health/boot")
        health = health_response.json()
        revision = health.get("runtime_revision") or {}
        if (revision.get("actual_commit_sha") != args.expected_commit or revision.get("verified") is not True
                or revision.get("source_current") is not True):
            raise RuntimeError("running desktop revision is not the requested verified checkpoint")
        if health.get("conversation_ready") is not True:
            raise RuntimeError("the running desktop conversation lane is not ready")

        async def save(record):
            with local_internal_governed_scope("desktop_campaign_case"):
                await get_file_write_gateway().write_json_async(args.out, {**record, "runtime_revision": revision},
                                                               schema_version=1, source="desktop_campaign_case")

        key = f"aura-campaign-{uuid.uuid4()}"
        result = await run_case(client, message=args.message_file.read_text(), session="desktop-campaign",
                                key=key, seconds=args.seconds, save=save)
    print(json.dumps({"receipt": str(args.out.resolve()), "state": result["state"], "qualified": False}))
    return 0 if result["delivery_terminal"] else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
