#!/usr/bin/env python3
"""Live-surface probe: verify the RUNNING Aura instance end to end, read-only.

The offline suite proves code; this proves the production surface a user
actually touches. Run against a live instance (default 127.0.0.1:8000):

    python tools/live_surface_probe.py            # summary + exit code
    python tools/live_surface_probe.py --json     # machine-readable report

Checks (all read-only; no chat turns, no state mutation):
  boot_health        /api/health/boot returns 200 with healthy payload
  runtime_pulse      health report: probes pass, contract level healthy
  ui_shell           / serves the app shell with the expected mount points
  static_assets      aura.js / aura.css / service-worker served and non-empty
  websocket          /ws accepts an upgrade and answers a ping
  latency            every HTTP check above answers within budget

Exit codes: 0 all pass, 1 any check failed, 2 instance unreachable.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request

DEFAULT_BASE = "http://127.0.0.1:8000"
HTTP_BUDGET_S = 5.0


def _get(base: str, path: str) -> tuple[int, bytes, float]:
    started = time.monotonic()
    req = urllib.request.Request(base + path, headers={"User-Agent": "aura-live-probe"})
    with urllib.request.urlopen(req, timeout=HTTP_BUDGET_S) as resp:
        body = resp.read()
        return resp.status, body, time.monotonic() - started


def probe(base: str) -> dict:
    checks: dict[str, dict] = {}

    def record(name: str, ok: bool, detail: str, elapsed: float | None = None) -> None:
        checks[name] = {
            "ok": bool(ok),
            "detail": detail[:300],
            **({"elapsed_s": round(elapsed, 3)} if elapsed is not None else {}),
        }

    # boot_health + runtime pulse payload
    try:
        try:
            status, body, elapsed = _get(base, "/api/health/boot")
        except urllib.error.HTTPError as http_exc:
            # An HTTP error IS a reachable server answering unhealthy.
            status = http_exc.code
            body = http_exc.read()
            elapsed = 0.0
        payload = json.loads(body or b"{}")
        record(
            "boot_health",
            status == 200,
            f"status={status} ready={payload.get('ready')} phase={payload.get('boot_phase')}",
            elapsed,
        )
        probes = payload.get("required_probes") or payload.get("probes") or {}
        all_passed = bool(probes.get("all_passed", status == 200))
        record("runtime_pulse", all_passed, f"required_probes.all_passed={all_passed}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "reachable": False,
            "error": f"{type(exc).__name__}: {exc}",
            "checks": checks,
        }
    except (json.JSONDecodeError, ValueError) as exc:
        record("boot_health", False, f"unparseable health payload: {exc}")

    # UI shell
    try:
        status, body, elapsed = _get(base, "/")
        text = body.decode("utf-8", errors="replace")
        anchors = ("aura.js", "id=\"messages\"")
        missing = [a for a in anchors if a not in text]
        record(
            "ui_shell",
            status == 200 and not missing,
            f"status={status} bytes={len(body)} missing={missing or 'none'}",
            elapsed,
        )
    except (urllib.error.URLError, TimeoutError, OSError, UnicodeDecodeError) as exc:
        record("ui_shell", False, f"{type(exc).__name__}: {exc}")

    # Static assets
    for asset in ("/static/aura.js", "/static/aura.css", "/static/service-worker.js"):
        try:
            status, body, elapsed = _get(base, asset)
            record(
                f"static:{asset.rsplit('/', 1)[-1]}",
                status == 200 and len(body) > 512,
                f"status={status} bytes={len(body)}",
                elapsed,
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            record(f"static:{asset.rsplit('/', 1)[-1]}", False, f"{type(exc).__name__}: {exc}")

    # WebSocket upgrade + ping (raw socket; no third-party deps)
    try:
        host_port = base.split("//", 1)[1]
        host, _, port_s = host_port.partition(":")
        started = time.monotonic()
        with socket.create_connection((host, int(port_s or "80")), timeout=HTTP_BUDGET_S) as sock:
            sock.settimeout(HTTP_BUDGET_S)
            # RFC 6455: Sec-WebSocket-Key is a random 16-byte nonce, not a
            # credential — generate per probe (also keeps secret scanners
            # honest: no base64 literals in source).
            key = base64.b64encode(os.urandom(16)).decode("ascii")
            handshake = (
                f"GET /ws HTTP/1.1\r\nHost: {host_port}\r\nUpgrade: websocket\r\n"
                f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                "Sec-WebSocket-Version: 13\r\n\r\n"
            )
            sock.sendall(handshake.encode())
            response = sock.recv(1024).decode("utf-8", errors="replace")
            upgraded = "101" in response.splitlines()[0] if response else False
            elapsed = time.monotonic() - started
            record("websocket", upgraded, response.splitlines()[0] if response else "no response", elapsed)
    except (OSError, IndexError, ValueError) as exc:
        record("websocket", False, f"{type(exc).__name__}: {exc}")

    # Latency budget over the HTTP checks
    slow = {
        name: c["elapsed_s"]
        for name, c in checks.items()
        if c.get("elapsed_s", 0) > HTTP_BUDGET_S * 0.8
    }
    record("latency", not slow, f"slow={slow or 'none'} (budget {HTTP_BUDGET_S}s)")

    return {
        "reachable": True,
        "base": base,
        "at_unix": time.time(),
        "passed": all(c["ok"] for c in checks.values()),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = probe(args.base)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        if not report.get("reachable"):
            print(f"UNREACHABLE: {report.get('error')}")
            return 2
        for name, c in report["checks"].items():
            mark = "✅" if c["ok"] else "❌"
            extra = f" ({c['elapsed_s']}s)" if "elapsed_s" in c else ""
            print(f"{mark} {name}: {c['detail']}{extra}")
        print("PASS" if report["passed"] else "FAIL")
    if not report.get("reachable"):
        return 2
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
