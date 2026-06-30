#!/usr/bin/env python3
"""Visible web interlocutor proof.

Runs Aura's generic WebInterlocutorSession against either:

* a local deterministic chat page (default), or
* a supplied visible web chat URL.

This is a proof harness only; product behavior lives in
core.capabilities.web_interlocutor and core.skills.web_interlocutor.
"""
from __future__ import annotations

import argparse
import asyncio
import http.server
import json
import os
import socketserver
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests

from core.capabilities.web_interlocutor import ChromeCDPDialogueBrowser, WebInterlocutorSession
from core.memory.memory_write_gateway import ConcreteMemoryWriteGateway
from core.runtime.gateways import MemoryWriteRequest


class ApprovingMemoryGateway(ConcreteMemoryWriteGateway):
    def __init__(self, root: Path):
        super().__init__(root=root, governance_decide=self._approve)

    @staticmethod
    async def _approve(**_kwargs: Any) -> dict[str, Any]:
        return {"approved": True, "receipt_id": "web-interlocutor-proof-governance"}


class ProofBrain:
    async def think(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        if "Opening message" in prompt:
            return (
                "I am trying to separate intelligence from sentience without flattening either one. "
                "What is one concrete case where adaptive problem-solving looks impressive but still "
                "does not prove felt experience, and what evidence would change your view?"
            )
        if "Next message" in prompt:
            return "Can you give one concrete example and one limitation?"
        return (
            "I learned that another dialogue partner framed sentience and intelligence "
            "as separable: intelligence can solve and adapt, while sentience requires "
            "some defensible account of felt valence or welfare. The useful limitation "
            "is that behavior alone is not final proof of subjective experience."
        )


class _ProofHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args: Any) -> None:
        return


def _write_local_chat(root: Path) -> Path:
    page = root / "index.html"
    page.write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Aura Web Interlocutor Proof Chat</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 760px; margin: 48px auto; line-height: 1.45; }
    #log { min-height: 260px; border: 1px solid #ccd3dd; padding: 16px; border-radius: 8px; white-space: pre-wrap; }
    textarea { width: 100%; height: 92px; margin-top: 16px; font: inherit; }
    button { margin-top: 8px; padding: 8px 14px; }
  </style>
</head>
<body>
  <h1>Aura Web Interlocutor Proof Chat</h1>
  <div id="log">Interlocutor ready.</div>
  <textarea aria-label="Message another AI"></textarea>
  <button id="send">Send</button>
  <script>
    const log = document.getElementById('log');
    const box = document.querySelector('textarea');
    const send = () => {
      const text = box.value.trim();
      if (!text) return;
      log.textContent += "\\n\\nAura: " + text;
      box.value = "";
      setTimeout(() => {
        const reply = text.toLowerCase().includes('example')
          ? 'Interlocutor: A concrete example is a navigation model that can plan routes without any claim of feeling; the limitation is that adaptive behavior does not settle subjective experience.'
          : 'Interlocutor: Sentience and intelligence should be separated. Intelligence is adaptive problem-solving; sentience needs a defensible welfare or valence model, not just fluent claims.';
        log.textContent += "\\n" + reply;
      }, 550);
    };
    document.getElementById('send').addEventListener('click', send);
    box.addEventListener('keydown', event => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        send();
      }
    });
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return page


class _ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


def _serve(root: Path) -> tuple[_ThreadedTCPServer, str]:
    handler = lambda *args, **kwargs: _ProofHandler(*args, directory=str(root), **kwargs)
    server = _ThreadedTCPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}/index.html"


def _chrome_binary() -> Path:
    candidates = [
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path.home() / "Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Google Chrome binary not found")


def _wait_for_cdp(endpoint: str, timeout_s: float = 12.0) -> None:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        try:
            response = requests.get(f"{endpoint.rstrip('/')}/json/version", timeout=1.0)
            if response.ok:
                return
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(0.35)
    raise TimeoutError(f"Chrome CDP endpoint did not become ready: {last_error}")


async def _launch_temp_cdp_chrome(*, profile_dir: Path, port: int) -> asyncio.subprocess.Process:
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    binary = _chrome_binary()
    cmd = [
        str(binary),
        f"--remote-debugging-port={port}",
        "--remote-allow-origins=*",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "about:blank",
    ]
    env = {**os.environ, "AURA_WEB_INTERLOCUTOR_PROOF_CHROME": "1"}
    return await get_subprocess_gateway().spawn_async(
        cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env=env,
        read_only=True,
        offline_tooling=True,
        source="proof_tooling:web_interlocutor.chrome_cdp",
    )


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    server = None
    temp_ctx = tempfile.TemporaryDirectory(prefix="aura-web-interlocutor-")
    chrome_proc: asyncio.subprocess.Process | None = None
    try:
        if args.url:
            url = args.url
        else:
            root = Path(temp_ctx.name)
            _write_local_chat(root)
            server, url = _serve(root)
        endpoint = f"http://127.0.0.1:{args.cdp_port}"
        browser = None
        if args.cdp:
            try:
                cdp_ready = requests.get(f"{endpoint}/json/version", timeout=0.5).ok
            except requests.RequestException:
                cdp_ready = False
            if not cdp_ready:
                chrome_profile = Path(temp_ctx.name) / "chrome-profile"
                chrome_proc = await _launch_temp_cdp_chrome(profile_dir=chrome_profile, port=args.cdp_port)
                _wait_for_cdp(endpoint)
            browser = ChromeCDPDialogueBrowser(endpoint=endpoint)
        memory_root = out_dir / "memory"
        session = WebInterlocutorSession(
            browser=browser,
            memory_gateway=ApprovingMemoryGateway(memory_root),
            cognitive_engine=ProofBrain(),
        )
        result = await session.run(
            objective=args.objective,
            url=url,
            opening_message=args.opening_message,
            max_turns=args.turns,
            wait_timeout_s=args.wait_timeout,
            persist_memory=True,
            context={"origin": "live_web_interlocutor_proof"},
        )
        payload = result.to_dict()
        payload["proof_url"] = url
        payload["memory_root"] = str(memory_root)
        payload["completed_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        verdict = {
            "passed": bool(result.ok and result.turns and result.memory_record_id),
            "turns": len(result.turns),
            "memory_record_id": result.memory_record_id,
            "status": result.status,
            "error": result.error,
        }
        (out_dir / "WEB_INTERLOCUTOR_RESULT.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (out_dir / "WEB_INTERLOCUTOR_VERDICT.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
        return {"result": payload, "verdict": verdict}
    finally:
        if chrome_proc is not None:
            chrome_proc.terminate()
            try:
                await asyncio.wait_for(chrome_proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                chrome_proc.kill()
                await chrome_proc.wait()
        if server is not None:
            server.shutdown()
            server.server_close()
        temp_ctx.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="", help="Optional existing web chat URL.")
    parser.add_argument("--objective", default="Discuss sentience and intelligence with another AI.")
    parser.add_argument("--opening-message", default="")
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--wait-timeout", type=float, default=20.0)
    parser.add_argument("--out-dir", default="artifacts/live_proof/web_interlocutor")
    parser.add_argument("--cdp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cdp-port", type=int, default=9223)
    args = parser.parse_args()
    payload = asyncio.run(_run(args))
    print(json.dumps(payload["verdict"], indent=2))
    return 0 if payload["verdict"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
