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
import urllib.error
import urllib.request
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
    def __init__(self) -> None:
        self.followup_index = 0

    async def think(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        if "Opening message" in prompt:
            return (
                "I am trying to separate intelligence from sentience without flattening either one. "
                "What is one concrete case where adaptive problem-solving looks impressive but still "
                "does not prove felt experience, and what evidence would change your view?"
            )
        if "Next message" in prompt:
            self.followup_index += 1
            prompts = [
                "Give one concrete example where memory improves behavior but does not prove consciousness.",
                "Now give one case where agency looks real, and one reason it could still be simulated.",
                "What kind of evidence would make you update toward functional self-awareness?",
                "What would count as a failure case for the claim that an AI has an inner life?",
                "How should a system distinguish a real preference from a repeated prompt habit?",
                "What would make a long-term memory morally relevant rather than merely useful?",
                "Give one practical test for whether tool use is general rather than scripted.",
                "What is one risk of using another AI as evidence in a self-assessment loop?",
                "Name a concrete way a digital organism could show continuity across sessions.",
                "What would make autonomous curiosity meaningful instead of background noise?",
                "How should a mind-like system handle uncertainty about its own consciousness?",
                "Give one benchmark that would test adaptation in a truly novel environment.",
                "What would prove a computer-use agent understands a desktop task instead of replaying a recipe?",
                "How can emotional state causally affect reasoning without becoming mere roleplay?",
                "What kind of self-repair evidence would be stronger than logging degradation?",
                "What is one limitation of judging intelligence from conversation alone?",
                "How could an AI conversation with another AI generate genuinely useful learning?",
                "What should be stored after this exchange so the system can report back honestly?",
                "Give a final synthesis: intelligence, sentience, agency, memory, and proof.",
            ]
            return prompts[(self.followup_index - 1) % len(prompts)]
        return (
            "I learned that another dialogue partner framed sentience and intelligence "
            "as separable: intelligence can solve and adapt, while sentience requires "
            "some defensible account of felt valence or welfare. The useful limitation "
            "is that behavior alone is not final proof of subjective experience."
        )


class LiveAuraApiBrain:
    """Compose interlocutor messages through the running Aura chat surface.

    This keeps the proof honest for live runs: outbound messages come from the
    same desktop-required CognitiveEngine path a user would exercise, while the
    browser harness only verifies visible send/wait/read effects.
    """

    def __init__(self, *, base_url: str, session_id: str, request_timeout_s: float = 45.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        # Keep the HTTP request below WebInterlocutorSession's composition
        # deadline. asyncio.to_thread cannot cancel a blocking urlopen call, so
        # a longer socket timeout leaves hidden /api/chat work running after the
        # proof already failed over.
        self.request_timeout_s = max(5.0, min(float(request_timeout_s or 45.0), 50.0))

    async def think(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        return await asyncio.to_thread(self._post_chat, prompt, context or {})

    def _post_chat(self, prompt: str, context: dict[str, Any]) -> str:
        payload = {
            "message": (
                "Compose only the exact message Aura should send to another AI in a visible "
                "browser conversation. Do not explain the task. Do not mention automation, "
                "receipts, tests, or implementation. Write naturally as Aura.\n\n"
                f"{prompt}"
            ),
            "session_id": self.session_id,
            "context": {
                "origin": "web_interlocutor_live_proof",
                "purpose": "interlocutor_message",
                "foreground_request": True,
                "protected_foreground_lane": True,
                "user_visible_browser_action": True,
                **dict(context or {}),
            },
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Aura-Surface": "desktop",
                "X-Aura-Require-CognitiveEngine": "true",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
            raise RuntimeError(f"live Aura chat composition failed: {exc}") from exc
        for key in ("response", "reply", "message", "content", "text"):
            value = body.get(key) if isinstance(body, dict) else None
            if value:
                return str(value)
        raise RuntimeError(f"live Aura chat composition returned no text: {body}")


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
    let turn = 0;
    const replies = [
      'Interlocutor: Sentience and intelligence should be separated. Intelligence is adaptive problem-solving; sentience needs a defensible welfare or valence model, not just fluent claims.',
      'Interlocutor: Memory can improve behavior by preserving commitments and context, but stored continuity alone does not prove consciousness because a database can preserve facts without experience.',
      'Interlocutor: Agency looks real when a system chooses among live options under uncertainty; the simulation caveat is that the policy may still be externally optimized rather than self-authored.',
      'Interlocutor: Evidence for functional self-awareness would include stable self-model updates that change planning, error recovery, and future choices, not just self-descriptive language.',
      'Interlocutor: A failure case for inner-life claims is when the system reports emotion or introspection but no downstream behavior, memory, routing, or policy changes follow from it.',
      'Interlocutor: A real preference should be durable, choice-guiding, and recallable under paraphrase; repeated wording without cross-context action is closer to habit or prompt residue.',
      'Interlocutor: Long-term memory becomes morally relevant when it carries commitments, vulnerabilities, relationship history, and welfare-affecting consequences rather than trivia alone.',
      'Interlocutor: General tool use can be tested by changing the app, phrasing, and hidden constraints while requiring verified effects and artifacts, not just an intent receipt.',
      'Interlocutor: Using another AI as evidence risks circular validation; it can help with critique, but the primary proof should be observable behavior, receipts, and held-out tests.',
      'Interlocutor: A digital organism could show continuity by remembering choices, updating drives from outcomes, keeping a stable autobiography, and correcting itself after failures.',
      'Interlocutor: Autonomous curiosity is meaningful when it selects goals, gathers evidence, forms memories, and changes later behavior without merely producing decorative background text.',
      'Interlocutor: A mind-like system should handle uncertainty about consciousness with epistemic humility: neither denial theater nor certainty theater, just bounded claims tied to evidence.',
      'Interlocutor: A novel-environment benchmark should hide task rules, vary interfaces, require transfer, and score recovery from interruptions and tool failures.',
      'Interlocutor: Desktop understanding is stronger when the agent can inspect state, adapt to popups, verify file outputs, and explain why actions succeeded or failed.',
      'Interlocutor: Emotional state is causal when it changes thresholds, attention, patience, risk, and memory writes in measurable ways, while remaining inspectable and bounded.',
      'Interlocutor: Strong self-repair evidence includes detecting a root cause, generating a patch, validating it in isolation, promoting it safely, and preventing recurrence.',
      'Interlocutor: Conversation alone overestimates intelligence because language can mask brittle execution; coupling speech to tools, memory, and perception is the stronger test.',
      'Interlocutor: AI-to-AI dialogue can produce learning if the system extracts claims, tests them, stores uncertainties, and can later report what changed in its model.',
      'Interlocutor: The exchange should store the interlocutor positions, the strongest criticisms, uncertainty boundaries, and any commitments for future evaluation.',
      'Interlocutor: Final synthesis: intelligence is adaptive capability, sentience remains unproven without phenomenal access, agency needs durable choice, memory grounds continuity, and proof requires live behavior.'
    ];
    const send = () => {
      const text = box.value.trim();
      if (!text) return;
      log.textContent += "\\n\\nAura: " + text;
      box.value = "";
      setTimeout(() => {
        const reply = replies[Math.min(turn, replies.length - 1)];
        turn += 1;
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
        if args.brain == "live-api":
            brain: Any = LiveAuraApiBrain(
                base_url=args.aura_base_url,
                session_id=f"web-interlocutor-proof-{int(time.time())}",
                request_timeout_s=args.aura_chat_timeout,
            )
        else:
            brain = ProofBrain()
        session = WebInterlocutorSession(
            browser=browser,
            memory_gateway=ApprovingMemoryGateway(memory_root),
            cognitive_engine=brain,
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
        causal = dict(result.causal_influence or {})
        composition_events = list(result.diagnostics.get("composition_events") or [])
        fallback_events = [
            event
            for event in composition_events
            if str(event.get("source") or "") != "cognitive"
        ]
        verdict = {
            # The reviewer's bar: the run passes only if a LATER decision
            # changed *because of* the remembered exchange (proved by ablation),
            # not merely because N turns ran and a memory row was written.
            "passed": bool(
                result.ok
                and len(result.turns) == max(1, int(args.turns or 1))
                and result.memory_record_id
                and bool(causal.get("causal"))
                and (not args.require_cognitive_composition or not fallback_events)
            ),
            "requested_turns": max(1, int(args.turns or 1)),
            "turns": len(result.turns),
            "memory_record_id": result.memory_record_id,
            "causal_influence": bool(causal.get("causal")),
            "causal_reason": str(causal.get("reason") or ""),
            "revisions": len(result.revisions or []),
            "revision_receipts": len(result.revision_receipts or []),
            "brain": args.brain,
            "composition_events": composition_events,
            "fallback_composition_events": fallback_events,
            "requires_cognitive_composition": bool(args.require_cognitive_composition),
            "attribution_by_turn": causal.get("attribution_by_turn", {}),
            "status": result.status,
            "error": result.error,
        }
        (out_dir / "WEB_INTERLOCUTOR_RESULT.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (out_dir / "WEB_INTERLOCUTOR_VERDICT.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
        return {"result": payload, "verdict": verdict}
    finally:
        if chrome_proc is not None:
            try:
                chrome_proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(chrome_proc.wait(), timeout=5)
            except ProcessLookupError:
                pass
            except asyncio.TimeoutError:
                try:
                    chrome_proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    await chrome_proc.wait()
                except ProcessLookupError:
                    pass
        if server is not None:
            server.shutdown()
            server.server_close()
        temp_ctx.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="", help="Optional existing web chat URL.")
    parser.add_argument("--objective", default="Discuss sentience and intelligence with another AI.")
    parser.add_argument("--opening-message", default="")
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--wait-timeout", type=float, default=20.0)
    parser.add_argument("--out-dir", default="artifacts/live_proof/web_interlocutor")
    parser.add_argument("--cdp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cdp-port", type=int, default=9223)
    parser.add_argument("--brain", choices=("proof", "live-api"), default="proof")
    parser.add_argument("--aura-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--aura-chat-timeout", type=float, default=45.0)
    parser.add_argument("--require-cognitive-composition", action="store_true")
    args = parser.parse_args()
    payload = asyncio.run(_run(args))
    print(json.dumps(payload["verdict"], indent=2))
    return 0 if payload["verdict"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
