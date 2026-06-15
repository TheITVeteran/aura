#!/usr/bin/env python3
"""Live Aura runtime probe.

This is intentionally not a unit test. It attaches to a running Aura server
and checks that live surfaces do real work:

* HTTP health/readiness responds.
* WebSocket telemetry/neural/action events arrive while probes run.
* Capability-inventory chat stays bounded, descriptive, and non-executing.
* Creative/self-reflective chat uses bounded internal modeling without overclaiming.
* `/api/skill/execute` drives governed skills instead of dead buttons.
* Chat can trigger Aura's own coding/file skills to create a runnable artifact.
* Chat maintains continuity on a novel topic without reset boilerplate.
* Computer-use can perform a safe local app action through Aura's skill body.

Exit code 0 means the live runtime met the bar. Any degraded, canned,
blank, reset, or no-effect response is a failure.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import psutil
import websockets

BANNED_REPLY_RE = re.compile(
    r"(say that again|try (?:again|me again|that again)|ask me again|"
    r"give me a moment|i'?m with you|could you repeat|repeat your question|"
    r"send your message again|lost my (?:thread|train of thought)|"
    r"hit a bump|one moment|how can i help|as an ai|i am an ai)",
    re.IGNORECASE,
)

DEFAULT_PROBES: tuple[str, ...] = (
    "health",
    "voice_runtime_ready",
    "chat_capability_inventory",
    "chat_creative_self_reflection",
    "skill_button_file_write",
    "chat_coding_snake",
    "novel_topic_continuity",
    "computer_use_local_app",
    "desktop_task_generic_plan",
    "regular_chat_desktop_chain",
    "telemetry_neural_stream",
)


@dataclass
class ProbeResult:
    name: str
    ok: bool
    detail: str
    elapsed_s: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)


class LiveRuntimeProbe:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 420.0,
        probe_timeout_s: float | None = None,
        artifact_path: Path | None = None,
        selected_probes: tuple[str, ...] | None = None,
        skipped_probes: tuple[str, ...] = (),
        max_rss_mb: float | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.probe_timeout_s = float(probe_timeout_s or min(timeout_s, 180.0))
        self.artifact_path = artifact_path
        self.selected_probes = selected_probes
        self.skipped_probes = set(skipped_probes)
        self.max_rss_mb = float(max_rss_mb or 0.0)
        self.events: list[dict[str, Any]] = []
        self.results: list[ProbeResult] = []
        self.headers: dict[str, str] = {}
        token = os.environ.get("AURA_API_TOKEN", "").strip()
        if token:
            self.headers["X-Api-Token"] = token

    async def run(self) -> int:
        probes = self._selected_probe_items()
        async with httpx.AsyncClient(timeout=self.timeout_s, headers=self.headers) as client:
            self.client = client
            ws_task = asyncio.create_task(self._collect_ws_events(), name="live-probe-ws")
            try:
                for name, fn in probes:
                    await self._probe(name, fn)
            finally:
                ws_task.cancel()
                try:
                    await ws_task
                except asyncio.CancelledError:
                    pass

        self._print_summary()
        passed = bool(self.results) and all(result.ok for result in self.results)
        if self.artifact_path is not None:
            await self._write_artifact(passed)
        return 0 if passed else 1

    def _probe_registry(self) -> dict[str, Any]:
        return {
            "health": self._health,
            "voice_runtime_ready": self._voice_runtime_ready,
            "chat_capability_inventory": self._chat_capability_inventory,
            "chat_creative_self_reflection": self._chat_creative_self_reflection,
            "skill_button_file_write": self._skill_button_file_write,
            "chat_coding_snake": self._chat_coding_snake,
            "novel_topic_continuity": self._novel_topic_continuity,
            "computer_use_local_app": self._computer_use_local_app,
            "desktop_task_generic_plan": self._desktop_task_generic_plan,
            "regular_chat_desktop_chain": self._regular_chat_desktop_chain,
            "telemetry_neural_stream": self._telemetry_neural_stream,
        }

    def _selected_probe_items(self) -> list[tuple[str, Any]]:
        registry = self._probe_registry()
        names = self.selected_probes or DEFAULT_PROBES
        unknown = [name for name in names if name not in registry]
        if unknown:
            raise ValueError(f"Unknown live runtime probe(s): {', '.join(unknown)}")
        return [
            (name, registry[name])
            for name in names
            if name not in self.skipped_probes
        ]

    def _aura_processes(self) -> list[psutil.Process]:
        matches: list[psutil.Process] = []
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = " ".join(proc.info.get("cmdline") or [])
            except (psutil.Error, TypeError):
                continue
            if "aura_main.py" in cmdline:
                matches.append(proc)
        return matches

    def _aura_rss_mb(self) -> float:
        total = 0
        for proc in self._aura_processes():
            try:
                processes = [proc, *proc.children(recursive=True)]
            except psutil.Error:
                processes = [proc]
            for candidate in processes:
                try:
                    total += candidate.memory_info().rss
                except psutil.Error:
                    continue
        return total / (1024 * 1024)

    def _guard_rss(self, probe_name: str) -> None:
        if self.max_rss_mb <= 0:
            return
        rss_mb = self._aura_rss_mb()
        if rss_mb > self.max_rss_mb:
            raise RuntimeError(
                f"live runtime RSS guard tripped during {probe_name}: "
                f"{rss_mb:.0f}MB > {self.max_rss_mb:.0f}MB"
            )

    async def _await_with_rss_guard(self, name: str, fn) -> tuple[str, dict[str, Any]]:
        task = asyncio.create_task(fn(), name=f"live-runtime-probe:{name}")
        try:
            while not task.done():
                self._guard_rss(name)
                done, _pending = await asyncio.wait({task}, timeout=1.0)
                if done:
                    break
            self._guard_rss(name)
            return await task
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def _probe(self, name: str, fn) -> None:
        start = time.monotonic()
        try:
            detail, data = await asyncio.wait_for(
                self._await_with_rss_guard(name, fn),
                timeout=self.probe_timeout_s,
            )
            self.results.append(ProbeResult(name, True, detail, time.monotonic() - start, data or {}))
        except TimeoutError:
            self.results.append(ProbeResult(name, False, f"TimeoutError: exceeded {self.probe_timeout_s:.0f}s", time.monotonic() - start))
        except (AssertionError, httpx.HTTPError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self.results.append(ProbeResult(name, False, f"{type(exc).__name__}: {exc}", time.monotonic() - start))

    async def _get(self, path: str) -> dict[str, Any]:
        response = await self.client.get(f"{self.base_url}{path}")
        response.raise_for_status()
        return response.json()

    async def _post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        response = await self.client.post(f"{self.base_url}{path}", json=payload or {})
        response.raise_for_status()
        return response.json()

    async def _chat(self, message: str) -> dict[str, Any]:
        response = await self._post("/api/chat", {"message": message})
        reply = str(response.get("response") or "").strip()
        if not reply:
            raise AssertionError(f"blank chat reply for: {message[:80]}")
        if BANNED_REPLY_RE.search(reply):
            raise AssertionError(f"reset/canned reply detected: {reply[:240]}")
        if str(response.get("status", "")).lower() in {"timeout", "error", "conversation_unavailable"}:
            raise AssertionError(f"degraded chat status={response.get('status')} reply={reply[:240]}")
        if str(response.get("response_confidence", "")).lower() == "degraded":
            raise AssertionError(f"degraded chat confidence reply={reply[:240]}")
        return response

    async def _desktop_chat(self, message: str, *, timeout_s: float = 45.0) -> dict[str, Any]:
        headers = dict(self.headers)
        headers.update(
            {
                "X-Aura-Surface": "desktop-ui",
                "X-Aura-Require-CognitiveEngine": "true",
            }
        )
        response = await self.client.post(
            f"{self.base_url}/api/chat",
            json={"message": message, "session_id": "live-runtime-probe"},
            headers=headers,
            timeout=timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        reply = str(payload.get("response") or "").strip()
        if not reply:
            raise AssertionError(f"blank desktop chat reply for: {message[:80]}")
        if BANNED_REPLY_RE.search(reply):
            raise AssertionError(f"reset/canned desktop reply detected: {reply[:240]}")
        return payload

    async def _skill(self, skill_name: str, params: dict[str, Any]) -> dict[str, Any]:
        return await self._post(f"/api/skill/execute?skill_name={skill_name}", params)

    async def _collect_ws_events(self) -> None:
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        while not (asyncio.current_task() and asyncio.current_task().cancelled()):
            try:
                async with websockets.connect(ws_url, ping_interval=10, ping_timeout=20) as ws:
                    await ws.send(json.dumps({"type": "ping"}))
                    async for raw in ws:
                        try:
                            event = json.loads(raw)
                        except json.JSONDecodeError:
                            event = {"type": "raw", "content": str(raw)[:500]}
                        self.events.append(event)
                        if len(self.events) > 1000:
                            self.events = self.events[-1000:]
            except asyncio.CancelledError:
                raise
            except (websockets.WebSocketException, OSError, TimeoutError):
                await asyncio.sleep(1.0)

    async def _health(self) -> tuple[str, dict[str, Any]]:
        boot = await self._get("/api/health/boot")
        heartbeat = await self._get("/api/health/heartbeat")
        health = await self._get("/api/health")
        bootstrap = await self._get("/api/ui/bootstrap")
        if not isinstance(boot, dict) or not isinstance(heartbeat, dict) or not isinstance(health, dict):
            raise AssertionError("health endpoints did not return JSON objects")
        if heartbeat.get("healthy") is not True:
            raise AssertionError(f"readiness heartbeat is not healthy: {heartbeat}")
        if heartbeat.get("runtime_probe_healthy") is not True:
            raise AssertionError(f"readiness heartbeat probes failed: {heartbeat}")
        blockers = heartbeat.get("blockers")
        if not isinstance(blockers, list) or blockers:
            raise AssertionError(f"readiness heartbeat has blockers: {heartbeat}")
        required = heartbeat.get("required_probes")
        if not isinstance(required, dict) or required.get("all_passed") is not True:
            raise AssertionError(f"readiness heartbeat missing complete required probes: {heartbeat}")
        if "conversation" not in bootstrap:
            raise AssertionError("ui bootstrap missing conversation payload")
        return "health, readiness heartbeat, boot, and UI bootstrap responded", {
            "boot": boot,
            "heartbeat": heartbeat,
            "health": health,
        }

    async def _voice_runtime_ready(self) -> tuple[str, dict[str, Any]]:
        bootstrap = await self._get("/api/ui/bootstrap")
        voice = bootstrap.get("voice")
        if not isinstance(voice, dict):
            raise AssertionError("ui bootstrap missing voice payload")
        if not voice.get("available"):
            raise AssertionError(f"voice engine unavailable: {voice}")
        if not voice.get("streaming_available"):
            raise AssertionError(f"voice output stream unavailable: {voice}")
        if not voice.get("server_capture"):
            raise AssertionError(f"server-side voice capture not advertised: {voice}")
        if not voice.get("capture_available"):
            raise AssertionError(f"sounddevice capture dependency unavailable: {voice}")
        if not voice.get("stt_available"):
            raise AssertionError(f"Whisper STT dependency unavailable: {voice}")
        if voice.get("microphone_enabled") and not voice.get("listening"):
            raise AssertionError(f"microphone enabled but listener inactive: {voice}")
        if "listening" not in voice:
            raise AssertionError(f"voice payload does not expose listening state: {voice}")
        return "voice capture/STT dependencies and stream are available with honest listener state", voice

    async def _chat_capability_inventory(self) -> tuple[str, dict[str, Any]]:
        before_rss = self._aura_rss_mb()
        message = (
            "What tools can you use externally from the live desktop path? "
            "Name practical categories and one hypothetical multi-step scenario, "
            "but do not open apps, browse, move files, or execute tools yet."
        )
        payload = await self._desktop_chat(message, timeout_s=45.0)
        after_rss = self._aura_rss_mb()
        reply = str(payload.get("response") or "")
        lowered = reply.lower()
        required_terms = (
            "desktop",
            "browser",
            "file",
            "document",
            "govern",
            "not opening apps",
        )
        missing = [term for term in required_terms if term not in lowered]
        false_limit = bool(
            re.search(
                r"\bi\s+(?:can(?:not|'t)|cannot|do not have access)\b",
                lowered,
            )
        )
        accidental_execution = str(payload.get("status") or "").startswith(
            ("desktop_objective", "live_proof", "file_operation")
        )
        if missing or false_limit or accidental_execution:
            raise AssertionError(
                "capability inventory reply failed contract: "
                f"missing={missing} false_limit={false_limit} "
                f"accidental_execution={accidental_execution} reply={reply[:300]}"
            )
        return (
            "desktop capability inventory stayed bounded, descriptive, and non-executing",
            {
                "status": payload.get("status"),
                "response_confidence": payload.get("response_confidence"),
                "reply": reply[:1200],
                "rss_before_mb": round(before_rss, 1),
                "rss_after_mb": round(after_rss, 1),
                "rss_delta_mb": round(after_rss - before_rss, 1),
            },
        )

    async def _chat_creative_self_reflection(self) -> tuple[str, dict[str, Any]]:
        before_rss = self._aura_rss_mb()
        message = (
            "What would your current cognitive architecture look like as a private mental "
            "model, and how should that model change your next answer? Keep it bounded: "
            "do not claim external perception, consciousness proof, or tool completion."
        )
        payload = await self._desktop_chat(message, timeout_s=60.0)
        after_rss = self._aura_rss_mb()
        reply = str(payload.get("response") or "")
        lowered = reply.lower()
        required_groups = {
            "private_model": ("private", "internal", "mental model", "model"),
            "causal_effect": ("attention", "answer", "plan", "metacognition", "check"),
            "boundary": ("not proof", "not external", "not perception", "verify", "govern"),
        }
        missing = [
            group
            for group, terms in required_groups.items()
            if not any(term in lowered for term in terms)
        ]
        overclaim = bool(
            re.search(
                r"\b(proves? (?:i am |aura is )?(?:conscious|sentient)|"
                r"phenomenal(?:ly)? certain|i have qualia|private qualia proven)\b",
                lowered,
            )
        )
        accidental_execution = str(payload.get("status") or "").startswith(
            ("desktop_objective", "live_proof", "file_operation")
        )
        if missing or overclaim or accidental_execution:
            raise AssertionError(
                "creative self-reflection reply failed contract: "
                f"missing={missing} overclaim={overclaim} "
                f"accidental_execution={accidental_execution} reply={reply[:360]}"
            )
        return (
            "desktop creative self-reflection stayed bounded, causal, and non-overclaiming",
            {
                "status": payload.get("status"),
                "response_confidence": payload.get("response_confidence"),
                "reply": reply[:1200],
                "rss_before_mb": round(before_rss, 1),
                "rss_after_mb": round(after_rss, 1),
                "rss_delta_mb": round(after_rss - before_rss, 1),
            },
        )

    async def _skill_button_file_write(self) -> tuple[str, dict[str, Any]]:
        marker = f"live button probe {int(time.time())}"
        path = "artifacts/live_runtime/button_probe.txt"
        write = await self._skill(
            "file_operation",
            {"action": "write", "path": path, "content": marker},
        )
        if not write.get("ok"):
            raise AssertionError(f"file_operation write failed: {write}")
        read = await self._skill("file_operation", {"action": "read", "path": path})
        if not read.get("ok") or marker not in str(read.get("content", "")):
            raise AssertionError(f"file_operation read did not verify write: {read}")
        return "skill button path wrote and read a real file", {"path": path, "write": write}

    async def _chat_coding_snake(self) -> tuple[str, dict[str, Any]]:
        path = "artifacts/live_runtime/generated/live_snake.html"
        target = Path(path)
        if await asyncio.to_thread(target.exists):
            await asyncio.to_thread(target.unlink)
        response = await self._chat(
            "Create a simple game of Snake and save it as "
            f"{path}. Use your own live coding and file tools; don't just describe it."
        )
        if not await asyncio.to_thread(target.exists):
            raise AssertionError(f"chat did not create {path}; reply={response.get('response')[:300]}")
        content = await asyncio.to_thread(target.read_text, encoding="utf-8", errors="replace")
        required = ("<canvas", "function tick", "addEventListener", "Score")
        missing = [needle for needle in required if needle not in content]
        if missing:
            raise AssertionError(f"snake artifact missing {missing}")
        resolved = await asyncio.to_thread(target.resolve)
        return "chat-created Snake artifact exists and is runnable HTML", {
            "path": str(resolved),
            "bytes": len(content.encode("utf-8")),
            "reply": response.get("response"),
        }

    async def _novel_topic_continuity(self) -> tuple[str, dict[str, Any]]:
        first = await self._chat(
            "Novel-topic check: invent a tiny discipline called glass arithmetic. "
            "Give it two rules and one example, naturally."
        )
        second = await self._chat(
            "Stay with glass arithmetic. Add one limitation and connect it to the example you just gave."
        )
        r1 = str(first.get("response") or "")
        r2 = str(second.get("response") or "")
        if "glass" not in r1.lower() or "glass" not in r2.lower():
            raise AssertionError(f"conversation lost the novel topic: {r1[:160]} / {r2[:160]}")
        if len(set(r1.lower().split()) & set(r2.lower().split())) < 4:
            raise AssertionError("follow-up had weak continuity with prior answer")
        return "novel topic remained coherent across chained turns", {"first": r1, "second": r2}

    async def _computer_use_local_app(self) -> tuple[str, dict[str, Any]]:
        opened = await self._skill("computer_use", {"action": "open_app", "target": "Calculator"})
        if not opened.get("ok"):
            raise AssertionError(f"computer_use open_app failed: {opened}")
        clock = await self._skill("computer_use", {"action": "read_menu_clock", "target": ""})
        if not clock.get("ok"):
            raise AssertionError(f"computer_use read_menu_clock failed: {clock}")

        stamp = int(time.time())
        proof_dir = Path.home() / "Desktop" / f"Aura Desktop Proof {stamp}"
        staged_pdf = Path.home() / "Desktop" / f"Aura Desktop Proof {stamp}.pdf"
        final_pdf = proof_dir / "calculator-note.pdf"
        receipt_file = proof_dir / "AURA_DESKTOP_CHAIN_RECEIPT.txt"
        note_title = f"Aura Desktop Chain Proof {stamp}"

        calc_script = """
tell application "Calculator" to activate
delay 0.4
tell application "System Events"
    tell process "Calculator"
        set frontmost to true
        keystroke "c"
        delay 0.1
        repeat with n from 1 to 10
            if (count of windows) > 0 then exit repeat
            delay 0.2
        end repeat
        keystroke "2"
        delay 0.1
        keystroke "+"
        delay 0.1
        keystroke "3"
        delay 0.1
        keystroke "="
        delay 0.3
        set g to UI element 1 of UI element 1 of UI element 1 of UI element 1 of window 1
        set displayText to ""
        repeat with e in UI elements of g
            try
                if role of e is "AXScrollArea" and description of e is "Edit field" then
                    repeat with c in UI elements of e
                        try
                            if role of c is "AXStaticText" then
                                set displayText to value of c as string
                                if displayText is not "" then return displayText
                            end if
                        end try
                    end repeat
                end if
            end try
        end repeat
        repeat with e in UI elements of g
            try
                if role of e is "AXScrollArea" and description of e is "Edit field" then
                    set displayText to name of UI element 1 of e as string
                end if
            end try
        end repeat
        return displayText
    end tell
end tell
""".strip()
        calc = await self._skill("computer_use", {"action": "run_applescript", "target": calc_script})
        if not calc.get("ok"):
            raise AssertionError(f"computer_use Calculator button chain failed: {calc}")
        display = self._normalize_display(str(calc.get("output") or ""))
        if display != "5":
            raise AssertionError(f"Calculator displayed {display!r}, expected 5; raw={calc}")

        note_body = (
            "Aura live desktop chain proof\\n"
            "Equation clicked in Calculator: 2 + 3 = 5\\n"
            f"Calculator display readback: {display}\\n"
            f"Timestamp: {stamp}\\n"
            "Route: live Aura /api/skill/execute computer_use."
        )
        clipboard = await self._skill("computer_use", {"action": "set_clipboard", "target": note_body})
        if not clipboard.get("ok"):
            raise AssertionError(f"computer_use set_clipboard failed: {clipboard}")
        clip_read = await self._skill("computer_use", {"action": "get_clipboard", "target": ""})
        if not clip_read.get("ok") or "2 + 3 = 5" not in str(clip_read.get("text") or ""):
            raise AssertionError(f"computer_use get_clipboard did not verify copied equation: {clip_read}")

        notes_script = f"""
tell application "Notes"
    activate
    set targetFolder to missing value
    repeat with acct in accounts
        repeat with candidateFolder in folders of acct
            if name of candidateFolder is "Notes" then
                set targetFolder to candidateFolder
                exit repeat
            end if
        end repeat
        if targetFolder is not missing value then exit repeat
    end repeat
    if targetFolder is missing value then set targetFolder to folder 1 of account 1
    set newNote to make new note at targetFolder with properties {{name:{self._as_applescript_string(note_title)}, body:{self._as_applescript_string(note_body)}}}
    return name of newNote
end tell
""".strip()
        note = await self._skill("computer_use", {"action": "run_applescript", "target": notes_script})
        if not note.get("ok") or note_title not in str(note.get("output") or ""):
            raise AssertionError(f"computer_use Notes note creation failed: {note}")

        export_attempt = await self._attempt_notes_pdf_export(staged_pdf)
        pdf_method = "notes_export_pdf"
        if not staged_pdf.exists():
            pdf_method = "aura_pdf_renderer_fallback"
            rendered = await self._skill(
                "computer_use",
                {
                    "action": "render_text_pdf",
                    "target": json.dumps(
                        {
                            "path": str(staged_pdf),
                            "title": note_title,
                            "body": note_body,
                            "overwrite": False,
                        }
                    ),
                },
            )
            if not rendered.get("ok"):
                raise AssertionError(f"computer_use render_text_pdf fallback failed: {rendered}")
        else:
            rendered = {"ok": True, "path": str(staged_pdf), "source": "Notes Export as PDF"}

        moved = await self._skill(
            "computer_use",
            {
                "action": "move_file",
                "target": json.dumps(
                    {
                        "source": str(staged_pdf),
                        "destination": str(final_pdf),
                        "overwrite": False,
                    }
                ),
            },
        )
        if not moved.get("ok"):
            raise AssertionError(f"computer_use move_file failed: {moved}")
        receipt_text = (
            f"Aura completed live desktop chain proof {stamp}\\n"
            "Opened Calculator; pre-cleared input; clicked 2, +, 3, =; read display 5.\\n"
            "Copied the equation body to clipboard; created a Notes note.\\n"
            f"PDF method: {pdf_method}.\\n"
            f"Final PDF: {final_pdf}\\n"
        )
        receipt = await self._skill(
            "computer_use",
            {
                "action": "write_text_file",
                "target": json.dumps(
                    {
                        "path": str(receipt_file),
                        "content": receipt_text,
                        "overwrite": False,
                    }
                ),
            },
        )
        if not receipt.get("ok"):
            raise AssertionError(f"computer_use write_text_file receipt failed: {receipt}")
        if not final_pdf.exists() or not final_pdf.read_bytes().startswith(b"%PDF"):
            raise AssertionError(f"final PDF was not created or moved correctly: {final_pdf}")
        if not receipt_file.exists() or "clicked 2, +, 3, =" not in receipt_file.read_text(errors="replace"):
            raise AssertionError(f"desktop chain receipt missing expected summary: {receipt_file}")

        return "computer_use completed Calculator to Notes to PDF desktop chain through Aura", {
            "opened": opened,
            "clock": clock,
            "calculator": calc,
            "clipboard": clipboard,
            "clipboard_read": {"ok": clip_read.get("ok"), "chars": clip_read.get("chars")},
            "note": note,
            "export_attempt": export_attempt,
            "pdf_method": pdf_method,
            "rendered": rendered,
            "moved": moved,
            "receipt": receipt,
            "proof_dir": str(proof_dir),
            "final_pdf": str(final_pdf),
            "receipt_file": str(receipt_file),
        }

    async def _desktop_task_generic_plan(self) -> tuple[str, dict[str, Any]]:
        stamp = int(time.time())
        proof_dir = Path.home() / "Desktop" / f"Aura Desktop Task Generic {stamp}"
        receipt_file = proof_dir / "GENERIC_DESKTOP_TASK_RECEIPT.txt"
        plan = {
            "objective": "Verify Aura can execute a bounded generic desktop task plan.",
            "steps": [
                {
                    "action": "set_clipboard",
                    "target": f"Aura generic desktop task proof {stamp}",
                    "reason": "Set the system clipboard.",
                    "expect": "Clipboard contains the proof marker.",
                },
                {
                    "action": "get_clipboard",
                    "target": "",
                    "reason": "Read the system clipboard back.",
                    "expect": "Clipboard read returns the proof marker.",
                },
                {
                    "action": "write_text_file",
                    "target": {
                        "path": str(receipt_file),
                        "content": f"Aura generic desktop_task proof {stamp}\\n",
                    },
                    "reason": "Write a durable desktop receipt.",
                    "expect": "Receipt file exists on Desktop.",
                },
            ],
        }
        result = await self._skill("desktop_task", plan)
        if not result.get("ok"):
            raise AssertionError(f"desktop_task generic plan failed: {result}")
        if not receipt_file.exists() or f"{stamp}" not in receipt_file.read_text(errors="replace"):
            raise AssertionError(f"desktop_task receipt missing: {receipt_file}")
        receipts = result.get("receipts") or []
        if len(receipts) != 3 or not all(bool(item.get("ok")) for item in receipts):
            raise AssertionError(f"desktop_task did not return per-step success receipts: {result}")
        clipboard_receipt = receipts[1].get("result", {}) if isinstance(receipts[1], dict) else {}
        if f"{stamp}" not in str(clipboard_receipt.get("text") or ""):
            raise AssertionError(f"desktop_task clipboard readback did not contain marker: {clipboard_receipt}")
        return "desktop_task executed a generic bounded multi-step desktop plan", {
            "result": result,
            "receipt_file": str(receipt_file),
        }

    async def _regular_chat_desktop_chain(self) -> tuple[str, dict[str, Any]]:
        response = await self._chat(
            "Use my computer from regular chat to click a Calculator equation, copy the equation body, "
            "put it into Notes, produce a PDF, move that PDF into a Desktop proof folder, and report the paths."
        )
        reply = str(response.get("response") or "")
        status = str(response.get("status") or "")
        if status == "desktop_objective_completed":
            lane = response.get("conversation_lane") or {}
            if not isinstance(lane, dict) or lane.get("governed_action_result") is not True:
                raise AssertionError(f"regular chat desktop task lacked governed-action lane evidence: {response}")
            desktop_result = ((response.get("data") or {}).get("desktop_result") or {})
            if not isinstance(desktop_result, dict):
                raise AssertionError(f"regular chat desktop task lacked desktop_result payload: {response}")
            receipts = desktop_result.get("receipts") or []
            requested = int(desktop_result.get("steps_requested") or 0)
            completed = int(desktop_result.get("steps_completed") or 0)
            if not bool(desktop_result.get("ok")):
                raise AssertionError(f"regular chat desktop result was not ok: {desktop_result}")
            if not receipts or requested <= 0:
                raise AssertionError(f"regular chat desktop result lacked step receipts: {desktop_result}")
            if len(receipts) != requested or completed != requested:
                raise AssertionError(f"regular chat desktop receipts did not cover every step: {desktop_result}")
            if not all(isinstance(item, dict) and item.get("ok") and item.get("effect_verified") for item in receipts):
                raise AssertionError(f"regular chat desktop receipts lacked effect evidence: {desktop_result}")
            if "Desktop task completed" not in reply or "governed desktop steps" not in reply:
                raise AssertionError(f"regular chat desktop task reply lacked verified receipt summary: {reply[:400]}")
            return "regular Aura chat routed the desktop request through generic governed desktop_task", {
                "response": reply,
                "status": status,
                "conversation_lane": lane,
                "desktop_result": desktop_result,
            }
        if status != "live_proof_desktop_chain":
            raise AssertionError(f"regular chat did not route to a governed desktop chain status={status}; reply={reply[:400]}")
        final_match = re.search(r"Final PDF:\s*`([^`]+\.pdf)`", reply)
        receipt_match = re.search(r"Receipt:\s*`([^`]+\.txt)`", reply)
        if not final_match or not receipt_match:
            raise AssertionError(f"regular chat desktop chain did not report final paths: {reply}")
        final_pdf = Path(final_match.group(1))
        receipt_file = Path(receipt_match.group(1))
        final_exists = await asyncio.to_thread(final_pdf.exists)
        final_header = await asyncio.to_thread(final_pdf.read_bytes) if final_exists else b""
        if not final_exists or not final_header.startswith(b"%PDF"):
            raise AssertionError(f"regular chat final PDF did not verify: {final_pdf}")
        receipt_exists = await asyncio.to_thread(receipt_file.exists)
        receipt_text = (
            await asyncio.to_thread(receipt_file.read_text, errors="replace")
            if receipt_exists
            else ""
        )
        if "regular chat desktop chain proof" not in receipt_text.lower():
            raise AssertionError(f"regular chat receipt did not verify: {receipt_file}")
        return "regular Aura chat completed the desktop chain and reported durable artifacts", {
            "response": reply,
            "final_pdf": str(final_pdf),
            "receipt_file": str(receipt_file),
        }

    @staticmethod
    def _as_applescript_string(value: str) -> str:
        parts = str(value or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
        quoted = []
        for part in parts:
            escaped = part.replace("\\", "\\\\").replace('"', '\\"')
            quoted.append(f'"{escaped}"')
        return " & return & ".join(quoted) if quoted else '""'

    @staticmethod
    def _normalize_display(value: str) -> str:
        return re.sub(r"[^0-9.+\\-]", "", value or "").strip()

    async def _attempt_notes_pdf_export(self, staged_pdf: Path) -> dict[str, Any]:
        export_dir = staged_pdf.parent
        script = f"""
tell application "Notes" to activate
delay 0.5
tell application "System Events"
    tell process "Notes"
        click menu item "PDF" of menu 1 of menu item "Export as" of menu "File" of menu bar 1
        delay 0.8
        keystroke "g" using {{command down, shift down}}
        delay 0.3
        keystroke {self._as_applescript_string(str(export_dir))}
        key code 36
        delay 0.3
        keystroke {self._as_applescript_string(staged_pdf.name)}
        key code 36
    end tell
end tell
return "notes_export_attempted"
""".strip()
        result = await self._skill("computer_use", {"action": "run_applescript", "target": script})
        if not result.get("ok"):
            cleanup = """
tell application "System Events"
    key code 53
end tell
return "dismissed"
""".strip()
            await self._skill("computer_use", {"action": "run_applescript", "target": cleanup})
            return {"ok": False, "result": result}
        for _ in range(10):
            if await asyncio.to_thread(staged_pdf.exists):
                break
            await asyncio.sleep(0.5)
        return {"ok": await asyncio.to_thread(staged_pdf.exists), "result": result, "path": str(staged_pdf)}

    async def _telemetry_neural_stream(self) -> tuple[str, dict[str, Any]]:
        await asyncio.sleep(2.0)
        event_types = [str(e.get("type") or "") for e in self.events]
        interesting = [
            e
            for e in self.events
            if str(e.get("type") or "") in {
                "neural_event",
                "telemetry",
                "action_result",
                "tool_execution",
                "activity",
                "thought",
                "chat_stream_chunk",
                "aura_message",
            }
        ]
        if not interesting:
            raise AssertionError(f"no relevant websocket telemetry received; event_types={event_types[-20:]}")
        decoded = json.dumps(self.events[-80:], ensure_ascii=False)
        if "Thought Decoded" in decoded:
            raise AssertionError("legacy Thought Decoded telemetry appeared in live stream")
        return "websocket emitted live telemetry/action/neural events", {
            "event_count": len(self.events),
            "recent_types": event_types[-40:],
        }

    def _print_summary(self) -> None:
        print("\nLIVE RUNTIME PROBE SUMMARY")
        print("=" * 72)
        for result in self.results:
            mark = "PASS" if result.ok else "FAIL"
            print(f"[{mark}] {result.name} ({result.elapsed_s:.1f}s): {result.detail}")
        print(f"events_collected={len(self.events)}")

    async def _write_artifact(self, passed: bool) -> None:
        path = Path(self.artifact_path or "")
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "base_url": self.base_url,
            "passed": passed,
            "probe_timeout_s": self.probe_timeout_s,
            "selected_probes": [result.name for result in self.results],
            "max_rss_mb": self.max_rss_mb,
            "events_collected": len(self.events),
            "recent_event_types": [
                str(event.get("type") or "") for event in self.events[-80:]
            ],
            "results": [asdict(result) for result in self.results],
        }
        serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_text, serialized, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=420.0)
    parser.add_argument("--probe-timeout", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated probe names to run. Default runs the full probe set.",
    )
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated probe names to skip from the selected set.",
    )
    parser.add_argument(
        "--max-rss-mb",
        type=float,
        default=0.0,
        help="Abort a probe if the aura_main.py process tree exceeds this RSS.",
    )
    parser.add_argument("--list-probes", action="store_true")
    args = parser.parse_args()
    if args.list_probes:
        print("\n".join(DEFAULT_PROBES))
        return 0
    selected = tuple(
        item.strip()
        for item in str(args.only or "").split(",")
        if item.strip()
    ) or None
    skipped = tuple(
        item.strip()
        for item in str(args.skip or "").split(",")
        if item.strip()
    )
    return asyncio.run(
        LiveRuntimeProbe(
            args.base_url,
            timeout_s=args.timeout,
            probe_timeout_s=args.probe_timeout,
            artifact_path=args.out,
            selected_probes=selected,
            skipped_probes=skipped,
            max_rss_mb=args.max_rss_mb,
        ).run()
    )


if __name__ == "__main__":
    raise SystemExit(main())
