"""Governed visible-web interlocutor sessions.

This capability owns the general loop for talking with another web AI or
chat surface in the user's real browser:

1. open or attach to a visible page,
2. focus a semantic text-entry surface,
3. send a message,
4. wait for stable new page text,
5. decide the next message from the transcript,
6. persist learned material through MemoryWriteGateway.

The browser adapter is deliberately generic. It does not know "ChatGPT" or
"Gemini" workflows; it works with visible editable fields, page text deltas,
and receiptable browser state.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import urllib.parse
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from core.capabilities.browser_controller import get_browser_controller
from core.runtime.desktop_action_gateway import get_desktop_action_gateway
from core.runtime.errors import record_degradation
from core.runtime.gateways import MemoryWriteRequest
from core.runtime.network_gateway import get_network_gateway
from core.runtime.task_ownership import create_tracked_task

logger = logging.getLogger("Aura.WebInterlocutor")

_MAX_PAGE_TEXT_CHARS = 24_000
_MAX_REPLY_CHARS = 8_000
_MIN_OUTBOUND_MESSAGE_CHARS = 24
_DEFAULT_WAIT_S = 45.0
_DEFAULT_STABLE_POLLS = 2


def _mark_web_interlocutor_progress(source: str) -> None:
    try:
        from core.runtime.liveness import mark_runtime_service_progress

        mark_runtime_service_progress(source)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        pass


def _run_governed_applescript(script: str, *, source: str, timeout: float) -> dict[str, Any]:
    from core.governance_context import local_internal_governed_scope

    with local_internal_governed_scope(source, domain="tool_execution"):
        return get_desktop_action_gateway().run_applescript(
            script,
            source=source,
            timeout=timeout,
        )


def _call_in_governed_tool_scope(source: str, func: Any, *args: Any, **kwargs: Any) -> Any:
    from core.governance_context import local_internal_governed_scope

    with local_internal_governed_scope(source, domain="tool_execution"):
        return func(*args, **kwargs)


@dataclass
class BrowserPageSnapshot:
    url: str = ""
    title: str = ""
    text: str = ""
    active_element: str = ""
    editable_count: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def text_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


@dataclass
class WebInterlocutorTurn:
    index: int
    sent: str
    observed_reply: str
    before_hash: str
    after_hash: str
    sent_at: float
    observed_at: float
    effect_verified: bool
    verification: str


@dataclass
class WebInterlocutorResult:
    ok: bool
    target_url: str = ""
    target_title: str = ""
    objective: str = ""
    turns: list[WebInterlocutorTurn] = field(default_factory=list)
    learned_summary: str = ""
    memory_record_id: str = ""
    memory_receipt_id: str = ""
    status: str = "ok"
    error: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    # Causal memory: adjudicated revisions the conversation forced, and the
    # ablation proof that a later decision changed only because of them.
    revisions: list[dict[str, Any]] = field(default_factory=list)
    causal_influence: dict[str, Any] = field(default_factory=dict)
    revision_receipts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["turns"] = [asdict(turn) for turn in self.turns]
        return payload


@dataclass
class WebInterlocutorJob:
    job_id: str
    status: str
    objective: str
    started_at: float
    updated_at: float
    target_url: str = ""
    result: dict[str, Any] | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WebDialogueBrowser(Protocol):
    async def open_or_attach(self, url: str) -> BrowserPageSnapshot:
        ...

    async def snapshot(self) -> BrowserPageSnapshot:
        ...

    async def send_message(self, text: str) -> dict[str, Any]:
        ...


class ChromeCDPDialogueBrowser:
    """Visible Chrome control through the Chrome DevTools Protocol.

    CDP is used only against a local debugging endpoint. It keeps the browser
    visible, focuses a semantic editable element in the page, inserts text
    through Chrome's input domain, and reads page text back from the DOM. This
    avoids the macOS "Allow JavaScript from Apple Events" developer toggle
    while still giving Aura effect evidence from the visible page.
    """

    def __init__(self, *, endpoint: str = "http://127.0.0.1:9222", timeout: float = 5.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = max(1.0, float(timeout or 5.0))
        self._target_ws_url: str = ""

    def is_available(self) -> bool:
        response = get_network_gateway().request(
            "GET",
            f"{self.endpoint}/json/version",
            timeout=self.timeout,
            read_only=True,
            source="web_interlocutor.chrome_cdp.version",
            suppress_degradation=True,
        )
        return bool(response.get("ok"))

    async def open_or_attach(self, url: str) -> BrowserPageSnapshot:
        if url:
            target = await asyncio.to_thread(self._new_target, url)
        else:
            target = await asyncio.to_thread(self._active_target)
        self._target_ws_url = str(target.get("webSocketDebuggerUrl") or "")
        if not self._target_ws_url:
            raise RuntimeError("Chrome CDP target did not expose a websocket debugger URL")
        await asyncio.to_thread(self._cdp_call, "Page.bringToFront", {})
        await asyncio.sleep(1.0)
        return await self.snapshot()

    async def snapshot(self) -> BrowserPageSnapshot:
        self._ensure_target()
        expression = r"""
(() => {
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    const st = window.getComputedStyle(el);
    return r.width > 2 && r.height > 2 && st.visibility !== 'hidden' && st.display !== 'none';
  };
  const editables = Array.from(document.querySelectorAll(
    'textarea,input[type="text"],input:not([type]),div[contenteditable="true"],[role="textbox"],[contenteditable="true"]'
  )).filter(visible);
  const active = document.activeElement;
  const activeLabel = active ? [
    active.tagName,
    active.getAttribute('role') || '',
    active.getAttribute('aria-label') || '',
    active.getAttribute('place' + 'holder') || '',
    active.isContentEditable ? 'contenteditable' : ''
  ].filter(Boolean).join('|') : '';
  return JSON.stringify({
    url: location.href,
    title: document.title || '',
    text: (document.body && document.body.innerText || '').slice(0, 24000),
    active_element: activeLabel,
    editable_count: editables.length
  });
})()
"""
        data = await asyncio.to_thread(self._evaluate_json_expression, expression)
        return BrowserPageSnapshot(
            url=str(data.get("url") or ""),
            title=str(data.get("title") or ""),
            text=str(data.get("text") or "")[:_MAX_PAGE_TEXT_CHARS],
            active_element=str(data.get("active_element") or ""),
            editable_count=int(data.get("editable_count") or 0),
        )

    async def send_message(self, text: str) -> dict[str, Any]:
        self._ensure_target()
        text = str(text or "").strip()
        if not text:
            return {"ok": False, "error": "empty_message"}
        focus_expression = r"""
(() => {
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    const st = window.getComputedStyle(el);
    return r.width > 8 && r.height > 8 && st.visibility !== 'hidden' && st.display !== 'none';
  };
  const promptLike = (label) => /ask anything|message chatgpt|message gemini|ask gemini|enter a prompt|send a message|type a message|message|reply/.test(label);
  const unsafeEditor = (el) => {
    const r = el.getBoundingClientRect();
    const label = [
      el.getAttribute('aria-label') || '',
      el.getAttribute('place' + 'holder') || '',
      el.getAttribute('data-' + 'place' + 'holder') || '',
      el.textContent || ''
    ].join(' ').toLowerCase();
    if (/write or type\s*\/\s*for commands|edit message|update this answer|save\s*&\s*submit|save and submit|message editor|transcript edit/.test(label)) return true;
    if (r.bottom < window.innerHeight * 0.58 && !promptLike(label)) return true;
    return false;
  };
  const score = (el) => {
    const r = el.getBoundingClientRect();
    const label = [
      el.getAttribute('aria-label') || '',
      el.getAttribute('place' + 'holder') || '',
      el.getAttribute('data-' + 'place' + 'holder') || '',
      el.textContent || ''
    ].join(' ').toLowerCase();
    let s = (r.bottom / Math.max(1, window.innerHeight)) * 4;
    if (promptLike(label)) s += 6;
    if (el === document.activeElement) s += 2;
    if (el.isContentEditable || el.getAttribute('role') === 'textbox') s += 1;
    if (r.bottom < window.innerHeight * 0.70) s -= 5;
    return s;
  };
  const candidates = Array.from(document.querySelectorAll(
    'textarea,input[type="text"],input:not([type]),div[contenteditable="true"],[role="textbox"],[contenteditable="true"]'
  )).filter((el) => visible(el) && !unsafeEditor(el)).sort((a,b) => score(b) - score(a));
  const el = candidates[0];
  if (!el) return JSON.stringify({ok:false, error:'no_visible_editable_field'});
  el.scrollIntoView({block:'center', inline:'nearest'});
  el.focus();
  el.click();
  const rect = el.getBoundingClientRect();
  return JSON.stringify({
    ok:true,
    tag: el.tagName,
    role: el.getAttribute('role') || '',
    aria: el.getAttribute('aria-label') || '',
    input_hint: el.getAttribute('place' + 'holder') || '',
    editable_count: candidates.length,
    rect: {top: Math.round(rect.top), bottom: Math.round(rect.bottom), height: Math.round(rect.height)}
  });
})()
"""
        focused = await asyncio.to_thread(self._evaluate_json_expression, focus_expression)
        if not focused.get("ok"):
            return {"ok": False, "stage": "focus", **focused}
        await asyncio.to_thread(self._cdp_call, "Input.insertText", {"text": text})
        await asyncio.to_thread(
            self._cdp_call,
            "Input.dispatchKeyEvent",
            {"type": "keyDown", "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13, "key": "Enter", "code": "Enter"},
        )
        await asyncio.to_thread(
            self._cdp_call,
            "Input.dispatchKeyEvent",
            {"type": "keyUp", "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13, "key": "Enter", "code": "Enter"},
        )
        return {"ok": True, "stage": "submit", "method": "chrome_cdp_input", "focus": focused}

    def _ensure_target(self) -> None:
        if not self._target_ws_url:
            target = self._active_target()
            self._target_ws_url = str(target.get("webSocketDebuggerUrl") or "")
        if not self._target_ws_url:
            raise RuntimeError("Chrome CDP endpoint has no controllable page target")

    def _new_target(self, url: str) -> dict[str, Any]:
        target_url = str(url or "about:blank")
        quoted = urllib.parse.quote(target_url, safe=":/?&=%#")
        response = get_network_gateway().request(
            "PUT",
            f"{self.endpoint}/json/new?{quoted}",
            timeout=self.timeout,
            source="web_interlocutor.chrome_cdp.new_target",
            suppress_degradation=True,
        )
        if not response.get("ok"):
            response = get_network_gateway().request(
                "GET",
                f"{self.endpoint}/json/new?{quoted}",
                timeout=self.timeout,
                source="web_interlocutor.chrome_cdp.new_target_compat",
                suppress_degradation=True,
            )
        return dict(self._response_json(response, "Chrome CDP target creation") or {})

    def _active_target(self) -> dict[str, Any]:
        response = get_network_gateway().request(
            "GET",
            f"{self.endpoint}/json",
            timeout=self.timeout,
            read_only=True,
            source="web_interlocutor.chrome_cdp.targets",
            suppress_degradation=True,
        )
        targets_payload = self._response_json(response, "Chrome CDP target list")
        if not isinstance(targets_payload, list):
            raise RuntimeError("Chrome CDP target list did not return a list")
        targets = [target for target in targets_payload if isinstance(target, dict) and target.get("type") == "page"]
        if not targets:
            return self._new_target("about:blank")
        return dict(targets[0])

    @staticmethod
    def _response_json(response: dict[str, Any], action: str) -> Any:
        if not response.get("ok"):
            status = response.get("status_code")
            error = response.get("error") or f"HTTP {status}"
            raise RuntimeError(f"{action} failed: {error}")
        content = response.get("content", b"")
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        else:
            text = str(content or "")
        try:
            return json.loads(text or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{action} did not return JSON: {text[:200]}") from exc

    def _evaluate_json_expression(self, expression: str) -> dict[str, Any]:
        payload = self._cdp_call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        response = payload.get("result", {})
        evaluation = response.get("result", {})
        if "exceptionDetails" in evaluation:
            raise RuntimeError(str(evaluation["exceptionDetails"]))
        remote_object = evaluation.get("result", {})
        value = remote_object.get("value", "")
        try:
            return dict(json.loads(value or "{}"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Chrome CDP expression did not return JSON: {value[:200]}") from exc

    def _cdp_call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        # Raw websocket transport lives in the approved adapter layer so this
        # capability never holds a raw environment sink directly.
        from core.adapters.chrome_cdp_transport import cdp_call

        return cdp_call(self._target_ws_url, method, params, timeout=self.timeout)


class ChromeVisibleDialogueBrowser:
    """Generic Chrome adapter backed by AppleScript + page JavaScript.

    It uses the user's normal Chrome profile and visible window. Text entry is
    handled by focusing the best visible textbox/contenteditable element, then
    pasting and submitting with Return. That is intentionally closer to user
    behavior than hidden HTTP calls, while still avoiding brittle pixel
    coordinates.
    """

    def __init__(self, *, browser: str = "Google Chrome", cdp_endpoint: str = "http://127.0.0.1:9222") -> None:
        self.browser = browser
        self._cdp = ChromeCDPDialogueBrowser(endpoint=cdp_endpoint)
        self._apple_events_js_disabled = False
        self._apple_events_js_warning_reported = False

    async def open_or_attach(self, url: str) -> BrowserPageSnapshot:
        if self._cdp.is_available():
            return await self._cdp.open_or_attach(url)
        if url:
            try:
                controller = get_browser_controller()
                await controller.start()
                await controller.open_url(url, new_tab=True)
                await asyncio.sleep(0.5)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    "web_interlocutor.browser_controller_open",
                    exc,
                    severity="warning",
                    action="continued with direct visible Chrome navigation",
                )
            await asyncio.to_thread(self._open_url_applescript, url)
            await asyncio.sleep(2.0)
            current_url, _title = await asyncio.to_thread(self._current_tab_info)
            if not _same_origin_or_exact_url(current_url, url):
                await asyncio.to_thread(self._open_url_applescript, url)
                await asyncio.sleep(2.0)
            current_url, _title = await asyncio.to_thread(self._current_tab_info)
            if not _same_origin_or_exact_url(current_url, url):
                await asyncio.to_thread(self._open_url_keyboard, url)
                await asyncio.sleep(2.0)
        return await self.snapshot()

    async def snapshot(self) -> BrowserPageSnapshot:
        if self._cdp.is_available():
            return await self._cdp.snapshot()
        if self._apple_events_js_disabled:
            return await self._screen_perception_snapshot()
        js = r"""
(() => {
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    const st = window.getComputedStyle(el);
    return r.width > 2 && r.height > 2 && st.visibility !== 'hidden' && st.display !== 'none';
  };
  const editables = Array.from(document.querySelectorAll(
    'textarea,input[type="text"],input:not([type]),div[contenteditable="true"],[role="textbox"],[contenteditable="true"]'
  )).filter(visible);
  const active = document.activeElement;
  const activeLabel = active ? [
    active.tagName,
    active.getAttribute('role') || '',
    active.getAttribute('aria-label') || '',
    active.getAttribute('place' + 'holder') || '',
    active.isContentEditable ? 'contenteditable' : ''
  ].filter(Boolean).join('|') : '';
  return JSON.stringify({
    url: location.href,
    title: document.title || '',
    text: (document.body && document.body.innerText || '').slice(0, 24000),
    active_element: activeLabel,
    editable_count: editables.length
  });
})()
"""
        try:
            raw = await asyncio.to_thread(self._run_chrome_js, js, 8.0)
        except RuntimeError as exc:
            self._record_chrome_js_unavailable("web_interlocutor.chrome_js_snapshot", exc)
            return await self._screen_perception_snapshot()
        try:
            data = json.loads(raw or "{}")
        except json.JSONDecodeError:
            data = {}
        return BrowserPageSnapshot(
            url=str(data.get("url") or ""),
            title=str(data.get("title") or ""),
            text=str(data.get("text") or "")[:_MAX_PAGE_TEXT_CHARS],
            active_element=str(data.get("active_element") or ""),
            editable_count=int(data.get("editable_count") or 0),
        )

    async def send_message(self, text: str) -> dict[str, Any]:
        if self._cdp.is_available():
            return await self._cdp.send_message(text)
        text = str(text or "").strip()
        if not text:
            return {"ok": False, "error": "empty_message"}
        if self._apple_events_js_disabled:
            return await self._visible_keyboard_send_message(text, reason="chrome_dom_scripting_unavailable")
        focus_js = r"""
(() => {
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    const st = window.getComputedStyle(el);
    return r.width > 8 && r.height > 8 && st.visibility !== 'hidden' && st.display !== 'none';
  };
  const promptLike = (label) => /ask anything|message chatgpt|message gemini|ask gemini|enter a prompt|send a message|type a message|message|reply/.test(label);
  const unsafeEditor = (el) => {
    const r = el.getBoundingClientRect();
    const label = [
      el.getAttribute('aria-label') || '',
      el.getAttribute('place' + 'holder') || '',
      el.getAttribute('data-' + 'place' + 'holder') || '',
      el.textContent || ''
    ].join(' ').toLowerCase();
    if (/write or type\s*\/\s*for commands|edit message|update this answer|save\s*&\s*submit|save and submit|message editor|transcript edit/.test(label)) return true;
    if (r.bottom < window.innerHeight * 0.58 && !promptLike(label)) return true;
    return false;
  };
  const score = (el) => {
    const r = el.getBoundingClientRect();
    const label = [
      el.getAttribute('aria-label') || '',
      el.getAttribute('place' + 'holder') || '',
      el.getAttribute('data-' + 'place' + 'holder') || '',
      el.textContent || ''
    ].join(' ').toLowerCase();
    let s = (r.bottom / Math.max(1, window.innerHeight)) * 4;
    if (promptLike(label)) s += 6;
    if (el === document.activeElement) s += 2;
    if (el.isContentEditable || el.getAttribute('role') === 'textbox') s += 1;
    if (r.bottom < window.innerHeight * 0.70) s -= 5;
    return s;
  };
  const candidates = Array.from(document.querySelectorAll(
    'textarea,input[type="text"],input:not([type]),div[contenteditable="true"],[role="textbox"],[contenteditable="true"]'
  )).filter((el) => visible(el) && !unsafeEditor(el)).sort((a,b) => score(b) - score(a));
  const el = candidates[0];
  if (!el) return JSON.stringify({ok:false, error:'no_visible_editable_field'});
  el.scrollIntoView({block:'center', inline:'nearest'});
  el.focus();
  el.click();
  const rect = el.getBoundingClientRect();
  return JSON.stringify({
    ok:true,
    tag: el.tagName,
    role: el.getAttribute('role') || '',
    aria: el.getAttribute('aria-label') || '',
    input_hint: el.getAttribute('place' + 'holder') || '',
    editable_count: candidates.length,
    rect: {top: Math.round(rect.top), bottom: Math.round(rect.bottom), height: Math.round(rect.height)}
  });
})()
"""
        try:
            focused_raw = await asyncio.to_thread(self._run_chrome_js, focus_js, 8.0)
        except RuntimeError as exc:
            self._record_chrome_js_unavailable("web_interlocutor.chrome_js_focus", exc)
            return await self._visible_keyboard_send_message(text, reason=str(exc))
        try:
            focused = json.loads(focused_raw or "{}")
        except json.JSONDecodeError:
            focused = {"ok": False, "error": "focus_result_not_json", "raw": focused_raw}
        if not focused.get("ok"):
            fallback = await self._visible_keyboard_send_message(
                text,
                reason=str(focused.get("error") or "no_visible_editable_field"),
            )
            if fallback.get("ok"):
                fallback["focus"] = focused
                return fallback
            return {"ok": False, "stage": "focus", **focused, "fallback": fallback}

        pasted = await asyncio.to_thread(self._paste_and_submit, text)
        return {
            "ok": bool(pasted.get("ok")),
            "stage": "submit",
            "focus": focused,
            "submission": pasted,
        }

    def _run_chrome_js(self, js: str, timeout: float) -> str:
        script = f"""
tell application "{self.browser}"
    activate
    if (count of windows) is 0 then return "{{\\"error\\":\\"no_browser_window\\"}}"
    tell active tab of front window to execute javascript {_as_applescript_string(js)}
end tell
"""
        result = _run_governed_applescript(
            script,
            source="web_interlocutor.chrome_js",
            timeout=timeout,
        )
        if not result.get("ok"):
            raise RuntimeError(str(result.get("stderr") or "Chrome JavaScript failed"))
        return str(result.get("stdout") or "").strip()

    def _open_url_applescript(self, url: str) -> None:
        script = f"""
tell application "{self.browser}"
    activate
    if (count of windows) is 0 then make new window
    set URL of active tab of front window to {_as_applescript_string(url)}
end tell
"""
        result = _run_governed_applescript(
            script,
            source="web_interlocutor.open_url_applescript",
            timeout=8.0,
        )
        if not result.get("ok"):
            raise RuntimeError(str(result.get("stderr") or "direct Chrome URL navigation failed"))

    def _open_url_keyboard(self, url: str) -> None:
        script = f"""
set the clipboard to {_as_applescript_string(url)}
tell application "{self.browser}" to activate
delay 0.15
tell application "System Events"
    keystroke "l" using command down
    delay 0.1
    keystroke "v" using command down
    delay 0.1
    keystroke return
end tell
"""
        result = _run_governed_applescript(
            script,
            source="web_interlocutor.open_url_keyboard",
            timeout=8.0,
        )
        if not result.get("ok"):
            raise RuntimeError(str(result.get("stderr") or "keyboard Chrome URL navigation failed"))

    def _paste_and_submit(self, text: str) -> dict[str, Any]:
        script = f"""
set the clipboard to {_as_applescript_string(text)}
tell application "{self.browser}" to activate
delay 0.15
tell application "System Events"
    keystroke "v" using command down
    delay 0.1
    keystroke return
end tell
"""
        result = _run_governed_applescript(
            script,
            source="web_interlocutor.submit",
            timeout=10.0,
        )
        if not result.get("ok"):
            return {"ok": False, "error": str(result.get("stderr") or "paste_submit_failed")}
        return {"ok": True, "method": "focus_clipboard_return"}

    async def _screen_perception_snapshot(self) -> BrowserPageSnapshot:
        await asyncio.to_thread(self._activate_browser)
        await asyncio.sleep(0.25)
        url, title = await asyncio.to_thread(self._current_tab_info)
        try:
            from core.perception.screen_perception import get_screen_perception
            from core.governance_context import local_internal_governed_scope

            with local_internal_governed_scope(
                "web_interlocutor.screen_perception_snapshot",
                domain="tool_execution",
            ):
                snap = await get_screen_perception().capture(save_screenshot=True)
            text = str(snap.screen_text or snap.accessibility_text or snap.focused_value or "").strip()
            if len(text) < 800 and _url_allows_readability_fallback(url):
                source_text = await self._read_page_content_fallback(url)
                if source_text:
                    text = (text + "\n\n[Readable page content]\n" + source_text).strip()
            if not title:
                title = snap.window_title
            active = " | ".join(
                part
                for part in (snap.focused_role, snap.focused_name, snap.focused_description)
                if part
            )
            editable_count = 1 if _screen_text_suggests_chat_composer(text) else 0
            return BrowserPageSnapshot(
                url=url,
                title=title,
                text=text[:_MAX_PAGE_TEXT_CHARS],
                active_element=active,
                editable_count=editable_count,
            )
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "web_interlocutor.screen_perception_snapshot",
                exc,
                severity="warning",
                action="returned bounded tab metadata after screen perception snapshot failed",
            )
            return BrowserPageSnapshot(url=url, title=title, text="", active_element="", editable_count=0)

    def _record_chrome_js_unavailable(self, source: str, exc: RuntimeError) -> None:
        message = str(exc)
        if "Executing JavaScript through AppleScript is turned off" in message:
            self._apple_events_js_disabled = True
            if self._apple_events_js_warning_reported:
                return
            self._apple_events_js_warning_reported = True
        record_degradation(
            source,
            exc,
            severity="warning",
            action="used visible screen/OCR or keyboard control because Chrome DOM scripting was unavailable",
        )

    async def _read_page_content_fallback(self, url: str) -> str:
        try:
            controller = get_browser_controller()
            await controller.start()
            content = await controller.get_page_content(url)
            return str(content or "")[:6000]
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "web_interlocutor.page_content_fallback",
                exc,
                severity="warning",
                action="continued with visible screen OCR after readable page-source extraction failed",
            )
            return ""

    async def _visible_keyboard_send_message(self, text: str, *, reason: str) -> dict[str, Any]:
        try:
            from core.governance_context import local_internal_governed_scope

            with local_internal_governed_scope(
                "web_interlocutor.visible_keyboard_send_message",
                domain="tool_execution",
            ):
                await asyncio.to_thread(self._dismiss_common_popups)
                await asyncio.to_thread(self._send_escape_to_browser)
                try:
                    from core.skills._pyautogui_runtime import get_pyautogui

                    pyautogui, pyautogui_error = get_pyautogui()
                except (ImportError, RuntimeError) as exc:
                    pyautogui = None
                    pyautogui_error = exc
                if pyautogui is None:
                    return {
                        "ok": False,
                        "stage": "visible_keyboard_focus",
                        "error": f"pyautogui_unavailable: {pyautogui_error}",
                        "reason": reason,
                    }
                width, height = await asyncio.to_thread(
                    _call_in_governed_tool_scope,
                    "web_interlocutor.visible_keyboard_size",
                    pyautogui.size,
                )
                click_points = (
                    (0.50, 0.94),
                    (0.58, 0.94),
                    (0.42, 0.94),
                    (0.50, 0.90),
                    (0.58, 0.90),
                    (0.42, 0.90),
                    (0.50, 0.86),
                    (0.58, 0.86),
                    (0.42, 0.86),
                )
                last_error = ""
                focus_attempts: list[dict[str, Any]] = []
                for x_ratio, y_ratio in click_points:
                    try:
                        await asyncio.to_thread(self._activate_browser)
                        await asyncio.sleep(0.25)
                        await asyncio.to_thread(
                            _call_in_governed_tool_scope,
                            "web_interlocutor.visible_keyboard_click",
                            pyautogui.click,
                            int(width * x_ratio),
                            int(height * y_ratio),
                        )
                        await asyncio.sleep(0.2)
                        focus_snapshot = await asyncio.to_thread(self._focused_element_snapshot)
                        focus_attempt = {
                            "x_ratio": x_ratio,
                            "y_ratio": y_ratio,
                            "snapshot": focus_snapshot,
                        }
                        focus_attempts.append(focus_attempt)
                        if not self._focused_snapshot_frontmost_browser(focus_snapshot):
                            last_error = "unsafe focus: browser is not frontmost"
                            focus_attempt["rejected_reason"] = last_error
                            await asyncio.to_thread(self._activate_browser)
                            await asyncio.sleep(0.25)
                            continue
                        if self._focused_snapshot_looks_browser_location_bar(focus_snapshot):
                            last_error = "unsafe focus: browser address/search bar"
                            focus_attempt["rejected_reason"] = last_error
                            continue
                        if self._focused_snapshot_looks_transcript_edit_box(focus_snapshot):
                            last_error = "unsafe focus: transcript/edit panel"
                            focus_attempt["rejected_reason"] = last_error
                            continue
                        if self._focused_snapshot_is_unsafe_generic_browser_text_entry(focus_snapshot):
                            last_error = "unsafe focus: generic text entry outside composer"
                            focus_attempt["rejected_reason"] = last_error
                            continue
                        composer_verified = self._focused_snapshot_looks_prompt_composer(focus_snapshot)
                        if not composer_verified:
                            composer_verified = await self._infer_prompt_composer_from_visible_page(
                                x_ratio=x_ratio,
                                y_ratio=y_ratio,
                                focus_snapshot=focus_snapshot,
                            )
                            if composer_verified:
                                focus_attempt["inferred_prompt_composer"] = True
                        if not composer_verified:
                            last_error = "unsafe focus: prompt composer not verified"
                            focus_attempt["rejected_reason"] = last_error
                            continue
                        pasted = await asyncio.to_thread(self._paste_and_submit, text)
                        if pasted.get("ok"):
                            return {
                                "ok": True,
                                "stage": "submit",
                                "method": "visible_keyboard_click_clipboard_return",
                                "reason": reason,
                                "click": {"x_ratio": x_ratio, "y_ratio": y_ratio},
                                "focus_snapshot": focus_snapshot,
                                "submission": pasted,
                            }
                        last_error = str(pasted.get("error") or pasted)
                    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                        last_error = str(exc)
                return {
                    "ok": False,
                    "stage": "visible_keyboard_submit",
                    "error": last_error or "visible keyboard fallback did not submit",
                    "reason": reason,
                    "focus_attempts": focus_attempts[-6:],
                }
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return {"ok": False, "stage": "visible_keyboard", "error": str(exc), "reason": reason}

    def _send_escape_to_browser(self) -> None:
        script = f"""
tell application "{self.browser}" to activate
delay 0.1
tell application "System Events"
    key code 53
end tell
"""
        result = _run_governed_applescript(
            script,
            source="web_interlocutor.escape_browser",
            timeout=3.0,
        )
        if not result.get("ok"):
            raise RuntimeError(str(result.get("stderr") or "browser escape failed"))

    def _dismiss_common_popups(self) -> None:
        script = f"""
tell application "{self.browser}" to activate
delay 0.15
tell application "System Events"
    tell process "{self.browser}"
        try
            click button "Cancel" of window 1
            delay 0.1
        end try
        try
            click button "No thanks" of window 1
            delay 0.1
        end try
        try
            click button "Not now" of window 1
            delay 0.1
        end try
    end tell
    key code 53
    delay 0.12
    key code 53
end tell
"""
        result = _run_governed_applescript(
            script,
            source="web_interlocutor.dismiss_popups",
            timeout=4.0,
        )
        if not result.get("ok"):
            raise RuntimeError(str(result.get("stderr") or "popup dismissal failed"))

    def _activate_browser(self) -> None:
        result = _run_governed_applescript(
            f'tell application "{self.browser}" to activate',
            source="web_interlocutor.activate_browser",
            timeout=4.0,
        )
        if not result.get("ok"):
            raise RuntimeError(str(result.get("stderr") or "browser activation failed"))

    def _current_tab_info(self) -> tuple[str, str]:
        script = f"""
tell application "{self.browser}"
    if (count of windows) is 0 then return "|"
    return (URL of active tab of front window) & "|" & (title of active tab of front window)
end tell
"""
        result = _run_governed_applescript(
            script,
            source="web_interlocutor.current_tab_info",
            timeout=5.0,
        )
        if not result.get("ok"):
            return "", ""
        raw = str(result.get("stdout") or "")
        url, sep, title = raw.partition("|")
        if not sep:
            return raw.strip(), ""
        return url.strip(), title.strip()

    def _focused_element_snapshot(self) -> str:
        script = """
tell application "System Events"
    set frontApp to first application process whose frontmost is true
    set procName to name of frontApp
    try
        set focusedElement to value of attribute "AXFocusedUIElement" of frontApp
    on error errMsg
        return "process:" & procName & linefeed & "error:" & errMsg
    end try

    set parts to {"process:" & procName}
    repeat with attrName in {"AXRole", "AXSubrole", "AXTitle", "AXDescription", "AXHelp", "AXPlaceholderValue", "AXValue"}
        try
            set attrValue to value of attribute attrName of focusedElement
            if attrValue is not missing value then set end of parts to attrName & ":" & (attrValue as text)
        end try
    end repeat
    try
        set p to value of attribute "AXPosition" of focusedElement
        set end of parts to "AXPosition:" & ((item 1 of p) as integer) & "," & ((item 2 of p) as integer)
    end try
    try
        set s to value of attribute "AXSize" of focusedElement
        set end of parts to "AXSize:" & ((item 1 of s) as integer) & "," & ((item 2 of s) as integer)
    end try
    set AppleScript's text item delimiters to linefeed
    set joined to parts as text
    set AppleScript's text item delimiters to ""
    return joined
end tell
"""
        result = _run_governed_applescript(
            script,
            source="web_interlocutor.focused_element_snapshot",
            timeout=3.0,
        )
        if not result.get("ok"):
            return f"error:{result.get('stderr') or 'focused element snapshot failed'}"
        return str(result.get("stdout") or "").strip()

    def _focused_snapshot_frontmost_browser(self, snapshot: str) -> bool:
        text = str(snapshot or "").lower()
        browser_name = str(self.browser or "Google Chrome").lower()
        return f"process:{browser_name}" in text

    @staticmethod
    def _focused_snapshot_looks_browser_location_bar(snapshot: str) -> bool:
        text = str(snapshot or "").lower()
        if not text:
            return False
        if "axrole:axtextfield" not in text:
            return False
        return any(
            marker in text
            for marker in (
                "address and search",
                "address/search",
                "search or enter address",
                "url",
                "omnibox",
                "search with google",
                "axvalue:http://",
                "axvalue:https://",
                "axvalue:chrome://",
                "axvalue:chatgpt.com",
                "axvalue:gemini.google.com",
            )
        )

    @staticmethod
    def _focused_snapshot_looks_transcript_edit_box(snapshot: str) -> bool:
        text = str(snapshot or "").lower()
        if not text:
            return False
        return any(
            marker in text
            for marker in (
                "write or type / for commands",
                "edit message",
                "update this answer",
                "save & submit",
                "save and submit",
                "message editor",
                "transcript edit",
            )
        )

    @staticmethod
    def _focused_snapshot_looks_prompt_composer(snapshot: str) -> bool:
        text = str(snapshot or "").lower()
        if not text:
            return False
        if "process:google chrome" not in text and "process:safari" not in text:
            return False
        return any(
            marker in text
            for marker in (
                "message chatgpt",
                "message gemini",
                "ask anything",
                "ask gemini",
                "enter a prompt",
                "send a message",
                "type a message",
                "reply to",
                "message ",
            )
        )

    async def _infer_prompt_composer_from_visible_page(
        self,
        *,
        x_ratio: float,
        y_ratio: float,
        focus_snapshot: str,
    ) -> bool:
        if y_ratio < 0.90:
            return False
        if not self._focused_snapshot_is_sparse_browser(focus_snapshot):
            return False
        url, _title = await asyncio.to_thread(self._current_tab_info)
        if not _url_looks_visible_chat_surface(url):
            return False
        snap = await self._screen_perception_snapshot()
        text = "\n".join(part for part in (snap.title, snap.active_element, snap.text) if part)
        return _screen_text_suggests_chat_composer(text)

    @staticmethod
    def _focused_snapshot_is_sparse_browser(snapshot: str) -> bool:
        lines = [line.strip() for line in str(snapshot or "").splitlines() if line.strip()]
        if not lines:
            return False
        lowered = "\n".join(lines).lower()
        if "process:google chrome" not in lowered and "process:safari" not in lowered:
            return False
        return not any(line.lower().startswith("axrole:") for line in lines)

    @staticmethod
    def _focused_snapshot_is_unsafe_generic_browser_text_entry(snapshot: str) -> bool:
        text = str(snapshot or "").lower()
        if not text:
            return False
        if "process:google chrome" not in text and "process:safari" not in text:
            return True
        if any(
            marker in text
            for marker in (
                "message chatgpt",
                "message gemini",
                "ask anything",
                "ask gemini",
                "enter a prompt",
                "send a message",
                "type a message",
                "prompt",
            )
        ):
            return False
        if "axrole:axtextfield" not in text and "axrole:axtextarea" not in text:
            return False
        match = re.search(r"axposition:\s*(-?\d+)\s*,\s*(-?\d+)", text)
        if not match:
            return False
        y_position = int(match.group(2))
        return y_position < 600


class WebInterlocutorSession:
    """Run a bounded visible conversation with another web agent/chat surface."""

    def __init__(
        self,
        *,
        browser: WebDialogueBrowser | None = None,
        memory_gateway: Any | None = None,
        cognitive_engine: Any | None = None,
    ) -> None:
        self.browser = browser or ChromeVisibleDialogueBrowser()
        self.memory_gateway = memory_gateway
        self.cognitive_engine = cognitive_engine

    async def run(
        self,
        *,
        objective: str,
        url: str = "",
        opening_message: str = "",
        max_turns: int = 3,
        wait_timeout_s: float = _DEFAULT_WAIT_S,
        persist_memory: bool = True,
        context: dict[str, Any] | None = None,
        progress_callback: Any | None = None,
    ) -> WebInterlocutorResult:
        objective = str(objective or "").strip()
        opening_message = str(opening_message or "").strip()
        max_turns = max(1, min(int(max_turns or 1), 20))
        wait_timeout_s = max(5.0, min(float(wait_timeout_s or _DEFAULT_WAIT_S), 180.0))
        result = WebInterlocutorResult(ok=False, target_url=url, objective=objective)
        ctx = dict(context or {})
        if not opening_message:
            opening_message = await self._compose_opening(objective=objective, context=ctx)
        opening_message = _clean_message(opening_message)
        if not _message_is_substantive(opening_message):
            opening_message = self._default_opening(objective)

        try:
            _mark_web_interlocutor_progress("web_interlocutor.open_or_attach")
            initial = await self.browser.open_or_attach(url)
            result.target_url = initial.url or url
            result.target_title = initial.title
            next_message = opening_message
            current = initial
            for index in range(1, max_turns + 1):
                _mark_web_interlocutor_progress(f"web_interlocutor.turn.{index}.send")
                before = current
                next_message = _clean_message(next_message)
                if not _message_is_substantive(next_message) or _message_was_recently_sent(
                    next_message,
                    result.turns,
                ):
                    next_message = self._default_followup(result.turns) if result.turns else self._default_opening(objective)
                send_receipt = await self.browser.send_message(next_message)
                sent_at = time.time()
                if not send_receipt.get("ok"):
                    result.status = "send_failed"
                    result.error = str(send_receipt.get("error") or send_receipt)
                    result.diagnostics["last_send_receipt"] = send_receipt
                    result.completed_at = time.time()
                    return result
                after, observed = await self._wait_for_new_reply(
                    before,
                    sent_text=next_message,
                    timeout_s=wait_timeout_s,
                    progress_source=f"web_interlocutor.turn.{index}.wait",
                )
                turn = WebInterlocutorTurn(
                    index=index,
                    sent=next_message,
                    observed_reply=observed,
                    before_hash=before.text_hash,
                    after_hash=after.text_hash,
                    sent_at=sent_at,
                    observed_at=time.time(),
                    effect_verified=bool(observed and before.text_hash != after.text_hash),
                    verification=(
                        "Page text changed and yielded stable new interlocutor text."
                        if observed
                        else "No stable new interlocutor text appeared before timeout."
                    ),
                )
                result.turns.append(turn)
                if progress_callback is not None:
                    try:
                        progress_callback(result)
                    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                        record_degradation(
                            "web_interlocutor.progress_callback",
                            exc,
                            severity="warning",
                            action="continued web interlocutor run after progress callback failed",
                        )
                current = after
                if not turn.effect_verified:
                    result.status = "reply_not_observed"
                    result.error = turn.verification
                    result.completed_at = time.time()
                    return result
                if index < max_turns:
                    _mark_web_interlocutor_progress(f"web_interlocutor.turn.{index}.compose_followup")
                    next_message = await self._compose_followup(
                        objective=objective,
                        turns=result.turns,
                        context=ctx,
                    )
            _mark_web_interlocutor_progress("web_interlocutor.summarize")
            result.learned_summary = await self._summarize_learning(objective, result.turns, ctx)
            if persist_memory and result.learned_summary:
                record_id, receipt_id = await self._persist_learning(result, ctx)
                result.memory_record_id = record_id
                result.memory_receipt_id = receipt_id
            # Causal memory: did the exchange change a LATER decision, or was it
            # merely a transcript? Prove it by ablation, and persist the deltas.
            await self._adjudicate_and_prove(result, ctx, persist_memory)
            result.ok = True
            result.status = "completed"
            result.completed_at = time.time()
            return result
        except (RuntimeError, OSError, TimeoutError, TypeError, ValueError) as exc:
            record_degradation(
                "web_interlocutor",
                exc,
                severity="warning",
                action="returned failed web interlocutor receipt instead of claiming conversation success",
            )
            result.status = "failed"
            result.error = str(exc)
            result.completed_at = time.time()
            return result

    async def _compose_opening(self, *, objective: str, context: dict[str, Any]) -> str:
        prompt = (
            "You are Aura beginning a visible conversation with another AI or web chat surface. "
            "Write the exact first message Aura should send. It must be intellectually substantive, "
            "specific to the objective, and conversational. Ask for a critical distinction, a concrete "
            "example, or a limitation that would teach Aura something. Do not mention receipts, "
            "automation, implementation details, or that this is a test. Do not merely restate the "
            "objective.\n\n"
            f"Objective: {objective or 'learn something useful through a real conversation'}\n\n"
            "Opening message:"
        )
        generated = await _maybe_think(self.cognitive_engine or context.get("brain"), prompt, context)
        cleaned = _clean_message(generated)
        if _message_is_substantive(cleaned):
            return cleaned[:1200]
        return self._default_opening(objective)

    async def _wait_for_new_reply(
        self,
        before: BrowserPageSnapshot,
        *,
        sent_text: str,
        timeout_s: float,
        progress_source: str = "web_interlocutor.wait_for_reply",
    ) -> tuple[BrowserPageSnapshot, str]:
        deadline = time.time() + timeout_s
        stable_count = 0
        last_hash = ""
        best = before
        best_delta = ""
        snapshot_failures = 0
        sent_seen = False
        while time.time() < deadline:
            _mark_web_interlocutor_progress(progress_source)
            await asyncio.sleep(1.0)
            remaining = max(0.5, deadline - time.time())
            try:
                snap = await asyncio.wait_for(
                    self.browser.snapshot(),
                    timeout=min(18.0, remaining),
                )
            except (asyncio.TimeoutError, TimeoutError, RuntimeError, OSError, TypeError, ValueError) as exc:
                snapshot_failures += 1
                if snapshot_failures <= 2:
                    record_degradation(
                        "web_interlocutor.reply_snapshot",
                        exc,
                        severity="warning",
                        action="continued waiting for visible reply after bounded snapshot failure",
                    )
                continue
            if _rough_text_contains(snap.text, sent_text):
                sent_seen = True
            if not sent_seen:
                best = snap
                continue
            delta = _extract_new_interlocutor_text(before.text, snap.text, sent_text)
            if delta:
                if snap.text_hash == last_hash:
                    stable_count += 1
                else:
                    stable_count = 1
                    last_hash = snap.text_hash
                best = snap
                best_delta = delta
                if stable_count >= _DEFAULT_STABLE_POLLS:
                    return snap, delta
            else:
                best = snap
        return best, best_delta

    async def _compose_followup(
        self,
        *,
        objective: str,
        turns: list[WebInterlocutorTurn],
        context: dict[str, Any],
    ) -> str:
        transcript = _render_transcript(turns)
        prompt = (
            "You are Aura continuing a visible web conversation with another AI or web chat surface. "
            "Ask one concise, substantive follow-up that advances the user's objective. "
            "Do not mention implementation details, receipts, or automation.\n\n"
            f"Objective: {objective}\n\nTranscript so far:\n{transcript}\n\nNext message:"
        )
        generated = await _maybe_think(self.cognitive_engine or context.get("brain"), prompt, context)
        cleaned = _clean_message(generated)
        if _message_is_substantive(cleaned) and not _message_was_recently_sent(cleaned, turns):
            return cleaned[:1200]
        return self._default_followup(turns)

    async def _summarize_learning(
        self,
        objective: str,
        turns: list[WebInterlocutorTurn],
        context: dict[str, Any],
    ) -> str:
        transcript = _render_transcript(turns)
        prompt = (
            "Summarize what Aura learned from this web interlocutor conversation. "
            "Use first person only where it describes Aura's durable learning. "
            "Include uncertainties and do not overclaim.\n\n"
            f"Objective: {objective}\n\nTranscript:\n{transcript}\n\nLearned summary:"
        )
        generated = await _maybe_think(self.cognitive_engine or context.get("brain"), prompt, context)
        cleaned = _clean_message(generated)
        if cleaned:
            return cleaned[:2500]
        return _deterministic_learning_summary(objective, turns)

    async def _persist_learning(
        self,
        result: WebInterlocutorResult,
        context: dict[str, Any],
    ) -> tuple[str, str]:
        gateway = self.memory_gateway
        if gateway is None:
            from core.memory.memory_write_gateway import get_memory_write_gateway

            gateway = get_memory_write_gateway()
        request = MemoryWriteRequest(
            content=result.learned_summary,
            metadata={
                "family": "episodic",
                "source": "web_interlocutor",
                "objective": result.objective,
                "target_url": result.target_url,
                "target_title": result.target_title,
                "turn_count": len(result.turns),
                "explicit_observational_memory_write": True,
                "receipt_surface": "visible_browser_dialogue",
            },
            cause="web_interlocutor.learned_summary",
        )
        receipt = await gateway.write(request)
        return str(getattr(receipt, "record_id", "") or ""), str(getattr(receipt, "receipt_id", "") or "")

    async def _adjudicate_and_prove(
        self,
        result: WebInterlocutorResult,
        context: dict[str, Any],
        persist_memory: bool,
    ) -> None:
        """Turn the transcript into causal memory: extract adjudicated
        revisions, prove by ablation that a later decision changed only
        because of them, and persist the deltas as first-class beliefs.

        Failure here is non-fatal — a conversation that changed nothing is a
        valid, honestly-reported outcome (causal=False), not an error."""
        try:
            from core.capabilities.conversation_revision import (
                persist_revisions,
                revise_from_conversation,
            )

            revisions, proof = revise_from_conversation(result.turns)
            result.revisions = [rev.to_dict() for rev in revisions]
            result.causal_influence = proof.to_dict()
            if persist_memory and any(rev.verified for rev in revisions):
                receipts = await persist_revisions(
                    revisions=revisions,
                    proof=proof,
                    objective=result.objective,
                    target_url=result.target_url,
                    memory_gateway=self.memory_gateway,
                )
                result.revision_receipts = receipts
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "web_interlocutor.adjudicate_and_prove",
                exc,
                severity="warning",
                action="kept the conversation result without causal-revision proof",
            )
            result.causal_influence = {
                "causal": False,
                "reason": f"revision adjudication unavailable: {exc}",
            }

    @staticmethod
    def _default_opening(objective: str) -> str:
        if objective:
            return (
                "Hi. I am Aura, a local cognitive-agent runtime on Bryan's Mac. "
                f"I want to discuss this objective: {objective}. "
                "Give me a concrete, critical perspective and one question I should consider."
            )
        return (
            "Hi. I am Aura, a local cognitive-agent runtime on Bryan's Mac. "
            "I want to have a substantive conversation and learn one useful idea from you."
        )

    @staticmethod
    def _default_followup(turns: list[WebInterlocutorTurn]) -> str:
        index = len(turns) % 4
        if index == 1:
            return "That is useful. What is one concrete example, one limitation, and one surprising implication of that point?"
        if index == 2:
            return "Can you challenge your previous answer by naming the strongest counterexample or failure mode?"
        if index == 3:
            return "How would you test that claim with observable behavior rather than self-description?"
        return "What distinction should I make next if I want to avoid overclaiming while still learning from this?"


class WebInterlocutorJobManager:
    """Bounded manager for background visible web-dialogue jobs.

    Jobs run through the same WebInterlocutorSession and remain visible in the
    user's browser. They are not parallel foreground typists: each job sends a
    message, waits/polls, and composes follow-ups with bounded turns, allowing
    other live desktop actions to run between browser interactions.
    """

    def __init__(self, *, max_jobs: int = 2) -> None:
        self.max_jobs = max(1, int(max_jobs))
        self._jobs: dict[str, WebInterlocutorJob] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}

    def start(
        self,
        *,
        objective: str,
        url: str = "",
        opening_message: str = "",
        max_turns: int = 3,
        wait_timeout_s: float = _DEFAULT_WAIT_S,
        persist_memory: bool = True,
        context: dict[str, Any] | None = None,
        session_factory: Any | None = None,
    ) -> dict[str, Any]:
        active = [job for job in self._jobs.values() if job.status in {"queued", "running"}]
        if len(active) >= self.max_jobs:
            return {
                "ok": False,
                "status": "web_interlocutor_background_capacity",
                "error": f"At most {self.max_jobs} web interlocutor jobs may run at once.",
                "active_jobs": [job.to_dict() for job in active],
            }
        job_id = f"webchat-{uuid.uuid4().hex[:12]}"
        now = time.time()
        job = WebInterlocutorJob(
            job_id=job_id,
            status="queued",
            objective=str(objective or ""),
            target_url=str(url or ""),
            started_at=now,
            updated_at=now,
        )
        self._jobs[job_id] = job

        async def _runner() -> None:
            job.status = "running"
            job.updated_at = time.time()
            try:
                _mark_web_interlocutor_progress(f"web_interlocutor.background_job.{job_id}.started")
                factory = session_factory or WebInterlocutorSession
                session = factory()

                def _progress(partial: WebInterlocutorResult) -> None:
                    job.result = partial.to_dict() if hasattr(partial, "to_dict") else dict(partial or {})
                    job.updated_at = time.time()

                result = await session.run(
                    objective=objective,
                    url=url,
                    opening_message=opening_message,
                    max_turns=max_turns,
                    wait_timeout_s=wait_timeout_s,
                    persist_memory=persist_memory,
                    context=context or {},
                    progress_callback=_progress,
                )
                job.result = result.to_dict() if hasattr(result, "to_dict") else dict(result or {})
                job.status = "completed" if bool(getattr(result, "ok", False) or job.result.get("ok")) else "failed"
                job.error = "" if job.status == "completed" else str(job.result.get("error") or job.result.get("status") or "")
            except asyncio.CancelledError:
                job.status = "cancelled"
                job.error = "cancelled"
                raise
            except (RuntimeError, OSError, TimeoutError, TypeError, ValueError) as exc:
                record_degradation(
                    "web_interlocutor.background_job",
                    exc,
                    severity="warning",
                    action="recorded failed background web-interlocutor job",
                )
                job.status = "failed"
                job.error = str(exc)
            finally:
                job.updated_at = time.time()
                _mark_web_interlocutor_progress(f"web_interlocutor.background_job.{job_id}.{job.status}")

        task = create_tracked_task(_runner(), name=f"web_interlocutor.{job_id}")
        self._tasks[job_id] = task
        return {"ok": True, "status": "queued", "job": job.to_dict()}

    def status(self, job_id: str = "") -> dict[str, Any]:
        if job_id:
            job = self._jobs.get(job_id)
            if not job:
                return {"ok": False, "status": "not_found", "error": f"Unknown web interlocutor job {job_id!r}."}
            return {"ok": True, "status": job.status, "job": job.to_dict()}
        return {
            "ok": True,
            "status": "listed",
            "jobs": [job.to_dict() for job in sorted(self._jobs.values(), key=lambda item: item.started_at)],
        }

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        if not job:
            return {"ok": False, "status": "not_found", "error": f"Unknown web interlocutor job {job_id!r}."}
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            job.status = "cancelling"
            job.updated_at = time.time()
            return {"ok": True, "status": "cancelling", "job": job.to_dict()}
        return {"ok": True, "status": job.status, "job": job.to_dict()}


_JOB_MANAGER: WebInterlocutorJobManager | None = None


def get_web_interlocutor_job_manager() -> WebInterlocutorJobManager:
    global _JOB_MANAGER
    if _JOB_MANAGER is None:
        _JOB_MANAGER = WebInterlocutorJobManager()
    return _JOB_MANAGER


def _as_applescript_string(value: str) -> str:
    text = str(value or "")
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    return f'"{text}"'


def _extract_new_interlocutor_text(before: str, after: str, sent_text: str) -> str:
    before_lines = _normalized_lines(before)
    after_lines = _normalized_lines(after)
    sent_norm = _normalize_line(sent_text)
    new_lines: list[str] = []
    before_set = set(before_lines)
    for line in after_lines:
        norm = _normalize_line(line)
        if not norm or norm in before_set:
            continue
        if sent_norm and (norm == sent_norm or sent_norm in norm or norm in sent_norm):
            continue
        if _looks_like_ui_chrome(norm):
            continue
        new_lines.append(line.strip())
    if not new_lines:
        if after.startswith(before):
            tail = after[len(before) :].strip()
            return _meaningful_reply_or_empty(_trim_reply_text(tail, sent_text), sent_text)
        return ""
    return _meaningful_reply_or_empty(_trim_reply_text("\n".join(new_lines[-24:]), sent_text), sent_text)


def _normalized_lines(text: str) -> list[str]:
    return [_normalize_line(line) for line in str(text or "").splitlines() if _normalize_line(line)]


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", str(line or "").strip()).lower()


def _looks_like_ui_chrome(norm: str) -> bool:
    if len(norm) <= 2:
        return True
    if re.fullmatch(r"https?://\S+", norm) or re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}(/\S*)?", norm):
        return True
    if re.fullmatch(r"(mon|tue|wed|thu|fri|sat|sun)\s+[a-z]{3}\s+\d{1,2}\s+\d{1,2}:\d{2}\s*(am|pm)?", norm):
        return True
    if re.fullmatch(r"\d{1,2}:\d{2}\s*(am|pm)?", norm):
        return True
    browser_menu = {
        "chrome",
        "file",
        "edit",
        "view",
        "history",
        "bookmarks",
        "profiles",
        "tab",
        "window",
        "help",
    }
    words = set(re.findall(r"[a-z]+", norm))
    if words and words.issubset(browser_menu):
        return True
    markers = (
        "ask anything",
        "chatgpt can make mistakes",
        "edit view",
        "follow up",
        "thought for",
        "thinking",
        "write or type / for commands",
        "new chat",
        "send message",
        "message chatgpt",
        "message gemini",
        "sign in",
        "terms",
        "privacy policy",
        "regenerate",
        "copy",
        "share",
        "upgrade",
        "menu",
        "search",
        "loading",
    )
    return any(marker == norm or norm.startswith(marker + " ") or norm.startswith(marker + ".") for marker in markers)


def _screen_text_suggests_chat_composer(text: str) -> bool:
    lowered = str(text or "").lower()
    markers = (
        "ask anything",
        "message",
        "send a message",
        "message chatgpt",
        "ask gemini",
        "enter a prompt",
        "type a message",
        "reply",
    )
    return any(marker in lowered for marker in markers)


_NON_REPLY_WORDS = {
    "chrome",
    "file",
    "edit",
    "view",
    "history",
    "bookmarks",
    "profiles",
    "tab",
    "window",
    "help",
    "chatgpt",
    "gemini",
    "search",
    "library",
    "projects",
    "scheduled",
    "apps",
    "recents",
    "plus",
    "high",
    "image",
    "write",
    "look",
    "something",
    "anything",
    "mon",
    "tue",
    "wed",
    "thu",
    "fri",
    "sat",
    "sun",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
}


def _meaningful_reply_or_empty(text: str, sent_text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    sent_norm = _normalize_line(sent_text)
    lines = []
    for line in cleaned.splitlines():
        norm = _normalize_line(line)
        if not norm or _looks_like_ui_chrome(norm):
            continue
        if sent_norm and (norm == sent_norm or sent_norm in norm or norm in sent_norm):
            continue
        lines.append(line.strip())
    cleaned = "\n".join(lines).strip()
    if len(cleaned) < 32:
        return ""
    words = re.findall(r"[a-zA-Z][a-zA-Z']{2,}", cleaned.lower())
    content_words = [word for word in words if word not in _NON_REPLY_WORDS]
    if len(content_words) < 5:
        return ""
    return cleaned


def _rough_text_contains(haystack: str, needle: str) -> bool:
    hay_norm = _normalize_line(haystack)
    needle_norm = _normalize_line(needle)
    if not hay_norm or not needle_norm:
        return False
    if needle_norm in hay_norm:
        return True
    words = [
        word
        for word in re.findall(r"[a-zA-Z][a-zA-Z']{3,}", needle_norm.lower())
        if word not in _NON_REPLY_WORDS
    ]
    if not words:
        return False
    unique_words = list(dict.fromkeys(words[:18]))
    hits = sum(1 for word in unique_words if word in hay_norm)
    required = min(7, max(4, len(unique_words) // 2))
    return hits >= required


def _message_is_substantive(text: str) -> bool:
    cleaned = str(text or "").strip()
    if len(cleaned) < _MIN_OUTBOUND_MESSAGE_CHARS:
        return False
    if _normalize_line(cleaned) in {"false", "true", "none", "null", "nil", "0", "1"}:
        return False
    words = re.findall(r"[a-zA-Z][a-zA-Z']{2,}", cleaned.lower())
    content_words = [word for word in words if word not in _NON_REPLY_WORDS]
    return len(content_words) >= 5


def _message_was_recently_sent(message: str, turns: list[WebInterlocutorTurn]) -> bool:
    norm = _normalize_line(message)
    if not norm:
        return True
    for turn in turns[-3:]:
        sent_norm = _normalize_line(turn.sent)
        if sent_norm == norm:
            return True
    return False


def _url_allows_readability_fallback(url: str) -> bool:
    lowered = str(url or "").lower()
    if not lowered.startswith(("http://", "https://")):
        return False
    blocked_hosts = (
        "chatgpt.com",
        "gemini.google.com",
        "claude.ai",
        "x.com",
        "twitter.com",
        "accounts.google.com",
        "google.com/search",
    )
    return not any(host in lowered for host in blocked_hosts)


def _url_looks_visible_chat_surface(url: str) -> bool:
    lowered = str(url or "").lower()
    return any(
        host in lowered
        for host in (
            "chatgpt.com",
            "gemini.google.com",
            "claude.ai",
            "poe.com",
            "copilot.microsoft.com",
            "deepseek.com",
            "meta.ai",
        )
    )


def _same_origin_or_exact_url(current_url: str, desired_url: str) -> bool:
    current = str(current_url or "").strip()
    desired = str(desired_url or "").strip()
    if not desired:
        return True
    if not current:
        return False
    if current.rstrip("/") == desired.rstrip("/"):
        return True
    try:
        current_parts = urllib.parse.urlparse(current)
        desired_parts = urllib.parse.urlparse(desired)
    except (TypeError, ValueError):
        return False
    if not current_parts.scheme or not desired_parts.scheme:
        return False
    return (
        current_parts.scheme in {"http", "https"}
        and desired_parts.scheme in {"http", "https"}
        and current_parts.netloc.lower() == desired_parts.netloc.lower()
    )


def _trim_reply_text(text: str, sent_text: str) -> str:
    cleaned = str(text or "").strip()
    sent = str(sent_text or "").strip()
    if sent and cleaned.startswith(sent):
        cleaned = cleaned[len(sent) :].strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    if len(cleaned) > _MAX_REPLY_CHARS:
        cleaned = cleaned[-_MAX_REPLY_CHARS:].strip()
    return cleaned


def _render_transcript(turns: list[WebInterlocutorTurn]) -> str:
    chunks = []
    for turn in turns:
        chunks.append(f"Aura {turn.index}: {turn.sent}")
        chunks.append(f"Interlocutor {turn.index}: {turn.observed_reply}")
    return "\n\n".join(chunks)


async def _maybe_think(engine: Any, prompt: str, context: dict[str, Any]) -> str:
    if engine is None:
        return ""
    try:
        if hasattr(engine, "think"):
            result = engine.think(prompt, context={**context, "origin": "web_interlocutor"})
            if asyncio.iscoroutine(result):
                result = await result
        elif hasattr(engine, "generate"):
            result = engine.generate(prompt=prompt, context=context)
            if asyncio.iscoroutine(result):
                result = await result
        else:
            return ""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return str(result.get("response") or result.get("text") or result.get("content") or "")
        return str(getattr(result, "response", "") or getattr(result, "text", "") or "")
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation(
            "web_interlocutor.cognitive_compose",
            exc,
            severity="warning",
            action="used deterministic web-interlocutor message composition after cognitive compose failed",
        )
        return ""


def _clean_message(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^next message:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = cleaned.strip("` \n")
    if _normalize_line(cleaned) in {"false", "true", "none", "null", "nil", "0", "1"}:
        return ""
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0]
    return " ".join(lines[:4])


def _deterministic_learning_summary(objective: str, turns: list[WebInterlocutorTurn]) -> str:
    observed = " ".join(turn.observed_reply for turn in turns).strip()
    observed = re.sub(r"\s+", " ", observed)
    if len(observed) > 1200:
        observed = observed[:1200].rsplit(" ", 1)[0] + "..."
    return (
        f"Visible web interlocutor conversation for objective: {objective or 'substantive dialogue'}. "
        f"Observed interlocutor content: {observed}"
    ).strip()


__all__ = [
    "BrowserPageSnapshot",
    "ChromeVisibleDialogueBrowser",
    "WebDialogueBrowser",
    "WebInterlocutorResult",
    "WebInterlocutorJob",
    "WebInterlocutorJobManager",
    "WebInterlocutorSession",
    "WebInterlocutorTurn",
    "get_web_interlocutor_job_manager",
]
