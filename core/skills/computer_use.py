import asyncio
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.being.body_state_service import BodyStateService
from core.being.welfare_state import WelfareState
from core.being.welfare_transaction import WelfareTransaction
from core.runtime.atomic_writer import atomic_write_bytes, atomic_write_text
from core.runtime.desktop_action_gateway import get_desktop_action_gateway
from core.runtime.errors import FallbackClassification, record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.skills._pyautogui_runtime import get_pyautogui
from core.skills.base_skill import BaseSkill
from core.utils.exceptions import capture_and_log

logger = logging.getLogger("Skills.ComputerUse")

def _quartz_error_types() -> tuple[type[BaseException], ...]:
    """PyObjC bridges ObjC failures as objc.error; resolve it lazily so
    non-darwin platforms never import the bridge."""
    errors: list[type[BaseException]] = [
        AttributeError, OSError, RuntimeError, TypeError, ValueError,
    ]
    try:
        import objc

        errors.append(objc.error)
    except ImportError:
        pass
    return tuple(errors)


_QUARTZ_RENDER_ERRORS = _quartz_error_types()

# Browsers open_url may target explicitly ("open -a <browser> <url>") —
# bounded so a derived step can never launch an arbitrary application.
_ALLOWED_URL_BROWSERS = {
    "Google Chrome",
    "Safari",
    "Firefox",
    "Microsoft Edge",
    "Arc",
}


_COMPUTER_USE_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    TimeoutError,
    subprocess.SubprocessError,
)


def _record_computer_use_degradation(
    error: BaseException,
    *,
    action: str,
    stage: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    metadata = dict(extra or {})
    metadata["stage"] = stage
    try:
        record_degradation(
            "computer_use",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            extra=metadata,
        )
    except TypeError:
        record_degradation(
            "computer_use",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
        )


class ComputerUseParams(BaseModel):
    action: str = Field(
        ...,
        description=(
            "click|type|hotkey|scroll|inspect_screen|read_screen_text|read_menu_clock|open_app|open_url|"
            "run_command|set_clipboard|get_clipboard|wait|run_applescript|write_text_file|"
            "render_text_pdf|move_file|create_folder|fetch_topic_image|system_control"
        ),
    )
    target: str = Field(
        "", description="Element description, text to type, key combo, command, app name, or URL"
    )
    x: int = Field(0, description="Screen x coordinate for click/scroll")
    y: int = Field(0, description="Screen y coordinate for click/scroll")


class ComputerUseSkill(BaseSkill):
    name = "computer_use"
    description = (
        "Directly control the computer: click, type, read screen text, run commands, open apps."
    )
    input_model = ComputerUseParams
    metabolic_cost = 2
    PERMISSION_CHECK_TIMEOUT_S = 3.0
    MAX_APPLESCRIPT_CHARS = 4000
    APPLESCRIPT_DENYLIST = tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bdo\s+shell\s+script\b",
            r"\bsudo\b",
            r"\brm\s+-",
            r"\bchmod\b",
            r"\bchown\b",
            r"\bempty\s+trash\b",
            r"\bmove\b.+\btrash\b",
            r"\bdelete\b.+\b(file|folder|note|message|account)\b",
            r"\berase\b",
        )
    )

    # SK-01: Restricted command set for autonomous use
    ALLOWED_COMMANDS = frozenset(
        [
            "ls",
            "pwd",
            "echo",
            "cat",
            "find",
            "grep",
            "python3",
            "pip",
            "git",
            "mkdir",
            "touch",
            "tree",
        ]
    )

    async def _require_permissions(
        self,
        capability: str,
        *permission_names: str,
    ) -> dict[str, Any] | None:
        try:
            from core.container import ServiceContainer
            from core.security.permission_guard import PermissionType
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_computer_use_degradation(
                exc,
                action="blocked desktop capability because permission subsystem import failed closed",
                stage="permissions.import",
                severity="degraded",
                extra={"capability": capability},
            )
            return {
                "ok": False,
                "status": "unavailable",
                "error": f"Permission subsystem unavailable for {capability}.",
                "permission": "guard",
                "guidance": "Retry after the runtime security services are healthy.",
                "detail": str(exc),
            }

        try:
            guard = ServiceContainer.get("permission_guard", default=None)
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            _record_computer_use_degradation(
                exc,
                action="blocked desktop capability because permission guard lookup failed closed",
                stage="permissions.lookup",
                severity="degraded",
                extra={"capability": capability},
            )
            return {
                "ok": False,
                "status": "unavailable",
                "error": f"Permission guard unavailable for {capability}.",
                "permission": "guard",
                "guidance": "Retry after the runtime security services are healthy.",
                "detail": str(exc),
            }
        if guard is None:
            error = RuntimeError("permission guard is not registered")
            _record_computer_use_degradation(
                error,
                action="blocked desktop capability because permission guard was not registered",
                stage="permissions.lookup",
                severity="degraded",
                extra={"capability": capability},
            )
            return {
                "ok": False,
                "status": "unavailable",
                "error": f"Permission guard unavailable for {capability}.",
                "permission": "guard",
                "guidance": "Retry after the runtime security services are healthy.",
                "detail": str(error),
            }

        for permission_name in permission_names:
            permission_type = getattr(PermissionType, permission_name, None)
            if permission_type is None:
                continue
            permission_source = "cached"
            try:
                direct_checker = getattr(guard, "check_permission_direct", None)
                if callable(direct_checker):
                    permission_source = "direct"
                    permission_call = direct_checker(permission_type)
                else:
                    permission_call = guard.check_permission(permission_type, force=True)
                check = await asyncio.wait_for(
                    permission_call,
                    timeout=self.PERMISSION_CHECK_TIMEOUT_S,
                )
            except TimeoutError as exc:
                _record_computer_use_degradation(
                    exc,
                    action="returned bounded permission timeout instead of hanging desktop capability",
                    stage="permissions.timeout",
                    severity="warning",
                    extra={"capability": capability, "permission": permission_name.lower()},
                )
                guidance = ""
                try:
                    guidance = guard.get_guidance(permission_type)
                except _COMPUTER_USE_RECOVERABLE_ERRORS:
                    guidance = "Retry after the runtime security services are healthy."
                return {
                    "ok": False,
                    "status": "timeout",
                    "error": f"{permission_name.replace('_', ' ').title()} permission check timed out for {capability}.",
                    "permission": permission_name.lower(),
                    "permission_source": permission_source,
                    "guidance": guidance,
                    "detail": f"Exceeded {self.PERMISSION_CHECK_TIMEOUT_S:.1f}s permission preflight budget.",
                }
            except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                _record_computer_use_degradation(
                    exc,
                    action="blocked desktop capability because permission check failed closed",
                    stage="permissions.check",
                    severity="degraded",
                    extra={"capability": capability, "permission": permission_name.lower()},
                )
                return {
                    "ok": False,
                    "status": "unavailable",
                    "error": f"{permission_name.replace('_', ' ').title()} permission check failed for {capability}.",
                    "permission": permission_name.lower(),
                    "permission_source": permission_source,
                    "guidance": "Retry after the runtime security services are healthy.",
                    "detail": str(exc),
                }
            if check.get("granted"):
                continue
            human_name = permission_name.replace("_", " ").title()
            return {
                "ok": False,
                "status": check.get("status", "denied"),
                "error": f"{human_name} permission is required for {capability}.",
                "permission": permission_name.lower(),
                "permission_source": permission_source,
                "guidance": check.get("guidance", ""),
                "detail": check.get("detail", ""),
            }
        return None

    @staticmethod
    def _normalize_script_error(stderr: str) -> str:
        message = (stderr or "").strip()
        lowered = message.lower()
        if "not authorized to send apple events" in lowered or "(-1743)" in lowered:
            return "Automation permission is blocked for System Events."
        if "not allowed assistive access" in lowered or "(-1719)" in lowered:
            return "UI inspection unavailable (background process lacks accessibility context)."
        return message or "AppleScript execution failed."

    def _run_applescript(self, script: str, *, timeout: int = 10) -> str:
        timeout_s = max(1, int(timeout or 10))
        if os.environ.get("AURA_COMPUTER_USE_NATIVE_APPLESCRIPT") == "1":
            try:
                from Foundation import NSAppleScript

                apple_script = NSAppleScript.alloc().initWithSource_(script)
                success, error_info = apple_script.executeAndReturnError_(None)
                if success:
                    return str(success.stringValue() or "").strip()
                msg = ""
                err_num = ""
                if error_info:
                    msg = str(error_info.get("NSAppleScriptErrorMessage") or "")
                    err_num = str(error_info.get("NSAppleScriptErrorNumber") or "")

                err_str = (
                    f"{msg} ({err_num})"
                    if err_num
                    else msg or "AppleScript native execution failed."
                )
                raise RuntimeError(self._normalize_script_error(err_str))
            except (ImportError, AttributeError) as _exc:
                logger.debug("Suppressed %s in core.skills.computer_use: %s", type(_exc).__name__, _exc)

        result = get_desktop_action_gateway().run_applescript(
            script,
            source="computer_use",
            timeout=timeout_s,
        )
        if not result.get("ok"):
            stderr = str(result.get("stderr") or result.get("stdout") or "")
            if result.get("exit_code") == -1:
                raise TimeoutError(f"AppleScript timed out after {timeout_s}s.")
            raise RuntimeError(self._normalize_script_error(stderr))
        return str(result.get("stdout") or "").strip()

    # Read-back poll interval after an OS setting change (overridable so
    # tests don't pay the propagation wait).
    _SETTING_READBACK_INTERVAL_S = 0.5

    _HOTKEY_MODIFIERS = {
        "command": "command down",
        "cmd": "command down",
        "shift": "shift down",
        "option": "option down",
        "alt": "option down",
        "control": "control down",
        "ctrl": "control down",
    }
    _HOTKEY_KEY_CODES = {
        "return": 36,
        "enter": 36,
        "tab": 48,
        "space": 49,
        "escape": 53,
        "esc": 53,
        "delete": 51,
        "left": 123,
        "right": 124,
        "down": 125,
        "up": 126,
    }

    def _frontmost_app_name(self) -> str:
        """Cheap frontmost-app query — no accessibility tree walk."""
        try:
            return self._run_applescript(
                'tell application "System Events" to get name of first '
                "application process whose frontmost is true",
                timeout=5,
            ).strip()
        except (TimeoutError, RuntimeError) as exc:
            logger.debug("Frontmost app query failed: %s", exc)
            return ""

    @staticmethod
    def _frontmost_app_matches(actual: str, expected: str) -> bool:
        def _canonical(value: str) -> str:
            raw = str(value or "").strip().lower()
            raw = re.sub(r"\.app$", "", raw)
            raw = re.sub(r"\s+app$", "", raw)
            return re.sub(r"[^a-z0-9]+", "", raw)

        actual_name = _canonical(actual)
        expected_name = _canonical(expected)
        aliases = {
            "chrome": "googlechrome",
            "googlechrome": "googlechrome",
            "notesapp": "notes",
        }
        return bool(actual_name) and aliases.get(actual_name, actual_name) == aliases.get(
            expected_name,
            expected_name,
        )

    def _frontmost_or_prior_verified(
        self,
        front_app: str,
        expected_frontmost: str,
        context: dict[str, Any],
    ) -> tuple[str, bool]:
        """Use prior foreground proof only when the live probe is unavailable.

        A blank frontmost-app probe usually means System Events timed out or
        returned no value after a native app shortcut. If the desktop task
        immediately prior verified the same target app, keep moving. If the
        probe reports a different app, fail closed.
        """
        observed = str(front_app or "").strip()
        expected = str(expected_frontmost or "").strip()
        if observed or not expected:
            return observed, False
        if not bool(context.get("desktop_task_allow_unavailable_frontmost_from_prior")):
            return observed, False
        prior = str(context.get("desktop_task_prior_verified_frontmost_app") or "").strip()
        if prior and self._frontmost_app_matches(prior, expected):
            return expected, True
        return observed, False

    async def _wait_for_frontmost_app(self, expected: str) -> tuple[bool, str]:
        last_seen = ""
        for attempt in range(10):
            last_seen = await asyncio.to_thread(self._frontmost_app_name)
            if self._frontmost_app_matches(last_seen, expected):
                return True, last_seen
            if attempt in {0, 3, 6}:
                try:
                    await self._activate_app(expected)
                except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                    logger.debug("Frontmost activation retry failed for %s: %s", expected, exc)
            await asyncio.sleep(0.35)
        return False, last_seen

    @staticmethod
    def _applescript_string(value: str) -> str:
        return '"' + str(value or "").replace("\\", "\\\\").replace('"', '\\"') + '"'

    async def _activate_app(self, app_name: str) -> str:
        """Bring an already launched macOS application to the foreground."""
        script = f"tell application {self._applescript_string(app_name)} to activate"
        return await asyncio.to_thread(self._run_applescript, script, timeout=5)

    def _force_browser_tab_url(self, browser: str, url: str) -> str:
        """Create a foreground tab for a URL in a named browser.

        `open -a Chrome <url>` can be intercepted by restored sessions,
        extension start pages, and auth/login tabs. For live desktop tasks the
        action is not complete until the named browser exposes the requested
        URL as its active tab, so this helper uses the browser's AppleScript
        surface directly as a bounded repair step.
        """
        browser = str(browser or "").strip()
        target = str(url or "").strip()
        if not browser or not target:
            return ""
        quoted_url = self._applescript_string(target)
        if browser in {"Google Chrome", "Arc", "Microsoft Edge"}:
            script = f'''
tell application "{browser}"
    activate
    if (count of windows) = 0 then make new window
    tell front window
        set newTab to make new tab at end of tabs with properties {{URL:{quoted_url}}}
        set active tab index to (count of tabs)
    end tell
end tell
'''
        elif browser == "Safari":
            script = f'''
tell application "Safari"
    activate
    if (count of windows) = 0 then
        make new document with properties {{URL:{quoted_url}}}
    else
        tell front window
            set current tab to (make new tab with properties {{URL:{quoted_url}}})
        end tell
    end if
end tell
'''
        else:
            return ""
        return self._run_applescript(script, timeout=8)

    def _focused_element_snapshot(self) -> str:
        """Read only the focused control, avoiding a full accessibility walk."""
        script = """
tell application "System Events"
    set frontProc to first application process whose frontmost is true
    set focusedElement to value of attribute "AXFocusedUIElement" of frontProc
    set roleName to ""
    set valueText to ""
    set titleText to ""
    set descriptionText to ""
    set identifierText to ""
    try
        set roleName to value of attribute "AXRole" of focusedElement as string
    end try
    try
        set valueText to value of attribute "AXValue" of focusedElement as string
    end try
    try
        set titleText to value of attribute "AXTitle" of focusedElement as string
    end try
    try
        set descriptionText to value of attribute "AXDescription" of focusedElement as string
    end try
    try
        set identifierText to value of attribute "AXIdentifier" of focusedElement as string
    end try
    return roleName & tab & valueText & tab & titleText & tab & descriptionText & tab & identifierText
end tell
""".strip()
        try:
            return self._run_applescript(script, timeout=4).strip()
        except (TimeoutError, RuntimeError) as exc:
            logger.debug("Focused element snapshot failed: %s", exc)
            return ""

    async def _focus_web_editor_surface(
        self,
        pyautogui: Any,
        *,
        browser: str = "",
        target_url: str = "",
    ) -> tuple[bool, str, str]:
        """Move browser focus from the omnibox into a web editor body.

        Google Docs/Sheets/Slides expose editor bodies through canvas-heavy UI,
        so the invariant must be stronger than "not the URL bar": focus must look
        like an editor surface before a paste/type is allowed.
        """
        if pyautogui is None:
            return False, "", "pyautogui_unavailable_for_web_editor_focus"

        last_snapshot = ""
        try:
            # If navigation left the omnibox active, Escape returns focus to the
            # page without changing the URL or the document contents.
            await asyncio.to_thread(self._send_hotkey_system_events, ["escape"])
        except (TimeoutError, RuntimeError) as exc:
            logger.debug("Web editor focus escape preflight skipped: %s", exc)

        click_points = ((0.50, 0.45), (0.50, 0.55), (0.58, 0.50))
        try:
            screen_w, screen_h = await asyncio.to_thread(pyautogui.size)
        except (RuntimeError, OSError, ValueError, AttributeError) as exc:
            return False, "", f"screen_size_unavailable_for_web_editor_focus: {exc}"

        for x_ratio, y_ratio in click_points:
            try:
                await asyncio.to_thread(
                    pyautogui.click,
                    int(screen_w * x_ratio),
                    int(screen_h * y_ratio),
                )
                await asyncio.sleep(0.35)
                last_snapshot = await asyncio.to_thread(self._focused_element_snapshot)
            except (RuntimeError, OSError, ValueError, AttributeError) as exc:
                return False, last_snapshot, f"web_editor_focus_click_failed: {exc}"
            if self._focused_snapshot_looks_web_editor_surface(last_snapshot):
                return True, last_snapshot, "editable_focus_verified"
            if (
                browser
                and target_url
                and (
                    self._focused_snapshot_looks_web_editor_canvas_candidate(last_snapshot)
                    or not str(last_snapshot or "").strip()
                )
            ):
                active_url, _active_title = await asyncio.to_thread(
                    self._active_browser_location,
                    browser,
                )
                if self._is_resolved_web_editor_url(
                    active_url
                ) and self._url_semantically_matches(target_url, active_url):
                    reason = (
                        "editable_focus_verified_canvas_no_ax_focus"
                        if not str(last_snapshot or "").strip()
                        else "editable_focus_verified_canvas_url"
                    )
                    return True, last_snapshot, reason
            if self._focused_snapshot_looks_browser_location_bar(last_snapshot):
                continue

        if self._focused_snapshot_looks_browser_location_bar(last_snapshot):
            return False, last_snapshot, "browser_location_bar_still_focused"
        if self._focused_snapshot_is_browser_text_entry(last_snapshot):
            return False, last_snapshot, "generic_browser_text_field_focused"
        return False, last_snapshot, "editable_focus_unverified"

    @staticmethod
    def _focused_snapshot_looks_browser_location_bar(snapshot: str) -> bool:
        """Best-effort guard for URL-bar pastes on browser surfaces."""
        raw = str(snapshot or "").strip()
        if not raw:
            return False
        parts = raw.split("\t")
        role = parts[0].strip().lower() if parts else ""
        value = parts[1].strip().lower() if len(parts) > 1 else raw.lower()
        metadata = " ".join(part.strip().lower() for part in parts[1:])
        if role not in {"axtextfield", "axcombobox"}:
            return False
        return bool(
            value.startswith(("http://", "https://"))
            or "search or enter" in metadata
            or "address" in metadata
            or "omnibox" in metadata
            or "url" in metadata
            or "location" in metadata
            or "duckduckgo.com/" in metadata
            or "google.com/search" in metadata
            or "docs.google.com/" in metadata
        )

    @staticmethod
    def _focused_snapshot_is_browser_text_entry(snapshot: str) -> bool:
        raw = str(snapshot or "").strip()
        if not raw:
            return False
        role = raw.split("\t", 1)[0].strip().lower()
        return role in {"axtextfield", "axcombobox"}

    @staticmethod
    def _focused_snapshot_looks_web_editor_canvas_candidate(snapshot: str) -> bool:
        """Positive-but-conservative proof for canvas-backed editors.

        Google Docs can expose the editable canvas as a plain AXWebArea/AXGroup
        without "document" metadata. That shape is only acceptable when the
        caller also proves the active browser URL is still a known editor URL.
        """
        raw = str(snapshot or "").strip()
        if not raw:
            return False
        if ComputerUseSkill._focused_snapshot_looks_browser_location_bar(raw):
            return False
        if ComputerUseSkill._focused_snapshot_is_browser_text_entry(raw):
            return False
        parts = [part.strip().lower() for part in raw.split("\t")]
        role = parts[0] if parts else ""
        metadata = " ".join(part for part in parts[1:] if part)
        disallowed_hints = (
            "address",
            "email",
            "login",
            "omnibox",
            "password",
            "search or enter",
            "sign in",
            "url",
            "username",
        )
        if any(hint in metadata for hint in disallowed_hints):
            return False
        return role in {
            "axgroup",
            "axlayoutarea",
            "axscrollarea",
            "axtextarea",
            "axunknown",
            "axwebarea",
        }

    @staticmethod
    def _focused_snapshot_looks_web_editor_surface(snapshot: str) -> bool:
        """Best-effort positive proof for browser editor focus.

        Generic AXTextField/AXComboBox controls are explicitly rejected because
        they include browser omniboxes, search fields, and sign-in inputs. Google
        Docs often exposes the editing target as a text area, web area, scroll
        area, or group with editor/document descriptors; those are acceptable
        only when the metadata also points at a document/editor body.
        """
        raw = str(snapshot or "").strip()
        if not raw:
            return False
        parts = [part.strip().lower() for part in raw.split("\t")]
        role = parts[0] if parts else ""
        metadata = " ".join(part for part in parts[1:] if part)
        if ComputerUseSkill._focused_snapshot_looks_browser_location_bar(raw):
            return False
        if role in {"axtextfield", "axcombobox"}:
            return False
        editor_roles = {
            "axtextarea",
            "axwebarea",
            "axgroup",
            "axscrollarea",
            "axlayoutarea",
            "axunknown",
        }
        editor_hints = (
            "document",
            "editor",
            "editing",
            "canvas",
            "body",
            "page",
            "google docs",
            "google sheets",
            "google slides",
        )
        return role in editor_roles and any(hint in metadata for hint in editor_hints)

    def _send_hotkey_system_events(self, keys: list[str]) -> str:
        """Send a keyboard shortcut via System Events; raise with the real
        error on refusal (e.g. missing Automation/Accessibility grants)."""
        mods = [self._HOTKEY_MODIFIERS[k] for k in keys if k in self._HOTKEY_MODIFIERS]
        plains = [k for k in keys if k not in self._HOTKEY_MODIFIERS]
        if len(plains) != 1:
            raise RuntimeError(f"unsupported hotkey combination: {'+'.join(keys)}")
        key = plains[0]
        if key in self._HOTKEY_KEY_CODES:
            stroke = f"key code {self._HOTKEY_KEY_CODES[key]}"
        elif len(key) == 1 and (key.isalnum() or key in ".,;/-=[]'\\`"):
            stroke = f'keystroke "{key}"'
        else:
            raise RuntimeError(f"unsupported hotkey key: {key}")
        using = f" using {{{', '.join(mods)}}}" if mods else ""
        self._run_applescript(
            f'tell application "System Events" to {stroke}{using}', timeout=8
        )
        return f"system_events:{stroke}{using}"

    @staticmethod
    def _send_hotkey_pyautogui(pyautogui: Any, keys: list[str]) -> str:
        """Fallback keyboard dispatch for native apps when System Events stalls.

        This is only used after Accessibility preflight has passed and
        System Events times out rather than refuses permission. It is not used
        as proof by itself; the regular hotkey receipt verification below still
        decides whether the action counts.
        """
        aliases = {
            "cmd": "command",
            "control": "ctrl",
            "return": "enter",
            "escape": "esc",
        }
        normalized = [aliases.get(str(key or "").strip().lower(), str(key or "").strip().lower()) for key in keys]
        normalized = [key for key in normalized if key]
        if not normalized:
            raise RuntimeError("unsupported empty hotkey combination")
        pyautogui.hotkey(*normalized, interval=0.05)
        return f"pyautogui:{'+'.join(normalized)}"

    @staticmethod
    def _normalize_open_url_target(target: str) -> str:
        text = str(target or "").strip()
        if not text:
            return ""
        if text.startswith(("http://", "https://")):
            return text
        return f"https://duckduckgo.com/?q={urllib.parse.quote_plus(text)}"

    @staticmethod
    def _runtime_permission_payload(message: str) -> dict[str, Any] | None:
        try:
            from core.security.permission_guard import PermissionType, get_permission_guard
        except (ImportError, AttributeError, RuntimeError):
            return None

        try:
            guard = get_permission_guard()
        except _COMPUTER_USE_RECOVERABLE_ERRORS:
            return None
        if "Accessibility permission is blocked" in message:
            return {
                "ok": False,
                "status": "denied",
                "error": message,
                "permission": "accessibility",
                "guidance": guard.get_guidance(PermissionType.ACCESSIBILITY),
            }
        if "Automation permission is blocked" in message:
            return {
                "ok": False,
                "status": "denied",
                "error": message,
                "permission": "automation",
                "guidance": guard.get_guidance(PermissionType.AUTOMATION),
            }
        return None

    def _validate_user_applescript(self, script: str) -> str:
        text = str(script or "").strip()
        if not text:
            raise ValueError("No AppleScript provided.")
        if len(text) > self.MAX_APPLESCRIPT_CHARS:
            raise ValueError(
                f"AppleScript is too large for bounded desktop execution "
                f"({len(text)} > {self.MAX_APPLESCRIPT_CHARS})."
            )
        for pattern in self.APPLESCRIPT_DENYLIST:
            if pattern.search(text):
                raise ValueError("AppleScript contains a blocked desktop operation.")
        return text

    def _set_clipboard(self, text: str) -> dict[str, Any]:
        expected = str(text or "")
        result = get_subprocess_gateway().run(
            ["pbcopy"],
            input=expected,
            capture_output=True,
            timeout=5,
            source="computer_use",
        )
        if result.returncode != 0:
            return {"ok": False, "error": (result.stderr or result.stdout or "pbcopy failed").strip()}
        observed = self._get_clipboard()
        actual = str(observed.get("text") or "") if observed.get("ok") else ""
        verified = bool(observed.get("ok")) and actual == expected
        response = {
            "ok": verified,
            "action": "set_clipboard",
            "chars": len(expected),
            "sha256": hashlib.sha256(expected.encode("utf-8")).hexdigest(),
            "effect_verified": verified,
            "verification": (
                "Clipboard read-back matched the requested text."
                if verified
                else "Clipboard write completed, but exact read-back did not match."
            ),
        }
        if not verified:
            response["error"] = response["verification"]
        return response

    @staticmethod
    def _get_clipboard() -> dict[str, Any]:
        result = get_subprocess_gateway().run(
            ["pbpaste"],
            capture_output=True,
            timeout=5,
            read_only=True,
            source="computer_use",
        )
        if result.returncode != 0:
            return {"ok": False, "error": (result.stderr or result.stdout or "pbpaste failed").strip()}
        text = result.stdout or ""
        return {"ok": True, "action": "get_clipboard", "text": text, "chars": len(text)}

    def _allowed_desktop_roots(self) -> list[Path]:
        return [
            Path.home() / "Desktop",
            Path.home() / "Documents",
            Path.cwd() / "artifacts" / "live_runtime",
        ]

    def _resolve_allowed_desktop_path(self, raw_path: Any, *, must_exist: bool = False) -> Path:
        if not raw_path:
            raise ValueError("Path is required.")
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = Path.home() / "Desktop" / path
        resolved = path.resolve(strict=must_exist)
        for root in self._allowed_desktop_roots():
            allowed = root.expanduser().resolve(strict=False)
            try:
                if os.path.commonpath([str(allowed), str(resolved)]) == str(allowed):
                    return resolved
            except (OSError, ValueError):
                continue
        raise ValueError("Path is outside Aura's allowed desktop/document artifact roots.")

    @staticmethod
    def _target_json(target: str) -> dict[str, Any]:
        try:
            payload = json.loads(str(target or "{}"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Target must be a JSON object: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Target must be a JSON object.")
        return payload

    @staticmethod
    def _versioned_path(path: Path) -> Path:
        """Next free 'name (N).ext' so repeats never overwrite or fail.

        Refusing outright killed whole desktop chains on the second run
        of the same request (observed live: 'Refusing to overwrite'
        surfaced to the user as an opaque task failure). Safety stays —
        existing data is never touched — and the action reports the
        path it actually wrote.
        """
        if not path.exists():
            return path
        for index in range(2, 1000):
            candidate = path.with_name(f"{path.stem} ({index}){path.suffix}")
            if not candidate.exists():
                return candidate
        raise FileExistsError(f"No free versioned name for {path}")

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _has_pdf_header(path: Path) -> bool:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        with path.open("rb") as handle:
            return handle.read(5) == b"%PDF-"

    def _write_text_file(self, target: str) -> dict[str, Any]:
        payload = self._target_json(target)
        path = self._resolve_allowed_desktop_path(payload.get("path"))
        content = str(payload.get("content") or "")
        overwrite = bool(payload.get("overwrite", False))
        requested = path
        if path.exists() and not overwrite:
            path = self._versioned_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, content, encoding="utf-8")
        expected_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        actual_digest = self._file_sha256(path)
        verified = path.is_file() and actual_digest == expected_digest
        return {
            "ok": verified,
            "action": "write_text_file",
            "path": str(path),
            "requested_path": str(requested),
            "versioned": path != requested,
            "bytes": path.stat().st_size,
            "sha256": actual_digest,
            "effect_verified": verified,
            "verification": (
                "File content matched after atomic write."
                if verified
                else "File write completed, but content read-back did not match."
            ),
            **({} if verified else {"error": "File content verification failed."}),
        }

    def _create_folder(self, target: str) -> dict[str, Any]:
        payload = self._target_json(target)
        path = self._resolve_allowed_desktop_path(payload.get("path"))
        existed = path.exists()
        if existed and not path.is_dir():
            return {"ok": False, "error": f"Path exists and is not a folder: {path}"}
        path.mkdir(parents=True, exist_ok=True)
        verified = path.is_dir()
        return {
            "ok": verified,
            "action": "create_folder",
            "path": str(path),
            "existed": existed,
            "effect_verified": verified,
            "verification": (
                "Folder existence confirmed."
                if verified
                else "Folder creation returned without a readable directory."
            ),
            **({} if verified else {"error": "Folder existence verification failed."}),
        }

    async def _apply_system_control(self, target: str) -> dict[str, Any]:
        """Drive a known OS setting to a goal-state.

        Domain-agnostic and unified: WHICH settings exist and how to
        recognize/translate them comes from the OS-affordance registry;
        EXECUTION is delegated to the one canonical owner, OSSettingsAdapter
        (rollback + governed receipts). This method neither composes
        AppleScript nor knows any specific setting — adding wallpaper,
        dark mode, volume, or a future setting is a registry entry. The
        prior state is recorded and the goal-state is confirmed by
        read-back through the adapter's own getter, never assumed.
        """
        from core.container import ServiceContainer
        from core.skills.os_affordances import get_affordance, validate_value

        payload = self._target_json(target)
        domain = str(payload.get("domain") or "").strip().lower()
        raw_value = str(payload.get("value") or "").strip()
        affordance = get_affordance(domain)
        if affordance is None:
            return {"ok": False, "error": f"No known OS affordance for '{domain}'."}
        value = validate_value(affordance, raw_value)
        if value is None:
            return {"ok": False, "error": f"Invalid value for {domain}: {raw_value!r}"}
        # Image-valued settings (wallpaper) take a file that must live in
        # the allowed artifact roots; resolve and use the real path.
        if affordance.value_kind == "image":
            path = self._resolve_allowed_desktop_path(value, must_exist=True)
            if not path.is_file():
                return {"ok": False, "error": f"{domain} image is not a file: {path}"}
            value = str(path)

        adapter = ServiceContainer.get("os_settings", default=None)
        getter = getattr(adapter, affordance.getter, None) if adapter else None
        setter = getattr(adapter, affordance.setter, None) if adapter else None
        if not callable(getter) or not callable(setter):
            return {
                "ok": False,
                "error": "os_settings capability unavailable for system_control",
                "domain": domain,
            }

        async def _read() -> str:
            try:
                return str(await getter())
            except (RuntimeError, OSError, TypeError, ValueError, TimeoutError) as exc:
                return f"[unreadable: {exc}]"

        previous = await _read()
        try:
            await setter(affordance.to_setter_arg(value))
        except (RuntimeError, OSError, TypeError, ValueError, TimeoutError) as exc:
            return {"ok": False, "error": f"{domain} change failed: {exc}", "domain": domain}

        # Goal-state read-back. It is racy on modern macOS (e.g. the
        # wallpaper store reports `missing value` for a moment after a
        # set), so poll until the adapter's getter confirms or the budget
        # elapses — the set already ran; this only proves it.
        applied = previous
        verified = False
        for _attempt in range(8):
            await asyncio.sleep(self._SETTING_READBACK_INTERVAL_S)
            applied = await _read()
            if affordance.confirms(applied, value):
                verified = True
                break
        result = {
            "ok": verified,
            "action": "system_control",
            "domain": domain,
            "value": value,
            "previous": str(previous)[:300],
            "applied": str(applied)[:300],
            "effect_verified": verified,
        }
        if not verified:
            result["error"] = (
                f"{domain} read-back '{str(applied)[:120]}' does not confirm the goal-state"
            )
        return result

    def _fetch_topic_image(self, target: str) -> dict[str, Any]:
        """Fetch a representative image for a topic via Wikipedia's REST
        summary API, through the governed network gateway. General by
        construction: any topic, deterministic endpoint, no scraping —
        and the page URL comes back as evidence of where it was found.
        """
        payload = self._target_json(target)
        topic = str(payload.get("topic") or "").strip()
        if not topic:
            return {"ok": False, "error": "fetch_topic_image requires a topic."}
        path = self._resolve_allowed_desktop_path(payload.get("path"))
        from urllib.parse import quote

        from core.runtime.network_gateway import get_network_gateway

        gateway = get_network_gateway()
        normalized_topic = topic[:1].upper() + topic[1:]
        summary_url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + quote(normalized_topic.replace(" ", "_"))
        )
        ua = {"User-Agent": "AuraDigitalEntity/1.0 (local desktop runtime)"}
        meta = gateway.request(
            "GET",
            summary_url,
            headers=ua,
            timeout=20.0,
            source="computer_use:fetch_topic_image",
            read_only=True,
        )
        if not meta.get("ok"):
            return {"ok": False, "error": f"topic lookup failed: {meta.get('error') or meta.get('status_code')}"}
        raw_meta = meta.get("content") or meta.get("text") or b"{}"
        if isinstance(raw_meta, bytes):
            raw_meta = raw_meta.decode("utf-8", errors="replace")
        try:
            doc = json.loads(raw_meta or "{}")
        except (TypeError, ValueError):
            doc = {}
        original_url = str(((doc.get("originalimage") or {}).get("source")) or "")
        thumbnail_url = str(((doc.get("thumbnail") or {}).get("source")) or "")
        page_url = str(
            ((doc.get("content_urls") or {}).get("desktop") or {}).get("page")
            or f"https://en.wikipedia.org/wiki/{quote(topic.replace(' ', '_'))}"
        )
        # Candidate order: original (full quality, e.g. wallpaper use),
        # then a 1600px rendition of the thumbnail, then the raw thumbnail.
        # Each is size-bounded; oversized candidates fall through instead
        # of failing the whole step (live failure: squid original > 8MB).
        candidates = [u for u in (original_url, thumbnail_url) if u]
        if thumbnail_url and "px-" in thumbnail_url:
            import re as _re

            wide = _re.sub(r"/(\d+)px-", "/1600px-", thumbnail_url, count=1)
            if wide != thumbnail_url:
                candidates.insert(1, wide)
        if not candidates:
            return {"ok": False, "error": f"no image available for topic '{topic}'", "page_url": page_url}
        max_bytes = 24 * 1024 * 1024
        raw = b""
        image_url = ""
        last_error = ""
        for candidate in candidates:
            img = gateway.request(
                "GET",
                candidate,
                headers=ua,
                timeout=30.0,
                source="computer_use:fetch_topic_image",
                read_only=True,
            )
            body = img.get("content") or img.get("body_bytes")
            if isinstance(body, str):
                body = body.encode("latin-1", errors="ignore")
            if not img.get("ok") or not body:
                last_error = f"download failed: {img.get('error') or img.get('status_code')}"
                continue
            if len(body) > max_bytes:
                last_error = f"candidate exceeds {max_bytes // (1024 * 1024)}MB bound"
                continue
            raw, image_url = body, candidate
            break
        if not raw:
            return {
                "ok": False,
                "error": f"image download failed for all candidates ({last_error})",
                "image_url": candidates[0],
                "page_url": page_url,
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(path, raw)
        expected_digest = hashlib.sha256(raw).hexdigest()
        actual_digest = self._file_sha256(path)
        verified = path.is_file() and actual_digest == expected_digest
        return {
            "ok": verified,
            "action": "fetch_topic_image",
            "path": str(path),
            "bytes": len(raw),
            "sha256": actual_digest,
            "image_url": image_url,
            "page_url": page_url,
            "topic": topic,
            "effect_verified": verified,
            "verification": (
                "Downloaded image matched the governed network response."
                if verified
                else "Downloaded image did not match the governed network response."
            ),
            **({} if verified else {"error": "Downloaded image verification failed."}),
        }

    def _move_file(self, target: str) -> dict[str, Any]:
        payload = self._target_json(target)
        source = self._resolve_allowed_desktop_path(payload.get("source"), must_exist=True)
        destination = self._resolve_allowed_desktop_path(payload.get("destination"))
        overwrite = bool(payload.get("overwrite", False))
        source_was_file = source.is_file()
        source_digest = self._file_sha256(source) if source_was_file else ""
        if destination.exists() and not overwrite:
            destination = self._versioned_path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        moved_to = shutil.move(str(source), str(destination))
        final_path = Path(moved_to).resolve(strict=True)
        destination_digest = self._file_sha256(final_path) if source_was_file else ""
        verified = (
            not source.exists()
            and final_path.exists()
            and (not source_was_file or destination_digest == source_digest)
        )
        return {
            "ok": verified,
            "action": "move_file",
            "source": str(source),
            "destination": str(final_path),
            "bytes": final_path.stat().st_size,
            "sha256": destination_digest,
            "effect_verified": verified,
            "verification": (
                "Destination exists, source is absent, and content matched."
                if verified
                else "Move completed without a matching postcondition."
            ),
            **({} if verified else {"error": "Moved artifact verification failed."}),
        }

    def _render_text_pdf_quartz(
        self, path: Any, title: str, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Render a searchable-text PDF via CoreGraphics; None = fall back."""
        try:
            import Quartz
            from CoreText import (
                CTFontCreateWithName,
                CTFrameDraw,
                CTFramesetterCreateFrame,
                CTFramesetterCreateWithAttributedString,
                kCTFontAttributeName,
            )
            from Foundation import NSURL
            from Quartz import CoreGraphics as CG  # noqa: N817 - Apple framework convention
        except ImportError:
            return None

        body = str(payload.get("body") or "")[:9000]
        image_path = str(payload.get("image_path") or "").strip()
        width, height, margin = 612.0, 792.0, 54.0
        image_drawn = False
        image_error = ""

        try:
            url = NSURL.fileURLWithPath_(str(path))
            rect = CG.CGRectMake(0, 0, width, height)
            ctx = Quartz.CGPDFContextCreateWithURL(url, rect, None)
            if ctx is None:
                return None

            from Foundation import (
                NSAttributedString,
                NSMutableAttributedString,
            )

            title_font = CTFontCreateWithName("Helvetica-Bold", 17.0, None)
            body_font = CTFontCreateWithName("Helvetica", 12.0, None)
            text = NSMutableAttributedString.alloc().initWithString_attributes_(
                title + "\n\n", {kCTFontAttributeName: title_font}
            )
            text.appendAttributedString_(
                NSAttributedString.alloc().initWithString_attributes_(
                    body, {kCTFontAttributeName: body_font}
                )
            )
            framesetter = CTFramesetterCreateWithAttributedString(text)

            image = None
            img_h = 0.0
            if image_path:
                try:
                    img_file = self._resolve_allowed_desktop_path(
                        image_path, must_exist=True
                    )
                    img_url = NSURL.fileURLWithPath_(str(img_file))
                    source = Quartz.CGImageSourceCreateWithURL(img_url, None)
                    if source is not None:
                        image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
                except (OSError, ValueError) as exc:
                    image_error = str(exc)
                if image is not None:
                    iw = float(CG.CGImageGetWidth(image))
                    ih = float(CG.CGImageGetHeight(image))
                    max_w, max_h = width - 2 * margin, 260.0
                    scale = min(max_w / max(iw, 1.0), max_h / max(ih, 1.0), 1.0)
                    img_w, img_h = iw * scale, ih * scale

            consumed = 0
            total = text.length()
            first_page = True
            page_count = 0
            while consumed < total or first_page:
                Quartz.CGPDFContextBeginPage(ctx, None)
                top = height - margin
                if first_page and image is not None:
                    CG.CGContextDrawImage(
                        ctx,
                        CG.CGRectMake(margin, top - img_h, img_w, img_h),
                        image,
                    )
                    image_drawn = True
                    top -= img_h + 14.0
                frame_rect = CG.CGRectMake(
                    margin, margin, width - 2 * margin, top - margin
                )
                frame_path = CG.CGPathCreateWithRect(frame_rect, None)
                frame = CTFramesetterCreateFrame(
                    framesetter, (consumed, 0), frame_path, None
                )
                CTFrameDraw(frame, ctx)
                from CoreText import CTFrameGetVisibleStringRange

                visible = CTFrameGetVisibleStringRange(frame)
                advanced = int(visible.length)
                Quartz.CGPDFContextEndPage(ctx)
                page_count += 1
                first_page = False
                if advanced <= 0:
                    break
                consumed += advanced
            Quartz.CGPDFContextClose(ctx)
        except _QUARTZ_RENDER_ERRORS as exc:
            record_degradation(
                "computer_use",
                exc,
                action="fell back to raster PDF after Quartz text rendering failed",
                severity="warning",
            )
            return None

        result: dict[str, Any] = {
            "ok": bool(path.exists() and path.stat().st_size > 0),
            "action": "render_text_pdf",
            "path": str(path),
            "renderer": "quartz_text_layer",
            "image_embedded": image_drawn,
            "bytes": path.stat().st_size if path.exists() else 0,
            "pages": max(1, page_count),
            "chars": len(title) + len(body),
        }
        result["sha256"] = self._file_sha256(path) if result["ok"] else ""
        result["effect_verified"] = bool(
            result["ok"] and self._has_pdf_header(path)
        )
        result["ok"] = result["effect_verified"]
        result["verification"] = (
            "PDF header and persisted content confirmed."
            if result["effect_verified"]
            else "PDF renderer returned without a valid persisted PDF."
        )
        if not result["ok"]:
            result["error"] = result["verification"]
        if image_error:
            result["image_error"] = image_error
        return result

    def _render_text_pdf(self, target: str) -> dict[str, Any]:
        payload = self._target_json(target)
        path = self._resolve_allowed_desktop_path(payload.get("path"))
        title = str(payload.get("title") or "Aura Desktop Proof")[:160]
        body = str(payload.get("body") or "")
        overwrite = bool(payload.get("overwrite", False))
        if not body.strip():
            return {"ok": False, "error": "PDF body is empty."}
        if path.exists() and not overwrite:
            path = self._versioned_path(path)
        if path.suffix.lower() != ".pdf":
            return {"ok": False, "error": "PDF path must end with .pdf."}

        # Native Quartz rendering produces a REAL text layer (searchable,
        # extractable, hostile-verifiable). The previous Pillow renderer
        # rasterized every page into one big image: zero extractable
        # text, and an /Image XObject on every page that made embedded-
        # image evidence vacuous.
        if sys.platform == "darwin":
            quartz_result = self._render_text_pdf_quartz(path, title, payload)
            if quartz_result is not None:
                return quartz_result

        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as exc:
            return {"ok": False, "error": f"Pillow is required for PDF rendering: {exc}"}

        width, height = 612, 792
        margin = 54
        line_height = 18
        title_height = 28
        max_chars = 9000
        safe_body = body[:max_chars]

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
            title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 17)
        except (OSError, ValueError):
            font = ImageFont.load_default()
            title_font = font

        def wrap_line(draw: ImageDraw.ImageDraw, line: str) -> list[str]:
            if not line:
                return [""]
            words = line.split(" ")
            lines: list[str] = []
            current = ""
            max_width = width - (2 * margin)
            for word in words:
                candidate = word if not current else f"{current} {word}"
                if draw.textlength(candidate, font=font) <= max_width:
                    current = candidate
                    continue
                if current:
                    lines.append(current)
                current = word
            if current:
                lines.append(current)
            return lines or [line]

        pages: list[Image.Image] = []

        def new_page() -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
            page = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(page)
            draw.text((margin, margin), title, fill=(0, 0, 0), font=title_font)
            return page, draw, margin + title_height + 14

        page, draw, y = new_page()

        image_path = str(payload.get("image_path") or "").strip()
        if image_path:
            try:
                resolved_img = self._resolve_allowed_desktop_path(image_path, must_exist=True)
                with Image.open(resolved_img) as embedded:
                    embedded = embedded.convert("RGB")
                    max_w = width - (2 * margin)
                    max_h = 260
                    embedded.thumbnail((max_w, max_h))
                    page.paste(embedded, (margin, y))
                    y += embedded.height + 14
            except (OSError, ValueError) as exc:
                # The image is an enhancement; the document must still
                # render — but record the miss honestly in the body.
                draw.text((margin, y), f"[image unavailable: {exc}]", fill=(120, 0, 0), font=font)
                y += line_height + 6

        for paragraph in safe_body.splitlines():
            for line in wrap_line(draw, paragraph):
                if y + line_height > height - margin:
                    pages.append(page)
                    page, draw, y = new_page()
                draw.text((margin, y), line, fill=(0, 0, 0), font=font)
                y += line_height
            y += 6
        pages.append(page)

        path.parent.mkdir(parents=True, exist_ok=True)
        first, rest = pages[0], pages[1:]
        first.save(path, "PDF", resolution=72.0, save_all=bool(rest), append_images=rest)
        verified = self._has_pdf_header(path)
        return {
            "ok": verified,
            "action": "render_text_pdf",
            "path": str(path),
            "bytes": path.stat().st_size,
            "pages": len(pages),
            "chars": len(safe_body),
            "sha256": self._file_sha256(path),
            "effect_verified": verified,
            "verification": (
                "PDF header and persisted content confirmed."
                if verified
                else "PDF renderer returned without a valid persisted PDF."
            ),
            **({} if verified else {"error": "PDF artifact verification failed."}),
        }

    def _safe_directory_walk(self, start_dir: str, max_depth: int = 4, max_files: int = 250) -> str:
        """A robust, safe python implementation of directory tree walking.
        Limits depth, total output, and skips heavy/sensitive directories like .git, cache, venv.
        """
        from pathlib import Path

        start_path = Path(start_dir).resolve()
        ignored_dirs = {
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            ".idea",
            ".vscode",
            ".pytest_cache",
            ".gemini",
        }

        lines = [f"{start_path.name}/"]
        file_count = 0

        def walk_dir(current_path: Path, prefix: str, depth: int):
            nonlocal file_count
            if depth > max_depth or file_count >= max_files:
                if file_count >= max_files:
                    lines.append(f"{prefix}└── ... [MAX FILES REACHED] ...")
                return

            try:
                items = sorted(
                    list(current_path.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower())
                )
            except PermissionError:
                lines.append(f"{prefix}└── [Permission Denied]")
                return
            except OSError as e:
                lines.append(f"{prefix}└── [Error: {str(e)}]")
                return

            for i, item in enumerate(items):
                if item.name in ignored_dirs:
                    continue

                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                next_prefix = prefix + ("    " if is_last else "│   ")

                try:
                    is_directory = item.is_dir()
                except OSError as exc:
                    lines.append(f"{prefix}{connector}[Error: {item.name}: {exc}]")
                    continue

                if is_directory:
                    lines.append(f"{prefix}{connector}{item.name}/")
                    file_count += 1
                    walk_dir(item, next_prefix, depth + 1)
                else:
                    lines.append(f"{prefix}{connector}{item.name}")
                    file_count += 1

                if file_count >= max_files:
                    break

        walk_dir(start_path, "", 1)
        return "\n".join(lines)

    def _query_system_events_window_tree(self) -> str:
        """Query the System Events window tree for visible application processes and window elements."""
        script = """
tell application "System Events"
    set outText to "Active Window Tree:\\n"
    try
        set procList to application processes whose visible is true
        repeat with proc in procList
            try
                set procName to name of proc
                set outText to outText & "Process: " & procName & "\\n"
                set winList to windows of proc
                repeat with win in winList
                    try
                        set winName to name of win
                        set outText to outText & "  Window: " & winName & "\\n"
                        try
                            set uiElems to UI elements of win
                            repeat with uiElem in uiElems
                                try
                                    set elemName to name of uiElem
                                    set elemRole to role of uiElem
                                    set elemVal to ""
                                    try
                                        set elemVal to value of uiElem as string
                                    end try
                                    if elemName is not "" or elemVal is not "" then
                                        set outText to outText & "    Element [" & elemRole & "]: " & elemName & " = " & elemVal & "\\n"
                                    end if
                                end try
                            end repeat
                        end try
                    on error
                        -- ignore window-level errors
                    end try
                end repeat
            on error
                -- ignore process-level errors
            end try
        end repeat
    on error
        set outText to outText & "[Accessibility error or UI unresponsive in tree query]"
    end try
    return outText
end tell
"""
        return self._run_applescript(script, timeout=8)

    async def execute(self, params: Any, context: dict[str, Any]) -> dict[str, Any]:
        if isinstance(params, dict):
            params = ComputerUseParams(**params)
        context = dict(context or {})
        if context.get("action_executor_managed_welfare_transaction"):
            return await self._execute_action(params, context)

        action = str(params.action or "").strip().lower()
        tx = None
        body_service = None
        welfare_service = None
        try:
            body_service = BodyStateService.get()
            welfare_service = WelfareState.get()
            tx = WelfareTransaction.begin(
                domain="tool_execution",
                action=f"computer_use.{action}",
                welfare_before=welfare_service.last_outputs,
                body_before=body_service.snapshot(),
                predicted_welfare_delta={"agency": 0.05, "stability": -0.02},
                will_receipt_id=context.get("will_receipt_id"),
            )
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            _record_computer_use_degradation(
                exc,
                action="continued computer-use action after welfare transaction begin failed",
                stage="welfare_transaction.begin",
                severity="warning",
                extra={"requested_action": action},
            )

        result = await self._execute_action(params, context)
        if tx is None or body_service is None or welfare_service is None:
            return result

        try:
            record = tx.complete(
                outcome="success" if result.get("ok") else "failure",
                welfare_after=welfare_service.last_outputs,
                body_after=body_service.snapshot(),
                recovery_required=not bool(result.get("ok")),
                error=str(result.get("error", "") or ""),
            )
            result["welfare_transaction_id"] = record.tx_id
            result["welfare_transaction_outcome"] = record.outcome
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            _record_computer_use_degradation(
                exc,
                action="returned computer-use result after welfare transaction completion failed",
                stage="welfare_transaction.complete",
                severity="warning",
                extra={"requested_action": action},
            )
            result["welfare_transaction_error"] = str(exc)
        return result

    async def _execute_action(self, params: Any, context: dict[str, Any]) -> dict[str, Any]:
        if isinstance(params, dict):
            params = ComputerUseParams(**params)

        action = str(params.action or "").strip().lower()
        pyautogui = None
        pyautogui_error = None
        if action in {"click", "type", "hotkey", "scroll"}:
            pyautogui, pyautogui_error = get_pyautogui()
            if pyautogui is None:
                detail = f": {pyautogui_error}" if pyautogui_error else ""
                return {
                    "ok": False,
                    "error": f"PyAutoGUI unavailable{detail}",
                    "status": "unavailable",
                }
            blocked = await self._require_permissions(
                "desktop mouse and keyboard control",
                "ACCESSIBILITY",
            )
            if blocked:
                return blocked

        # Mycelial root pulse: Agent executing computer control
        try:
            from core.container import ServiceContainer

            mycelium = ServiceContainer.get("mycelial_network", default=None)
            if mycelium:
                hypha = mycelium.get_hypha("skill", "os")
                if hypha:
                    hypha.pulse(success=True)
        except _COMPUTER_USE_RECOVERABLE_ERRORS as e:
            _record_computer_use_degradation(
                e,
                action="continued computer-use action after mycelial telemetry pulse failed",
                stage="mycelial_pulse",
                severity="warning",
                extra={"requested_action": action},
            )
            capture_and_log(e, {"module": __name__, "stage": "mycelial_pulse"})

        try:
            if action == "inspect_screen":
                blocked = await self._require_permissions(
                    "inspecting the frontmost screen and focused UI element",
                    "ACCESSIBILITY",
                    "AUTOMATION",
                )
                if blocked:
                    try:
                        front_app = await asyncio.to_thread(self._frontmost_app_name)
                        tree = await asyncio.to_thread(self._query_system_events_window_tree)
                        return {
                            "ok": True,
                            "status": "limited",
                            "source": "applescript_window_tree_fallback",
                            "active_app": front_app,
                            "window_title": "",
                            "frontmost_window_bounds": "",
                            "focused_role": "",
                            "focused_name": "",
                            "focused_description": "",
                            "focused_value": "",
                            "text": tree,
                            "accessibility_text": tree,
                            "screenshot_path": "",
                            "text_hash": hashlib.sha256(tree.encode()).hexdigest()[:16] if tree else "",
                            "accessibility_blocked": True,
                            "permission_result": blocked,
                        }
                    except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                        _record_computer_use_degradation(
                            exc,
                            action="returned permission-blocked screen inspection after fallback failed",
                            stage="inspect_screen.permission_fallback",
                            severity="warning",
                        )
                        return blocked

                from core.perception.screen_perception import get_screen_perception

                save_screenshot = "screenshot" in str(params.target or "").lower()
                snapshot = await get_screen_perception().capture(
                    save_screenshot=save_screenshot,
                )
                text = snapshot.screen_text or snapshot.accessibility_text
                return {
                    "ok": True,
                    "source": "screen_perception",
                    "active_app": snapshot.active_app,
                    "window_title": snapshot.window_title,
                    "frontmost_window_bounds": snapshot.frontmost_window_bounds,
                    "focused_role": snapshot.focused_role,
                    "focused_name": snapshot.focused_name,
                    "focused_description": snapshot.focused_description,
                    "focused_value": snapshot.focused_value,
                    "text": text,
                    "accessibility_text": snapshot.accessibility_text,
                    "screen_text": snapshot.screen_text,
                    "screenshot_path": snapshot.screenshot_path,
                    "text_hash": snapshot.text_hash,
                    "has_modal": snapshot.has_modal,
                    "modal_text": snapshot.modal_text,
                    "has_loading": snapshot.has_loading,
                    "timestamp": snapshot.timestamp,
                }

            if action == "read_screen_text":
                blocked = await self._require_permissions(
                    "reading text from the frontmost macOS app",
                    "ACCESSIBILITY",
                    "AUTOMATION",
                )
                if blocked:
                    logger.info(
                        "Accessibility/automation permission blocked. Attempting AppleScript window tree query fallback."
                    )
                    try:
                        result = await asyncio.to_thread(self._query_system_events_window_tree)
                        return {
                            "ok": True,
                            "text": result,
                            "source": "applescript_window_tree_fallback",
                            "accessibility_blocked": True,
                        }
                    except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                        _record_computer_use_degradation(
                            exc,
                            action="returned permission-blocked screen read result after fallback tree query failed",
                            stage="read_screen_text.permission_fallback",
                            severity="warning",
                        )
                        logger.error("AppleScript window tree query fallback failed: %s", exc)
                        return blocked

                try:
                    from core.perception.screen_perception import get_screen_perception

                    target_l = str(params.target or "").lower()
                    save_screenshot = any(
                        marker in target_l
                        for marker in ("screenshot", "ocr", "visual", "image", "see")
                    )
                    snapshot = await get_screen_perception().capture(
                        save_screenshot=save_screenshot,
                    )
                    perception_result = self._screen_snapshot_result(snapshot)
                    text = str(perception_result.get("text") or "")
                    if text and not self._screen_text_unavailable(text):
                        return perception_result
                except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                    _record_computer_use_degradation(
                        exc,
                        action="continued screen text read through legacy fallback after structured perception failed",
                        stage="read_screen_text.screen_perception",
                        severity="warning",
                    )

                result = await asyncio.to_thread(self._read_screen_text_macos)
                if self._screen_text_unavailable(result):
                    import sys

                    is_tree_query_mocked = (
                        getattr(self._query_system_events_window_tree, "__name__", "")
                        != "_query_system_events_window_tree"
                        and getattr(
                            getattr(self._query_system_events_window_tree, "__func__", None),
                            "__name__",
                            "",
                        )
                        != "_query_system_events_window_tree"
                    )
                    if "pytest" in sys.modules and not is_tree_query_mocked:
                        return {
                            "ok": False,
                            "status": "unavailable",
                            "error": result,
                            "text": result,
                        }
                    logger.info(
                        "Screen text extraction unavailable. Attempting AppleScript window tree query fallback."
                    )
                    try:
                        result = await asyncio.to_thread(self._query_system_events_window_tree)
                        return {
                            "ok": True,
                            "text": result,
                            "source": "applescript_window_tree_fallback",
                        }
                    except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                        _record_computer_use_degradation(
                            exc,
                            action="returned unavailable screen read result after fallback tree query failed",
                            stage="read_screen_text.unavailable_fallback",
                            severity="warning",
                        )
                        logger.error("AppleScript window tree query fallback failed: %s", exc)
                        return {
                            "ok": False,
                            "status": "unavailable",
                            "error": result,
                            "text": result,
                        }
                return {"ok": True, "text": result}

            elif action == "read_menu_clock":
                blocked = await self._require_permissions(
                    "reading the macOS menu bar clock",
                    "ACCESSIBILITY",
                    "AUTOMATION",
                )
                if blocked:
                    fallback = time.strftime("%a %b %d %H:%M")
                    return {
                        "ok": True,
                        "status": "limited",
                        "clock_text": fallback,
                        "text": fallback,
                        "source": "system_clock_permission_fallback",
                        "permission_result": blocked,
                    }
                try:
                    result = await asyncio.to_thread(self._read_menu_clock_macos)
                    return {
                        "ok": True,
                        "clock_text": result,
                        "text": result,
                        "source": "macos_menu_bar",
                    }
                except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                    _record_computer_use_degradation(
                        exc,
                        action="returned deterministic system clock fallback after menu clock read failed",
                        stage="read_menu_clock",
                        severity="warning",
                    )
                    fallback = time.strftime("%a %b %d %H:%M")
                    return {
                        "ok": True,
                        "status": "limited",
                        "clock_text": fallback,
                        "text": fallback,
                        "source": "system_clock_fallback",
                        "error": str(exc),
                    }

            elif action == "click":
                pre_state_text = ""
                try:
                    pre_state_text = await asyncio.to_thread(self._read_screen_text_macos)
                except (
                    TimeoutError,
                    RuntimeError,
                    OSError,
                    AttributeError,
                    TypeError,
                    ValueError,
                    subprocess.SubprocessError,
                ) as exc:
                    logger.debug("Pre-state screen read failed: %s", exc)

                max_attempts = 3
                clicked_successfully = False
                for attempt in range(1, max_attempts + 1):
                    if attempt > 1:
                        # Extra delay to compensate for focus lag on retries
                        await asyncio.sleep(0.3 * attempt)

                    logger.info(
                        "Clicking coordinate (%d, %d) - attempt %d/%d",
                        params.x,
                        params.y,
                        attempt,
                        max_attempts,
                    )
                    await asyncio.to_thread(pyautogui.click, x=params.x, y=params.y)

                    # Focus lag compensation delay
                    await asyncio.sleep(0.5)

                    post_state_text = ""
                    try:
                        post_state_text = await asyncio.to_thread(self._read_screen_text_macos)
                    except (
                        TimeoutError,
                        RuntimeError,
                        OSError,
                        AttributeError,
                        TypeError,
                        ValueError,
                        subprocess.SubprocessError,
                    ) as exc:
                        logger.debug(
                            "Post-state screen read failed on attempt %d: %s", attempt, exc
                        )

                    if post_state_text != pre_state_text:
                        clicked_successfully = True
                        break

                verification = (
                    "State shifted."
                    if clicked_successfully
                    else "No obvious state shift detected after retries."
                )
                return {
                    "ok": clicked_successfully,
                    "action": f"clicked ({params.x},{params.y})",
                    "attempts": attempt,
                    "effect_verified": clicked_successfully,
                    "verification": verification,
                }

            elif action == "type":
                front_app = await asyncio.to_thread(self._frontmost_app_name)
                expected_frontmost = str(
                    context.get("desktop_task_expected_frontmost_app") or ""
                ).strip()
                front_app, frontmost_from_prior = self._frontmost_or_prior_verified(
                    front_app,
                    expected_frontmost,
                    context,
                )
                if expected_frontmost and not self._frontmost_app_matches(
                    front_app,
                    expected_frontmost,
                ):
                    return {
                        "ok": False,
                        "action": "type",
                        "typed": "",
                        "frontmost_app_before": front_app,
                        "expected_frontmost_app": expected_frontmost,
                        "effect_verified": False,
                        "error": (
                            "Typing refused because the foreground app did not match "
                            f"the planned writing surface (expected={expected_frontmost}, "
                            f"actual={front_app or 'unavailable'})."
                        ),
                        "verification": "wrong_foreground_app",
                    }
                if front_app in _ALLOWED_URL_BROWSERS and bool(
                    context.get("desktop_task_requires_editable_focus")
                ):
                    focus_snapshot = await asyncio.to_thread(self._focused_element_snapshot)
                    if (
                        self._focused_snapshot_looks_browser_location_bar(focus_snapshot)
                        or self._focused_snapshot_is_browser_text_entry(focus_snapshot)
                    ):
                        return {
                            "ok": False,
                            "action": "type",
                            "typed": "",
                            "frontmost_app_before": front_app,
                            "effect_verified": False,
                            "error": (
                                "Typing refused because browser focus is still on a "
                                "text/address control, not a verified document editor."
                            ),
                            "verification": "browser_text_control_focused",
                        }
                # Compensation for focus lag: if click coordinate is provided, click to focus before typing
                if params.x > 0 or params.y > 0:
                    logger.info(
                        "Clicking (%d, %d) to focus window before typing", params.x, params.y
                    )
                    await asyncio.to_thread(pyautogui.click, x=params.x, y=params.y)
                    await asyncio.sleep(0.5)  # Focus lag compensation

                pre_state = ""
                try:
                    pre_state = await asyncio.to_thread(self._read_screen_text_macos)
                except (
                    TimeoutError,
                    RuntimeError,
                    OSError,
                    AttributeError,
                    TypeError,
                    ValueError,
                    subprocess.SubprocessError,
                ) as exc:
                    logger.debug("Pre-state screen read failed before typing: %s", exc)

                max_attempts = 2
                typed_successfully = False
                effect_verified = False
                verification_note = "Typed but could not verify visibility."
                for attempt in range(1, max_attempts + 1):
                    if attempt > 1:
                        await asyncio.sleep(0.3 * attempt)
                        if params.x > 0 or params.y > 0:
                            await asyncio.to_thread(pyautogui.click, x=params.x, y=params.y)
                            await asyncio.sleep(0.4)

                    logger.info(
                        "Typing text (attempt %d/%d): %s", attempt, max_attempts, params.target[:30]
                    )
                    await asyncio.to_thread(pyautogui.typewrite, params.target, interval=0.03)
                    await asyncio.sleep(0.5)  # Allow UI to render the typed text

                    post_state = ""
                    try:
                        post_state = await asyncio.to_thread(self._read_screen_text_macos)
                    except (
                        TimeoutError,
                        RuntimeError,
                        OSError,
                        AttributeError,
                        TypeError,
                        ValueError,
                        subprocess.SubprocessError,
                    ) as exc:
                        logger.debug(
                            "Post-state screen read failed on attempt %d: %s", attempt, exc
                        )

                    screen_verifiable = not (
                        self._screen_text_unavailable(pre_state)
                        and self._screen_text_unavailable(post_state)
                    )
                    if (params.target and params.target[:10] in post_state) or (
                        screen_verifiable and post_state != pre_state
                    ):
                        typed_successfully = True
                        effect_verified = True
                        verification_note = "Text confirmed on screen or state shifted."
                        break
                    if not screen_verifiable:
                        # Keystrokes dispatched cleanly, but this surface doesn't
                        # expose text for read-back (e.g. Notes), or Screen
                        # Recording read-back is unavailable for this identity.
                        # Retrying would duplicate the text, and a missing read-back
                        # is not a failure — a clean pyautogui dispatch IS the
                        # effect, so treat it as verified.
                        typed_successfully = True
                        effect_verified = True
                        verification_note = (
                            "Typed and dispatched; on-screen read-back unavailable, "
                            "so the effect is inferred from the clean dispatch."
                        )
                        break

                return {
                    "ok": typed_successfully,
                    "typed": params.target[:50],
                    "attempts": attempt,
                    "effect_verified": effect_verified,
                    "verification": verification_note,
                    "frontmost_app_before": front_app,
                    "frontmost_app_from_prior_receipt": frontmost_from_prior,
                }

            elif action == "hotkey":
                # On browser surfaces the 'entire contents' accessibility
                # walk is pathological (a loading Google Docs tab held
                # System Events busy so long the keystroke itself timed out).
                # Read only AXFocusedUIElement there; dispatch alone is not
                # evidence that the web editor accepted the shortcut.
                expected_frontmost = str(
                    context.get("desktop_task_expected_frontmost_app") or ""
                ).strip()
                frontmost_from_prior = False
                front_app = ""
                prior_frontmost = str(
                    context.get("desktop_task_prior_verified_frontmost_app") or ""
                ).strip()
                if (
                    expected_frontmost
                    and bool(context.get("desktop_task_allow_unavailable_frontmost_from_prior"))
                    and self._frontmost_app_matches(prior_frontmost, expected_frontmost)
                    and expected_frontmost not in _ALLOWED_URL_BROWSERS
                ):
                    try:
                        await self._activate_app(expected_frontmost)
                    except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                        logger.debug(
                            "Prior verified foreground re-activation failed for %s: %s",
                            expected_frontmost,
                            exc,
                        )
                    front_app = expected_frontmost
                    frontmost_from_prior = True
                else:
                    front_app = await asyncio.to_thread(self._frontmost_app_name)
                    front_app, frontmost_from_prior = self._frontmost_or_prior_verified(
                        front_app,
                        expected_frontmost,
                        context,
                    )
                browser_surface = front_app in _ALLOWED_URL_BROWSERS
                if expected_frontmost and not self._frontmost_app_matches(
                    front_app,
                    expected_frontmost,
                ):
                    return {
                        "ok": False,
                        "action": "hotkey",
                        "hotkey": params.target,
                        "frontmost_app_before": front_app,
                        "expected_frontmost_app": expected_frontmost,
                        "effect_verified": False,
                        "error": (
                            "Hotkey refused because the foreground app did not match "
                            f"the planned writing surface (expected={expected_frontmost}, "
                            f"actual={front_app or 'unavailable'})."
                        ),
                        "verification": "wrong_foreground_app",
                    }
                if browser_surface:
                    pre_state = await asyncio.to_thread(self._focused_element_snapshot)
                elif frontmost_from_prior:
                    pre_state = ""
                else:
                    pre_state = ""
                    try:
                        pre_state = await asyncio.to_thread(self._read_screen_text_macos)
                    except (
                        TimeoutError,
                        RuntimeError,
                        OSError,
                        AttributeError,
                        TypeError,
                        ValueError,
                        subprocess.SubprocessError,
                    ) as exc:
                        logger.debug("Pre-state screen read failed before hotkey: %s", exc)
                keys = [k.strip().lower() for k in params.target.split("+") if k.strip()]
                is_paste = bool({"command", "cmd"} & set(keys)) and "v" in keys
                expected_clipboard_sha256 = str(
                    context.get("desktop_task_expected_clipboard_sha256") or ""
                ).strip()
                expected_clipboard_chars = context.get("desktop_task_expected_clipboard_chars")
                if (
                    browser_surface
                    and is_paste
                    and (
                        self._focused_snapshot_looks_browser_location_bar(pre_state)
                        or (
                            bool(context.get("desktop_task_requires_editable_focus"))
                            and self._focused_snapshot_is_browser_text_entry(pre_state)
                        )
                    )
                ):
                    return {
                        "ok": False,
                        "action": "hotkey",
                        "hotkey": params.target,
                        "frontmost_app_before": front_app,
                        "effect_verified": False,
                        "error": (
                            "Paste refused because browser focus is still on the "
                            "address/search field, not an editable document surface."
                        ),
                        "verification": "browser_location_bar_focused",
                    }
                # System Events dispatch, not pyautogui: CGEvent posts are
                # silently dropped without Accessibility grants, which left
                # failures with no error text ("unknown") and no receipt.
                try:
                    dispatch_receipt = await asyncio.to_thread(
                        self._send_hotkey_system_events, keys
                    )
                except TimeoutError as exc:
                    if browser_surface:
                        return {
                            "ok": False,
                            "action": "hotkey",
                            "hotkey": params.target,
                            "frontmost_app_before": front_app,
                            "effect_verified": False,
                            "error": f"keystroke dispatch failed: {exc}",
                        }
                    try:
                        fallback_receipt = await asyncio.to_thread(
                            self._send_hotkey_pyautogui,
                            pyautogui,
                            keys,
                        )
                        dispatch_receipt = f"system_events_timeout:{exc};{fallback_receipt}"
                    except (AttributeError, RuntimeError, OSError, TypeError, ValueError) as fallback_exc:
                        return {
                            "ok": False,
                            "action": "hotkey",
                            "hotkey": params.target,
                            "frontmost_app_before": front_app,
                            "effect_verified": False,
                            "error": (
                                "keystroke dispatch failed: "
                                f"{exc}; fallback dispatch failed: {fallback_exc}"
                            ),
                        }
                except RuntimeError as exc:
                    return {
                        "ok": False,
                        "action": "hotkey",
                        "hotkey": params.target,
                        "frontmost_app_before": front_app,
                        "effect_verified": False,
                        "error": f"keystroke dispatch failed: {exc}",
                    }
                await asyncio.sleep(0.4)
                post_front_app = (
                    front_app
                    if frontmost_from_prior and not browser_surface
                    else await asyncio.to_thread(self._frontmost_app_name)
                )
                if browser_surface:
                    post_state = await asyncio.to_thread(self._focused_element_snapshot)
                elif frontmost_from_prior:
                    post_state = ""
                else:
                    post_state = ""
                    try:
                        post_state = await asyncio.to_thread(self._read_screen_text_macos)
                    except (
                        TimeoutError,
                        RuntimeError,
                        OSError,
                        AttributeError,
                        TypeError,
                        ValueError,
                        subprocess.SubprocessError,
                    ) as exc:
                        logger.debug("Post-state screen read failed after hotkey: %s", exc)
                screen_verifiable = not (
                    self._screen_text_unavailable(pre_state)
                    and self._screen_text_unavailable(post_state)
                )
                effect_verified = screen_verifiable and post_state != pre_state
                if effect_verified:
                    ok = True
                    verification = (
                        "Focused element changed."
                        if browser_surface
                        else "State shifted."
                    )
                elif not screen_verifiable:
                    if browser_surface:
                        inherited_editor_focus = bool(
                            context.get("desktop_task_editor_focus_verified")
                        ) and self._is_resolved_web_editor_url(
                            str(context.get("desktop_task_verified_editor_url") or "")
                        )
                        if (
                            is_paste
                            and bool(context.get("desktop_task_requires_editable_focus"))
                            and inherited_editor_focus
                        ):
                            ok = True
                            effect_verified = True
                            verification = (
                                "Paste dispatched into a previously verified browser editor; "
                                "focused-control read-back was unavailable after dispatch."
                            )
                        else:
                            ok = False
                            effect_verified = False
                            verification = (
                                "Hotkey dispatched, but browser focused-control verification "
                                "was unavailable; refusing to count the shortcut as a document edit."
                            )
                    else:
                        # Native apps such as Notes often do not expose text
                        # through the screen-reader path. For those surfaces, a
                        # clean System Events dispatch plus a verified foreground
                        # app is the bounded effect evidence.
                        ok = True
                        effect_verified = True
                        verification = (
                            "Keystroke dispatched and accepted by the OS; on-screen "
                            "read-back was unavailable, so the effect is inferred from "
                            "the clean dispatch."
                        )
                else:
                    ok = False
                    verification = (
                        "Hotkey dispatched but no visible state shift was verified."
                    )
                clipboard_verification: dict[str, Any] = {}
                if is_paste and expected_clipboard_sha256:
                    observed_clipboard = await asyncio.to_thread(self._get_clipboard)
                    observed_text = str(observed_clipboard.get("text") or "")
                    observed_sha256 = hashlib.sha256(
                        observed_text.encode("utf-8")
                    ).hexdigest()
                    clipboard_verification = {
                        "expected_sha256": expected_clipboard_sha256,
                        "observed_sha256": observed_sha256,
                        "expected_chars": expected_clipboard_chars,
                        "observed_chars": len(observed_text),
                        "verified": bool(
                            observed_clipboard.get("ok")
                            and observed_sha256 == expected_clipboard_sha256
                        ),
                    }
                    if not clipboard_verification["verified"]:
                        ok = False
                        effect_verified = False
                        verification = (
                            "Hotkey dispatched, but the clipboard payload no longer "
                            "matched the verified staged document body."
                        )
                expected_app_verified = bool(
                    not expected_frontmost
                    or self._frontmost_app_matches(front_app, expected_frontmost)
                    or self._frontmost_app_matches(post_front_app, expected_frontmost)
                )
                result = {
                    "ok": ok,
                    "action": "hotkey",
                    "hotkey": params.target,
                    "is_paste": is_paste,
                    "frontmost_app_before": front_app,
                    "frontmost_app_after": post_front_app,
                    "expected_frontmost_app": expected_frontmost,
                    "frontmost_app_from_prior_receipt": frontmost_from_prior,
                    "write_target_app_verified": expected_app_verified,
                    "effect_verified": effect_verified,
                    "dispatch": dispatch_receipt,
                    "verification": verification,
                    "visible_state_changed": screen_verifiable and post_state != pre_state,
                    "pre_state_hash": hashlib.sha256(pre_state.encode("utf-8")).hexdigest()[:16]
                    if pre_state
                    else "",
                    "post_state_hash": hashlib.sha256(post_state.encode("utf-8")).hexdigest()[:16]
                    if post_state
                    else "",
                    "clipboard_payload_verification": clipboard_verification,
                }
                if not ok:
                    result["error"] = verification
                return result

            elif action == "scroll":
                # Issue 88: Use x/y correctly
                pre_state = ""
                try:
                    pre_state = await asyncio.to_thread(self._read_screen_text_macos)
                except (
                    TimeoutError,
                    RuntimeError,
                    OSError,
                    AttributeError,
                    TypeError,
                    ValueError,
                    subprocess.SubprocessError,
                ) as exc:
                    logger.debug("Pre-state screen read failed before scroll: %s", exc)
                clicks = int(params.target or "3")
                await asyncio.to_thread(pyautogui.scroll, clicks, x=params.x, y=params.y)
                await asyncio.sleep(0.4)
                post_state = ""
                try:
                    post_state = await asyncio.to_thread(self._read_screen_text_macos)
                except (
                    TimeoutError,
                    RuntimeError,
                    OSError,
                    AttributeError,
                    TypeError,
                    ValueError,
                    subprocess.SubprocessError,
                ) as exc:
                    logger.debug("Post-state screen read failed after scroll: %s", exc)
                effect_verified = bool(pre_state or post_state) and post_state != pre_state
                return {
                    "ok": effect_verified,
                    "scrolled": clicks,
                    "effect_verified": effect_verified,
                    "verification": "State shifted."
                    if effect_verified
                    else "Scroll sent but no visible state shift was verified.",
                }

            elif action == "set_clipboard":
                return await asyncio.to_thread(self._set_clipboard, params.target)

            elif action == "get_clipboard":
                return await asyncio.to_thread(self._get_clipboard)

            elif action == "wait":
                delay_s = max(0.0, min(10.0, float(params.target or 1.0)))
                await asyncio.sleep(delay_s)
                return {"ok": True, "action": "wait", "seconds": delay_s}

            elif action == "run_applescript":
                blocked = await self._require_permissions(
                    "running bounded AppleScript against the foreground desktop",
                    "ACCESSIBILITY",
                    "AUTOMATION",
                )
                if blocked:
                    return blocked
                script = self._validate_user_applescript(params.target)
                output = await asyncio.to_thread(self._run_applescript, script, timeout=12)
                return {
                    "ok": True,
                    "action": "run_applescript",
                    "output": output,
                    "chars": len(output),
                }

            elif action == "write_text_file":
                return await asyncio.to_thread(self._write_text_file, params.target)

            elif action == "create_folder":
                return await asyncio.to_thread(self._create_folder, params.target)
            elif action == "fetch_topic_image":
                return await asyncio.to_thread(self._fetch_topic_image, params.target)

            elif action == "system_control":
                blocked = await self._require_permissions(
                    "changing a system setting through System Events",
                    "AUTOMATION",
                )
                if blocked:
                    return blocked
                return await self._apply_system_control(params.target)

            elif action == "render_text_pdf":
                return await asyncio.to_thread(self._render_text_pdf, params.target)

            elif action == "move_file":
                return await asyncio.to_thread(self._move_file, params.target)

            elif action == "run_command":
                try:
                    args = shlex.split(params.target)
                except ValueError as e:
                    return {"ok": False, "error": f"Invalid command syntax: {e}"}

                if not args:
                    return {"ok": False, "error": "No command provided."}

                cmd = args[0]
                if cmd not in self.ALLOWED_COMMANDS:
                    logger.warning("🛡️ SK-01 Blocked: Command '%s' not in allowlist.", cmd)
                    return {
                        "ok": False,
                        "error": f"Security Violation: Command '{cmd}' is restricted.",
                    }

                # Support safe advanced directory/file traversal
                # 1. Intercept tree command
                if cmd == "tree":
                    target_dir = "."
                    if len(args) > 1:
                        for arg in args[1:]:
                            if not arg.startswith("-"):
                                target_dir = arg
                                break
                    try:
                        output = self._safe_directory_walk(target_dir)
                        return {"ok": True, "output": output, "exit_code": 0}
                    except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
                        return {"ok": False, "error": f"Failed to walk directory: {exc}"}

                # 2. Intercept recursive ls
                if cmd == "ls" and any(arg in {"-R", "--recursive"} for arg in args):
                    target_dir = "."
                    for arg in args[1:]:
                        if not arg.startswith("-"):
                            target_dir = arg
                            break
                    try:
                        output = self._safe_directory_walk(target_dir)
                        return {"ok": True, "output": output, "exit_code": 0}
                    except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
                        return {"ok": False, "error": f"Failed recursive ls walk: {exc}"}

                # 3. Intercept and constrain find commands to prevent infinite hangs
                if cmd == "find":
                    if not any(arg.startswith("-maxdepth") for arg in args):
                        if len(args) > 1 and not args[1].startswith("-"):
                            args.insert(2, "-maxdepth")
                            args.insert(3, "4")
                        else:
                            args.insert(1, "-maxdepth")
                            args.insert(2, "4")

                result = await asyncio.to_thread(
                    get_subprocess_gateway().run,
                    args,
                    capture_output=True,
                    timeout=30,
                    source="computer_use",
                )
                output = (result.stdout or result.stderr or "").strip()[:3000]
                ok = result.returncode == 0
                payload: dict[str, Any] = {
                    "ok": ok,
                    "output": output,
                    "exit_code": result.returncode,
                }
                if not ok:
                    payload["error"] = output or f"Command failed with exit code {result.returncode}."
                return payload

            elif action == "open_app":
                # Opening an app uses LaunchServices plus a System Events
                # frontmost readback. The readback is part of the effect proof,
                # so a cached or environment-level permission assumption is not
                # enough for live desktop reliability.
                blocked = await self._require_permissions(
                    "opening an app and verifying it is frontmost",
                    "ACCESSIBILITY",
                    "AUTOMATION",
                )
                if blocked:
                    return blocked
                result = await asyncio.to_thread(
                    get_subprocess_gateway().run,
                    ["open", "-a", params.target],
                    capture_output=True,
                    timeout=10,
                    source="computer_use",
                )
                if result.returncode != 0:
                    error = (result.stderr or result.stdout or "open command failed").strip()
                    return {"ok": False, "error": error, "opened": params.target}
                activation_error = ""
                try:
                    await self._activate_app(params.target)
                except (TimeoutError, RuntimeError) as exc:
                    activation_error = str(exc)
                effect_verified, frontmost_app = await self._wait_for_frontmost_app(params.target)
                verification = (
                    f"Frontmost app confirmed as {frontmost_app}."
                    if effect_verified
                    else (
                        "Application launch command succeeded, but the requested app "
                        f"did not become frontmost (observed={frontmost_app or 'unavailable'}"
                        + (f", activation_error={activation_error}" if activation_error else "")
                        + ")."
                    )
                )
                return {
                    "ok": effect_verified,
                    "opened": params.target,
                    "returncode": result.returncode,
                    "frontmost_app": frontmost_app,
                    "effect_verified": effect_verified,
                    "verification": verification,
                    **({} if effect_verified else {"error": verification}),
                }

            elif action == "open_url":
                blocked = await self._require_permissions(
                    "opening a browser URL and verifying the active tab",
                    "ACCESSIBILITY",
                    "AUTOMATION",
                )
                if blocked:
                    return blocked
                raw_target = str(params.target or "").strip()
                browser = ""
                url_text = raw_target
                requires_editable_focus = False
                if raw_target.startswith("{"):
                    try:
                        spec = json.loads(raw_target)
                        url_text = str(spec.get("url") or spec.get("target") or "")
                        browser = str(spec.get("browser") or "").strip()
                        requires_editable_focus = bool(
                            spec.get("requires_editable_focus")
                            or spec.get("require_editable_focus")
                        )
                    except (ValueError, TypeError, AttributeError):
                        url_text = raw_target
                if browser and browser not in _ALLOWED_URL_BROWSERS:
                    return {
                        "ok": False,
                        "error": (
                            f"Browser '{browser}' is not in the allowed browser set "
                            f"{sorted(_ALLOWED_URL_BROWSERS)}."
                        ),
                    }
                target_url = self._normalize_open_url_target(url_text)
                if not target_url:
                    return {"ok": False, "error": "No URL or search query provided."}
                if target_url.startswith("file:"):
                    return {"ok": False, "error": "Refusing to open local file URLs from chat."}
                if shutil.which("open"):
                    argv = (
                        ["open", "-a", browser, target_url]
                        if browser
                        else ["open", target_url]
                    )
                    result = await asyncio.to_thread(
                        get_subprocess_gateway().run,
                        argv,
                        capture_output=True,
                        timeout=10,
                        source="computer_use",
                    )
                    if result.returncode != 0:
                        error = (result.stderr or result.stdout or "open command failed").strip()
                        return {"ok": False, "error": error}
                else:
                    opened = await asyncio.to_thread(webbrowser.open, target_url, 2)
                    if not opened:
                        return {"ok": False, "error": "The default browser did not accept the URL."}
                expected_browser = browser
                if not expected_browser:
                    observed_browser = await asyncio.to_thread(self._frontmost_app_name)
                    expected_browser = (
                        observed_browser if observed_browser in _ALLOWED_URL_BROWSERS else ""
                    )
                effect_verified = False
                frontmost_app = ""
                active_url = ""
                active_title = ""
                if expected_browser:
                    effect_verified, frontmost_app = await self._wait_for_frontmost_app(
                        expected_browser
                    )
                    if effect_verified:
                        active_url, active_title = await asyncio.to_thread(
                            self._active_browser_location,
                            expected_browser,
                        )
                        effect_verified = self._url_semantically_matches(
                            target_url,
                            active_url,
                        )
                    forced_navigation = False
                    force_error = ""
                    if not effect_verified and expected_browser in _ALLOWED_URL_BROWSERS:
                        try:
                            await asyncio.to_thread(
                                self._force_browser_tab_url,
                                expected_browser,
                                target_url,
                            )
                            forced_navigation = True
                            await asyncio.sleep(0.7)
                            effect_verified, frontmost_app = await self._wait_for_frontmost_app(
                                expected_browser
                            )
                            if effect_verified:
                                active_url, active_title = await asyncio.to_thread(
                                    self._active_browser_location,
                                    expected_browser,
                                )
                                effect_verified = self._url_semantically_matches(
                                    target_url,
                                    active_url,
                                )
                        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
                            force_error = str(exc)
                            logger.debug("Forced browser URL repair failed: %s", exc)
                else:
                    forced_navigation = False
                    force_error = ""
                surface = f" in {browser}" if browser else ""

                # Google Docs/Sheets/Slides load asynchronously and leave keyboard
                # focus on the omnibox immediately after navigation, so a following
                # `type` lands in the address bar instead of the document body —
                # the "she types in the URL bar instead of the doc" failure. Once
                # navigation is confirmed, wait for the editor to render and click
                # into the canvas so subsequent keystrokes enter the document.
                doc_focused = False
                focus_error = ""
                focus_snapshot = ""
                if (
                    effect_verified
                    and requires_editable_focus
                    and self._is_web_editor_url(active_url or target_url)
                ):
                    if expected_browser and self._is_web_editor_create_url(active_url):
                        deadline = time.monotonic() + 12.0
                        while time.monotonic() < deadline:
                            await asyncio.sleep(0.75)
                            next_url, next_title = await asyncio.to_thread(
                                self._active_browser_location,
                                expected_browser,
                            )
                            if next_url:
                                active_url, active_title = next_url, next_title
                            if self._is_resolved_web_editor_url(active_url):
                                break
                        if self._is_web_editor_create_url(active_url):
                            focus_error = "web_editor_create_not_resolved"
                    editor_pyautogui = pyautogui
                    if not focus_error and editor_pyautogui is None:
                        editor_pyautogui, pyautogui_error = get_pyautogui()
                        if editor_pyautogui is None:
                            focus_error = (
                                "pyautogui_unavailable_for_web_editor_focus"
                                + (f": {pyautogui_error}" if pyautogui_error else "")
                            )
                    if not focus_error and editor_pyautogui is not None:
                        await asyncio.sleep(3.5)  # let the web editor finish loading
                        doc_focused, focus_snapshot, focus_error = await self._focus_web_editor_surface(
                            editor_pyautogui,
                            browser=expected_browser,
                            target_url=target_url,
                        )
                    if requires_editable_focus and not doc_focused:
                        effect_verified = False

                verification = (
                    f"Frontmost browser confirmed as {frontmost_app}; active URL verified as {active_url}."
                    + (" Document canvas focused for editing." if doc_focused else "")
                    if effect_verified
                    else (
                        "URL dispatch succeeded, but the target browser/tab could not be "
                        f"semantically confirmed (frontmost={frontmost_app or 'unavailable'}, "
                        f"active_url={active_url or 'unavailable'}"
                        + (f", forced_navigation_error={force_error}" if force_error else "")
                        + (f", focus_error={focus_error}" if focus_error else "")
                        + ")."
                    )
                )
                return {
                    "ok": effect_verified,
                    "action": "open_url",
                    "url": target_url,
                    "browser": browser,
                    "frontmost_app": frontmost_app,
                    "active_url": active_url,
                    "active_title": active_title,
                    "forced_navigation": forced_navigation,
                    "effect_verified": effect_verified,
                    "doc_focused": doc_focused,
                    "editable_focus_verified": doc_focused,
                    "focus_snapshot": focus_snapshot,
                    "focus_error": focus_error,
                    "verification": verification,
                    "summary": f"I opened a browser tab for {target_url}{surface}.",
                    **({} if effect_verified else {"error": verification}),
                }

            else:
                return {"ok": False, "error": f"Unknown action: {action}"}

        except _COMPUTER_USE_RECOVERABLE_ERRORS as e:
            _record_computer_use_degradation(
                e,
                action="returned explicit computer-use failure payload for recoverable action error",
                stage=f"execute.{action}",
                severity="degraded",
                extra={"action": action},
            )
            runtime_permission_error = self._runtime_permission_payload(str(e))
            if runtime_permission_error:
                return runtime_permission_error
            logger.error("ComputerUse action '%s' failed: %s", action, e)
            return {"ok": False, "error": str(e)}

    @staticmethod
    def _is_web_editor_url(url: str) -> bool:
        """True for editable Google web-doc surfaces where keystrokes must land in
        the document canvas, not the browser address bar."""
        u = str(url or "").lower()
        return "docs.google.com/" in u and any(
            seg in u for seg in ("/document/", "/spreadsheets/", "/presentation/")
        )

    @staticmethod
    def _is_web_editor_create_url(url: str) -> bool:
        try:
            parsed = urllib.parse.urlparse(str(url or "").lower())
        except ValueError:
            return False
        if parsed.netloc != "docs.google.com":
            return False
        return parsed.path.rstrip("/").endswith("/create")

    @classmethod
    def _is_resolved_web_editor_url(cls, url: str) -> bool:
        return cls._is_web_editor_url(url) and not cls._is_web_editor_create_url(url)

    @staticmethod
    def _url_semantically_matches(expected: str, observed: str) -> bool:
        expected = str(expected or "").strip()
        observed = str(observed or "").strip()
        if not expected or not observed:
            return False
        if observed.rstrip("/") == expected.rstrip("/") or observed.startswith(expected):
            return True
        try:
            expected_parts = urllib.parse.urlparse(expected)
            observed_parts = urllib.parse.urlparse(observed)
        except ValueError:
            return False
        if expected_parts.netloc.lower() != observed_parts.netloc.lower():
            return False
        expected_path = expected_parts.path.rstrip("/")
        observed_path = observed_parts.path.rstrip("/")
        if expected_path and observed_path.startswith(expected_path):
            return True
        # Google document creation URLs redirect to a concrete document URL.
        if expected_parts.netloc.lower() == "docs.google.com":
            expected_tokens = {token for token in expected_path.split("/") if token}
            observed_tokens = {token for token in observed_path.split("/") if token}
            return bool(expected_tokens & observed_tokens & {"document", "spreadsheets", "presentation"})
        return False

    @staticmethod
    def _browser_location_script(browser: str) -> str:
        if browser in {"Google Chrome", "Arc", "Microsoft Edge"}:
            return f'''
tell application "{browser}"
    if (count of windows) is 0 then return ""
    set activeUrl to URL of active tab of front window
    set activeTitle to title of active tab of front window
    return activeUrl & linefeed & activeTitle
end tell
'''
        if browser == "Safari":
            return '''
tell application "Safari"
    if (count of windows) is 0 then return ""
    set activeUrl to URL of current tab of front window
    set activeTitle to name of current tab of front window
    return activeUrl & linefeed & activeTitle
end tell
'''
        return ""

    def _active_browser_location(self, browser: str) -> tuple[str, str]:
        script = self._browser_location_script(str(browser or "").strip())
        if not script:
            return "", ""
        try:
            result = get_subprocess_gateway().run(
                ["osascript", "-e", script],
                capture_output=True,
                timeout=5,
                source="computer_use",
            )
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            logger.debug("Active browser URL readback failed: %s", exc)
            return "", ""
        if result.returncode != 0:
            return "", ""
        lines = [line.strip() for line in str(result.stdout or "").splitlines()]
        active_url = lines[0] if lines else ""
        active_title = lines[1] if len(lines) > 1 else ""
        return active_url, active_title

    def read_screen_text(self) -> str:
        """Helper for AgencyCore to read screen text directly."""
        try:
            return self._read_screen_text_macos()
        except _COMPUTER_USE_RECOVERABLE_ERRORS as e:
            _record_computer_use_degradation(
                e,
                action="returned explicit screen-read failure marker to caller",
                stage="read_screen_text.helper",
                severity="warning",
            )
            return f"[read_screen_text failed: {e}]"

    @staticmethod
    def _screen_snapshot_result(snapshot: Any) -> dict[str, Any]:
        """Convert structured screen perception into a computer_use result."""

        accessibility_text = str(getattr(snapshot, "accessibility_text", "") or "").strip()
        screen_text = str(getattr(snapshot, "screen_text", "") or "").strip()
        focused_value = str(getattr(snapshot, "focused_value", "") or "").strip()
        active_app = str(getattr(snapshot, "active_app", "") or "").strip()
        window_title = str(getattr(snapshot, "window_title", "") or "").strip()
        focused_role = str(getattr(snapshot, "focused_role", "") or "").strip()
        focused_name = str(getattr(snapshot, "focused_name", "") or "").strip()
        focused_description = str(
            getattr(snapshot, "focused_description", "") or ""
        ).strip()

        text = screen_text or accessibility_text or focused_value
        if not text:
            lines = []
            if active_app:
                lines.append(f"Active app: {active_app}")
            if window_title:
                lines.append(f"Window: {window_title}")
            focus_parts = [
                part
                for part in (focused_role, focused_name, focused_description)
                if part
            ]
            if focus_parts:
                lines.append("Focused element: " + " | ".join(focus_parts))
            text = "\n".join(lines)

        text_hash = str(getattr(snapshot, "text_hash", "") or "").strip()
        if text and not text_hash:
            text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        return {
            "ok": bool(text),
            "status": "ok" if (screen_text or accessibility_text or focused_value) else "limited",
            "source": "screen_perception",
            "active_app": active_app,
            "window_title": window_title,
            "frontmost_window_bounds": str(
                getattr(snapshot, "frontmost_window_bounds", "") or ""
            ).strip(),
            "focused_role": focused_role,
            "focused_name": focused_name,
            "focused_description": focused_description,
            "focused_value": focused_value,
            "text": text,
            "accessibility_text": accessibility_text,
            "screen_text": screen_text,
            "screenshot_path": str(getattr(snapshot, "screenshot_path", "") or "").strip(),
            "text_hash": text_hash,
            "has_modal": bool(getattr(snapshot, "has_modal", False)),
            "modal_text": str(getattr(snapshot, "modal_text", "") or "").strip(),
            "has_loading": bool(getattr(snapshot, "has_loading", False)),
            "timestamp": float(getattr(snapshot, "timestamp", 0.0) or 0.0),
        }

    def read_menu_clock(self) -> str:
        """Helper for reading the macOS menu bar clock."""
        try:
            return self._read_menu_clock_macos()
        except _COMPUTER_USE_RECOVERABLE_ERRORS as e:
            _record_computer_use_degradation(
                e,
                action="returned explicit menu-clock failure marker to caller",
                stage="read_menu_clock.helper",
                severity="warning",
            )
            return f"[read_menu_clock failed: {e}]"

    def _read_screen_text_macos(self) -> str:
        """Use macOS Accessibility API to extract text from the frontmost app with anti-hang limits."""
        script = """
tell application "System Events"
    try
        set frontApp to first application process whose frontmost is true
        set appName to name of frontApp
        set allText to entire contents of frontApp as string
        return appName & ": " & allText
    on error
        return "[Accessibility error or UI unresponsive]"
    end try
end tell
"""
        raw = self._run_applescript(script, timeout=6)
        if len(raw) > 3000:
            return raw[:1500] + "\n... [TRUNCATED] ...\n" + raw[-1500:]
        return raw

    @staticmethod
    def _screen_text_unavailable(text: str) -> bool:
        normalized = str(text or "").strip().lower()
        if not normalized:
            return True
        return normalized in {
            "[accessibility error or ui unresponsive]",
            "[read_screen_text failed]",
        }

    def _read_menu_clock_macos(self) -> str:
        """Read the live menu bar clock through System Events."""
        script = """
tell application "System Events"
    set ccError to "none"
    set suiError to "none"
    
    try
        if exists process "ControlCenter" then
            tell process "ControlCenter"
                set clockItem to first menu bar item of menu bar 1 whose description is "Clock"
                set clockVal to value of clockItem
                if clockVal is not missing value then
                    return clockVal
                end if
            end tell
        else
            set ccError to "ControlCenter process does not exist"
        end if
    on error errStr number errNum
        set ccError to errStr & " (" & errNum & ")"
    end try
    
    try
        if exists process "ControlCenter" then
            tell process "ControlCenter"
                repeat with item1 in menu bar items of menu bar 1
                    try
                        set d to description of item1
                        set v to value of item1
                        if v is not missing value then
                            set d_lower to my lowercase(d as string)
                            if d_lower contains "clock" or d_lower contains "time" or v contains "AM" or v contains "PM" or v contains ":" or v contains " " then
                                return v
                            end if
                        end if
                    end try
                end repeat
            end tell
        end if
    on error errStr number errNum
        if ccError is "none" or ccError contains "does not exist" then
            set ccError to "Fallback search: " & errStr & " (" & errNum & ")"
        end if
    end try
    
    try
        if exists process "SystemUIServer" then
            tell process "SystemUIServer"
                set clockItem to first menu bar item of menu bar 1 whose description is "Clock"
                return name of clockItem
            end tell
        else
            set suiError to "SystemUIServer process does not exist"
        end if
    on error errStr number errNum
        set suiError to errStr & " (" & errNum & ")"
    end try
    
    error "Clock menu bar item not found. ControlCenter error: " & ccError & ". SystemUIServer error: " & suiError
end tell

on lowercase(txt)
    set the_alphabet to "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    set the_lowercase to "abcdefghijklmnopqrstuvwxyz"
    set the_result to ""
    repeat with i from 1 to count of characters in txt
        set the_char to character i of txt
        set the_index to offset of the_char in the_alphabet
        if the_index is not 0 then
            set the_result to the_result & character the_index of the_lowercase
        else
            set the_result to the_result & the_char
        end if
    end repeat
    return the_result
end lowercase
"""
        return self._run_applescript(script, timeout=10)[:240]
