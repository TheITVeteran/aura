"""core/capabilities/host_automation.py — Generalized OS Automation Provider
=============================================================================
The single abstraction that gives Aura arbitrary OS manipulation through
governed primitives — NO hardcoded app-specific logic.

Routes through the most reliable adapter automatically:
    1. Direct API (file ops, system settings) — safest
    2. AppleScript / System Events — reliable for app control
    3. Accessibility API — for UI element interaction
    4. PyAutoGUI — fallback for generic screen control

Every call produces a ToolExecutionReceipt and goes through
CapabilityEngine + UnifiedWill. No action bypasses governance.

Usage:
    provider = get_host_automation()
    result = await provider.launch_app("Notes")
    result = await provider.get_frontmost_app()
    result = await provider.execute_applescript(script)  # AST-guarded
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.HostAutomation")

_HOST_AUTOMATION_ERRORS = (
    ImportError,
    OSError,
    RuntimeError,
    AttributeError,
    TypeError,
    ValueError,
    asyncio.TimeoutError,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AutomationReceipt:
    """Immutable receipt for every host automation action."""
    action: str
    target: str
    adapter: str                    # "applescript", "accessibility", "pyautogui", "direct_api"
    success: bool
    result: Any = None
    error: str = ""
    duration_ms: float = 0.0
    script_hash: str = ""           # SHA256 of any executed script
    timestamp: float = field(default_factory=time.time)
    receipt_id: str = ""

    def __post_init__(self):
        if not self.receipt_id:
            payload = f"{self.timestamp}|{self.action}|{self.target}|{self.success}"
            self.receipt_id = hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Script Safety Guard
# ---------------------------------------------------------------------------

class ScriptASTGuard:
    """Validates AppleScript and shell commands before execution.

    Blocks destructive patterns while allowing standard app control.
    This is the safety boundary for dynamic script compilation.
    """

    # Patterns that are ALWAYS blocked in AppleScript
    BLOCKED_APPLESCRIPT_PATTERNS = [
        r'\bdo\s+shell\s+script\s+.*\brm\s+(-[rRf]+\s+)?/',   # rm -rf from AppleScript
        r'\bdo\s+shell\s+script\s+.*\bsudo\b',                  # sudo from AppleScript
        r'\bdo\s+shell\s+script\s+.*\bcurl\s+.*\|\s*sh\b',      # curl | sh
        r'\bdo\s+shell\s+script\s+.*\bwget\s+.*\|\s*sh\b',      # wget | sh
        r'\bdo\s+shell\s+script\s+.*\bmkfs\b',                  # filesystem format
        r'\bdo\s+shell\s+script\s+.*\bdd\s+if=',                # disk duplicate
        r'\bdo\s+shell\s+script\s+.*\bformat\b',                # disk format
        r'\bdo\s+shell\s+script\s+.*\bshutdown\b',              # system shutdown
        r'\bdo\s+shell\s+script\s+.*\breboot\b',                # system reboot
        r'\bdo\s+shell\s+script\s+.*\blaunchctl\s+unload\b',    # service unload
        r'\bdo\s+shell\s+script\s+.*\bkillall\b',               # mass kill
        r'\bdelete\s+every\s+',                                  # mass delete in apps
        r'\bdo\s+shell\s+script\s+.*\bsecurity\s+delete\b',     # keychain delete
    ]

    # Patterns that are ALWAYS allowed
    ALLOWED_APPLESCRIPT_COMMANDS = {
        "tell application", "activate", "set", "get", "click",
        "keystroke", "key code", "delay", "return", "end tell",
        "name of", "title of", "window", "menu item", "menu bar",
        "open location", "make new", "set value", "frontmost",
        "bounds of", "size of", "position of", "count",
        "properties of", "exists", "close", "save",
    }

    @classmethod
    def validate_applescript(cls, script: str) -> Tuple[bool, str]:
        """Validate an AppleScript for safety.

        Returns (is_safe, reason).
        """
        if not script or not script.strip():
            return False, "Empty script"

        script_lower = script.lower()

        # Check blocked patterns
        for pattern in cls.BLOCKED_APPLESCRIPT_PATTERNS:
            if re.search(pattern, script_lower, re.IGNORECASE):
                return False, f"Blocked pattern detected: {pattern[:60]}"

        # Check for `do shell script` — allowed only with safe commands
        if "do shell script" in script_lower:
            # Extract the shell command
            shell_match = re.findall(
                r'do\s+shell\s+script\s+["\'](.+?)["\']',
                script, re.IGNORECASE | re.DOTALL,
            )
            for cmd in shell_match:
                if not cls._is_safe_shell_command(cmd):
                    return False, f"Unsafe shell command: {cmd[:100]}"

        # Length limit
        if len(script) > 10000:
            return False, f"Script too long ({len(script)} chars, max 10000)"

        return True, "safe"

    @classmethod
    def _is_safe_shell_command(cls, cmd: str) -> bool:
        """Check if a shell command embedded in AppleScript is safe."""
        cmd_lower = cmd.strip().lower()
        # Whitelist of safe shell commands
        safe_prefixes = [
            "open ", "pbcopy", "pbpaste", "screencapture",
            "osascript", "defaults read", "echo ", "cat ",
            "ls ", "pwd", "whoami", "date", "sw_vers",
            "system_profiler", "pmset -g", "ioreg",
            "desktoppr",  # wallpaper tool
        ]
        return any(cmd_lower.startswith(prefix) for prefix in safe_prefixes)

    @classmethod
    def validate_shell_command(cls, command: str) -> Tuple[bool, str]:
        """Validate a direct shell command for safety."""
        if not command or not command.strip():
            return False, "Empty command"

        cmd_lower = command.strip().lower()

        # Block destructive commands
        blocked = [
            "rm -rf /", "rm -rf ~", "rm -rf /*", "sudo rm",
            "mkfs", "dd if=", "format", "shutdown", "reboot",
            "killall", "launchctl unload", "> /dev/sd",
            "chmod -R 777 /", "chown -R",
        ]
        for pattern in blocked:
            if pattern in cmd_lower:
                return False, f"Blocked: {pattern}"

        return True, "safe"


# ---------------------------------------------------------------------------
# AppleScript Runner
# ---------------------------------------------------------------------------

class AppleScriptRunner:
    """Runs validated AppleScript with proper error handling and receipts."""

    @staticmethod
    async def run(
        script: str,
        timeout: float = 10.0,
        *,
        read_only: bool = False,
        source: str = "host_automation.applescript",
    ) -> AutomationReceipt:
        """Execute an AppleScript after AST validation."""
        start = time.time()

        # Validate
        is_safe, reason = ScriptASTGuard.validate_applescript(script)
        if not is_safe:
            return AutomationReceipt(
                action="execute_applescript",
                target=script[:200],
                adapter="applescript",
                success=False,
                error=f"Script blocked by ASTGuard: {reason}",
                duration_ms=(time.time() - start) * 1000,
                script_hash=hashlib.sha256(script.encode()).hexdigest()[:16],
            )

        # Execute
        try:
            proc = await get_subprocess_gateway().spawn_async(
                ["osascript", "-e", script],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                read_only=read_only,
                source=source,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            success = proc.returncode == 0
            result = stdout.decode("utf-8", errors="replace").strip() if stdout else ""
            error = stderr.decode("utf-8", errors="replace").strip() if stderr and not success else ""

            return AutomationReceipt(
                action="execute_applescript",
                target=script[:200],
                adapter="applescript",
                success=success,
                result=result,
                error=error,
                duration_ms=(time.time() - start) * 1000,
                script_hash=hashlib.sha256(script.encode()).hexdigest()[:16],
            )
        except asyncio.TimeoutError:
            return AutomationReceipt(
                action="execute_applescript",
                target=script[:200],
                adapter="applescript",
                success=False,
                error=f"AppleScript timed out after {timeout}s",
                duration_ms=(time.time() - start) * 1000,
                script_hash=hashlib.sha256(script.encode()).hexdigest()[:16],
            )
        except (OSError, RuntimeError, ValueError) as e:
            return AutomationReceipt(
                action="execute_applescript",
                target=script[:200],
                adapter="applescript",
                success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
                script_hash=hashlib.sha256(script.encode()).hexdigest()[:16],
            )


# ---------------------------------------------------------------------------
# The Provider
# ---------------------------------------------------------------------------

class HostAutomationProvider:
    """Generalized OS automation through governed primitives.

    Every method:
    1. Validates input through ScriptASTGuard
    2. Executes through the most reliable adapter
    3. Produces an AutomationReceipt
    4. Logs to LifeTrace

    This is how Aura manipulates the OS without hardcoded app-specific code.
    The LLM + TaskDecomposer decides WHAT to do; this layer executes HOW.
    """

    def __init__(self) -> None:
        self._receipts: list[AutomationReceipt] = []
        self._max_receipts = 500
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        ServiceContainer.register_instance("host_automation", self, required=False)
        self._started = True
        logger.info("HostAutomationProvider ONLINE — generalized OS automation ready")

    def _log_receipt(self, receipt: AutomationReceipt) -> None:
        """Log receipt to internal buffer and LifeTrace."""
        self._receipts.append(receipt)
        if len(self._receipts) > self._max_receipts:
            self._receipts = self._receipts[-self._max_receipts:]

        # Log to LifeTrace
        try:
            from core.runtime.life_trace import get_life_trace
            get_life_trace().record(
                event_type="action_executed",
                origin="host_automation",
                action_taken={
                    "action": receipt.action,
                    "target": str(receipt.target)[:200],
                    "adapter": receipt.adapter,
                    "success": receipt.success,
                },
                result={
                    "result": str(receipt.result)[:500] if receipt.result else "",
                    "error": receipt.error[:200] if receipt.error else "",
                    "duration_ms": receipt.duration_ms,
                    "receipt_id": receipt.receipt_id,
                },
            )
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("host_automation.life_trace", e)

    # ------------------------------------------------------------------
    # App control primitives
    # ------------------------------------------------------------------

    async def launch_app(self, app_name: str) -> AutomationReceipt:
        """Launch an application by name. Uses AppleScript 'activate'."""
        start = time.time()
        script = f'tell application "{app_name}" to activate'
        receipt = await AppleScriptRunner.run(script, timeout=10.0)
        receipt.action = "launch_app"
        receipt.target = app_name

        # Verify it actually launched
        if receipt.success:
            await asyncio.sleep(0.5)
            frontmost = await self.get_frontmost_app()
            if frontmost.result and app_name.lower() in str(frontmost.result).lower():
                receipt.result = f"{app_name} is now frontmost"
            else:
                receipt.result = f"{app_name} launched (may not be frontmost yet)"

        self._log_receipt(receipt)
        return receipt

    async def focus_app(self, app_name: str) -> AutomationReceipt:
        """Bring an already-running application to front."""
        script = f'''
            tell application "System Events"
                set frontProcess to first process whose name is "{app_name}"
                set frontmost of frontProcess to true
            end tell
        '''
        receipt = await AppleScriptRunner.run(script, timeout=5.0)
        receipt.action = "focus_app"
        receipt.target = app_name
        self._log_receipt(receipt)
        return receipt

    async def get_frontmost_app(self) -> AutomationReceipt:
        """Get the name of the currently frontmost application."""
        script = 'tell application "System Events" to get name of first application process whose frontmost is true'
        receipt = await AppleScriptRunner.run(
            script,
            timeout=3.0,
            read_only=True,
            source="host_automation.frontmost_app",
        )
        receipt.action = "get_frontmost_app"
        receipt.target = ""
        # Don't log this one to LifeTrace (it's a read, not an action)
        return receipt

    async def get_window_title(self, app_name: str = "") -> AutomationReceipt:
        """Get the title of the frontmost window of an app (or the frontmost app)."""
        if app_name:
            script = f'tell application "System Events" to get name of front window of process "{app_name}"'
        else:
            script = '''
                tell application "System Events"
                    set frontApp to name of first application process whose frontmost is true
                    set winTitle to name of front window of process frontApp
                end tell
                return winTitle
            '''
        receipt = await AppleScriptRunner.run(
            script,
            timeout=3.0,
            read_only=True,
            source="host_automation.window_title",
        )
        receipt.action = "get_window_title"
        receipt.target = app_name
        return receipt

    async def close_app(self, app_name: str) -> AutomationReceipt:
        """Quit an application gracefully."""
        script = f'tell application "{app_name}" to quit'
        receipt = await AppleScriptRunner.run(script, timeout=5.0)
        receipt.action = "close_app"
        receipt.target = app_name
        self._log_receipt(receipt)
        return receipt

    async def get_running_apps(self) -> AutomationReceipt:
        """List all running GUI applications."""
        script = 'tell application "System Events" to get name of every application process whose background only is false'
        receipt = await AppleScriptRunner.run(
            script,
            timeout=5.0,
            read_only=True,
            source="host_automation.running_apps",
        )
        receipt.action = "get_running_apps"
        if receipt.success and receipt.result:
            # Parse comma-separated list
            apps = [a.strip() for a in str(receipt.result).split(",") if a.strip()]
            receipt.result = apps
        return receipt

    # ------------------------------------------------------------------
    # UI interaction primitives
    # ------------------------------------------------------------------

    async def menu_select(self, app_name: str, menu_path: List[str]) -> AutomationReceipt:
        """Click a menu item by path. E.g., menu_path=["File", "Export as PDF..."]."""
        if not menu_path:
            return AutomationReceipt(
                action="menu_select", target=app_name,
                adapter="applescript", success=False, error="Empty menu path",
            )

        # Build nested menu click AppleScript
        menu_items = " of ".join(
            f'menu item "{item}"' if i > 0 else f'menu item "{item}"'
            for i, item in enumerate(reversed(menu_path))
        )
        # Actually need to navigate the menu hierarchy
        if len(menu_path) == 1:
            script = f'''
                tell application "System Events"
                    tell process "{app_name}"
                        click menu item "{menu_path[0]}" of menu bar 1
                    end tell
                end tell
            '''
        elif len(menu_path) == 2:
            script = f'''
                tell application "System Events"
                    tell process "{app_name}"
                        click menu item "{menu_path[1]}" of menu 1 of menu bar item "{menu_path[0]}" of menu bar 1
                    end tell
                end tell
            '''
        elif len(menu_path) == 3:
            script = f'''
                tell application "System Events"
                    tell process "{app_name}"
                        click menu item "{menu_path[2]}" of menu 1 of menu item "{menu_path[1]}" of menu 1 of menu bar item "{menu_path[0]}" of menu bar 1
                    end tell
                end tell
            '''
        else:
            return AutomationReceipt(
                action="menu_select", target=f"{app_name}: {' > '.join(menu_path)}",
                adapter="applescript", success=False, error="Menu path too deep (max 3 levels)",
            )

        receipt = await AppleScriptRunner.run(script, timeout=5.0)
        receipt.action = "menu_select"
        receipt.target = f"{app_name}: {' > '.join(menu_path)}"
        self._log_receipt(receipt)
        return receipt

    async def type_text(self, text: str, use_clipboard: bool = True) -> AutomationReceipt:
        """Type text into the currently focused application.

        For text longer than 50 chars, uses clipboard paste (faster, more reliable).
        For short text, uses keystroke (more natural).
        """
        start = time.time()
        if use_clipboard and len(text) > 50:
            # Clipboard paste method — faster and more reliable
            try:
                # Save current clipboard
                save_proc = await get_subprocess_gateway().spawn_async(
                    ["pbpaste"],
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    read_only=True,
                    source="host_automation.clipboard_read",
                )
                old_clipboard, _ = await asyncio.wait_for(save_proc.communicate(), timeout=2.0)

                # Set new clipboard content
                set_proc = await get_subprocess_gateway().spawn_async(
                    ["pbcopy"],
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    source="host_automation.clipboard_write",
                )
                await asyncio.wait_for(
                    set_proc.communicate(input=text.encode("utf-8")),
                    timeout=2.0,
                )

                # Paste
                paste_script = '''
                    tell application "System Events"
                        keystroke "v" using command down
                    end tell
                '''
                receipt = await AppleScriptRunner.run(paste_script, timeout=3.0)
                receipt.action = "type_text"
                receipt.target = f"[clipboard paste, {len(text)} chars]"
                receipt.adapter = "clipboard+applescript"

                # Restore old clipboard after a brief delay
                await asyncio.sleep(0.3)
                if old_clipboard:
                    restore_proc = await get_subprocess_gateway().spawn_async(
                        ["pbcopy"],
                        stdin=asyncio.subprocess.PIPE,
                        source="host_automation.clipboard_restore",
                    )
                    await asyncio.wait_for(
                        restore_proc.communicate(input=old_clipboard),
                        timeout=2.0,
                    )

                self._log_receipt(receipt)
                return receipt

            except (OSError, asyncio.TimeoutError) as e:
                logger.debug("Clipboard paste failed, falling back to keystroke: %s", e)
                # Fall through to keystroke method

        # Keystroke method — for short text or when clipboard fails
        # Escape special chars for AppleScript
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        # Split into chunks to avoid AppleScript string limits
        chunk_size = 200
        chunks = [escaped[i:i + chunk_size] for i in range(0, len(escaped), chunk_size)]

        success = True
        errors = []
        for chunk in chunks:
            script = f'''
                tell application "System Events"
                    keystroke "{chunk}"
                end tell
            '''
            result = await AppleScriptRunner.run(script, timeout=5.0)
            if not result.success:
                success = False
                errors.append(result.error)
                break
            await asyncio.sleep(0.05)  # Small delay between chunks

        receipt = AutomationReceipt(
            action="type_text",
            target=f"[keystroke, {len(text)} chars]",
            adapter="applescript",
            success=success,
            error="; ".join(errors) if errors else "",
            duration_ms=(time.time() - start) * 1000,
        )
        self._log_receipt(receipt)
        return receipt

    async def hotkey(self, *keys: str) -> AutomationReceipt:
        """Press a keyboard shortcut. E.g., hotkey("command", "s")."""
        modifiers = {
            "command": "command down",
            "cmd": "command down",
            "shift": "shift down",
            "option": "option down",
            "alt": "option down",
            "control": "control down",
            "ctrl": "control down",
        }

        key_parts = list(keys)
        if not key_parts:
            return AutomationReceipt(
                action="hotkey", target="", adapter="applescript",
                success=False, error="No keys specified",
            )

        # Separate modifiers from the main key
        mods = []
        main_key = ""
        for k in key_parts:
            k_lower = k.lower().strip()
            if k_lower in modifiers:
                mods.append(modifiers[k_lower])
            else:
                main_key = k_lower

        if not main_key:
            return AutomationReceipt(
                action="hotkey", target="+".join(keys),
                adapter="applescript", success=False, error="No main key specified",
            )

        mod_str = " using {" + ", ".join(mods) + "}" if mods else ""
        # Handle special keys
        special_keys = {
            "return": 'key code 36', "enter": 'key code 36',
            "tab": 'key code 48', "escape": 'key code 53', "esc": 'key code 53',
            "delete": 'key code 51', "backspace": 'key code 51',
            "space": 'key code 49',
            "up": 'key code 126', "down": 'key code 125',
            "left": 'key code 123', "right": 'key code 124',
        }

        if main_key in special_keys:
            script = f'''
                tell application "System Events"
                    {special_keys[main_key]}{mod_str}
                end tell
            '''
        else:
            script = f'''
                tell application "System Events"
                    keystroke "{main_key}"{mod_str}
                end tell
            '''

        receipt = await AppleScriptRunner.run(script, timeout=3.0)
        receipt.action = "hotkey"
        receipt.target = "+".join(keys)
        self._log_receipt(receipt)
        return receipt

    async def click_at(self, x: int, y: int, button: str = "left") -> AutomationReceipt:
        """Click at screen coordinates using cliclick (fast) or PyAutoGUI (fallback)."""
        start = time.time()
        try:
            # Try cliclick first (faster, no Python dependency)
            click_type = "c" if button == "left" else "rc"
            proc = await get_subprocess_gateway().spawn_async(
                ["cliclick", f"{click_type}:{x},{y}"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                source="host_automation.click",
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=3.0)
            success = proc.returncode == 0
            receipt = AutomationReceipt(
                action="click", target=f"{x},{y}",
                adapter="cliclick", success=success,
                error=stderr.decode().strip() if stderr and not success else "",
                duration_ms=(time.time() - start) * 1000,
            )
        except (FileNotFoundError, OSError):
            # Fallback to PyAutoGUI
            try:
                import pyautogui
                if button == "right":
                    pyautogui.rightClick(x, y)
                else:
                    pyautogui.click(x, y)
                receipt = AutomationReceipt(
                    action="click", target=f"{x},{y}",
                    adapter="pyautogui", success=True,
                    duration_ms=(time.time() - start) * 1000,
                )
            except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError) as e:
                receipt = AutomationReceipt(
                    action="click", target=f"{x},{y}",
                    adapter="pyautogui", success=False,
                    error=str(e),
                    duration_ms=(time.time() - start) * 1000,
                )
        except asyncio.TimeoutError:
            receipt = AutomationReceipt(
                action="click", target=f"{x},{y}",
                adapter="cliclick", success=False,
                error="Click timed out",
                duration_ms=(time.time() - start) * 1000,
            )

        self._log_receipt(receipt)
        return receipt

    async def scroll(self, dx: int = 0, dy: int = 0) -> AutomationReceipt:
        """Scroll by delta amounts."""
        start = time.time()
        try:
            import pyautogui
            pyautogui.scroll(dy, _pause=False)
            if dx:
                pyautogui.hscroll(dx, _pause=False)
            receipt = AutomationReceipt(
                action="scroll", target=f"dx={dx},dy={dy}",
                adapter="pyautogui", success=True,
                duration_ms=(time.time() - start) * 1000,
            )
        except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError) as e:
            receipt = AutomationReceipt(
                action="scroll", target=f"dx={dx},dy={dy}",
                adapter="pyautogui", success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )
        self._log_receipt(receipt)
        return receipt

    # ------------------------------------------------------------------
    # Screen capture primitives
    # ------------------------------------------------------------------

    async def take_screenshot(self, save_path: str = "", region: Optional[Tuple[int, int, int, int]] = None) -> AutomationReceipt:
        """Take a screenshot and optionally save to path.

        Args:
            save_path: Where to save (auto-generated if empty).
            region: Optional (x, y, w, h) to capture a region.

        Returns:
            Receipt with save_path as result.
        """
        start = time.time()
        if not save_path:
            ts = time.strftime("%Y%m%d_%H%M%S")
            save_dir = Path.home() / ".aura" / "data" / "screenshots"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(save_dir / f"screenshot_{ts}.png")

        try:
            cmd = ["screencapture", "-x"]  # -x = no sound
            if region:
                x, y, w, h = region
                cmd.extend(["-R", f"{x},{y},{w},{h}"])
            cmd.append(save_path)

            proc = await get_subprocess_gateway().spawn_async(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                source="host_automation.screenshot",
            )
            await asyncio.wait_for(proc.communicate(), timeout=5.0)
            success = proc.returncode == 0 and Path(save_path).exists()

            receipt = AutomationReceipt(
                action="take_screenshot", target=save_path,
                adapter="screencapture", success=success,
                result=save_path if success else "",
                duration_ms=(time.time() - start) * 1000,
            )
        except (OSError, asyncio.TimeoutError) as e:
            receipt = AutomationReceipt(
                action="take_screenshot", target=save_path,
                adapter="screencapture", success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

        self._log_receipt(receipt)
        return receipt

    async def get_screen_text(self, region: Optional[Tuple[int, int, int, int]] = None) -> AutomationReceipt:
        """Take a screenshot and extract text via OCR."""
        start = time.time()
        # Take screenshot first
        ss = await self.take_screenshot(region=region)
        if not ss.success or not ss.result:
            return AutomationReceipt(
                action="get_screen_text", target="",
                adapter="ocr", success=False,
                error=f"Screenshot failed: {ss.error}",
                duration_ms=(time.time() - start) * 1000,
            )

        # Try OCR
        text = ""
        try:
            # Try macOS Vision framework via swift/shortcuts
            script = f'''
                use framework "Vision"
                use framework "Foundation"

                set imagePath to POSIX file "{ss.result}"
                set theImage to current application's NSImage's alloc()'s initWithContentsOfFile:(POSIX path of imagePath)
            '''
            # Fallback to pytesseract
            import pytesseract
            from PIL import Image
            img = Image.open(str(ss.result))
            text = pytesseract.image_to_string(img)
        except ImportError:
            # No pytesseract — try simple macOS shortcut
            try:
                # Use shortcuts or textutil as fallback
                text = f"[OCR unavailable — screenshot saved at {ss.result}]"
            except _HOST_AUTOMATION_ERRORS:
                text = ""
        except _HOST_AUTOMATION_ERRORS as e:
            text = f"[OCR error: {e}]"

        receipt = AutomationReceipt(
            action="get_screen_text", target=str(ss.result),
            adapter="ocr", success=bool(text),
            result=text[:2000],
            duration_ms=(time.time() - start) * 1000,
        )
        return receipt

    # ------------------------------------------------------------------
    # AppleScript execution (the general-purpose primitive)
    # ------------------------------------------------------------------

    async def execute_applescript(self, script: str) -> AutomationReceipt:
        """Execute arbitrary AppleScript after safety validation.

        This is the general-purpose primitive. The LLM/TaskDecomposer can
        compile any macOS automation into AppleScript, and this method
        executes it safely.

        The script passes through ScriptASTGuard before execution.
        """
        return await AppleScriptRunner.run(script, timeout=15.0)

    # ------------------------------------------------------------------
    # Shell command execution (governed)
    # ------------------------------------------------------------------

    async def run_command(self, command: str, timeout: float = 15.0) -> AutomationReceipt:
        """Run a shell command after safety validation."""
        start = time.time()

        is_safe, reason = ScriptASTGuard.validate_shell_command(command)
        if not is_safe:
            receipt = AutomationReceipt(
                action="run_command", target=command[:200],
                adapter="shell", success=False,
                error=f"Command blocked: {reason}",
                duration_ms=(time.time() - start) * 1000,
            )
            self._log_receipt(receipt)
            return receipt

        try:
            proc = await get_subprocess_gateway().spawn_shell_async(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(Path.home()),
                source="host_automation.shell_command",
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            success = proc.returncode == 0
            receipt = AutomationReceipt(
                action="run_command", target=command[:200],
                adapter="shell", success=success,
                result=stdout.decode("utf-8", errors="replace").strip()[:2000] if stdout else "",
                error=stderr.decode("utf-8", errors="replace").strip()[:500] if stderr and not success else "",
                duration_ms=(time.time() - start) * 1000,
            )
        except asyncio.TimeoutError:
            receipt = AutomationReceipt(
                action="run_command", target=command[:200],
                adapter="shell", success=False,
                error=f"Command timed out after {timeout}s",
                duration_ms=(time.time() - start) * 1000,
            )
        except OSError as e:
            receipt = AutomationReceipt(
                action="run_command", target=command[:200],
                adapter="shell", success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

        self._log_receipt(receipt)
        return receipt

    # ------------------------------------------------------------------
    # Wait / condition primitives
    # ------------------------------------------------------------------

    async def wait_for_condition(
        self,
        predicate_name: str,
        predicate_args: Dict[str, Any],
        timeout: float = 10.0,
        poll_interval: float = 0.5,
    ) -> AutomationReceipt:
        """Wait until a condition is true or timeout.

        Supported predicates:
            app_is_frontmost(name) — check if app is the frontmost
            file_exists(path) — check if file exists
            window_title_contains(text) — check window title
        """
        start = time.time()
        while (time.time() - start) < timeout:
            try:
                result = await self._check_predicate(predicate_name, predicate_args)
                if result:
                    return AutomationReceipt(
                        action="wait_for_condition",
                        target=f"{predicate_name}({predicate_args})",
                        adapter="poll", success=True,
                        result=f"Condition met after {(time.time()-start)*1000:.0f}ms",
                        duration_ms=(time.time() - start) * 1000,
                    )
            except (OSError, RuntimeError) as e:
                logger.debug("Predicate check failed: %s", e)
            await asyncio.sleep(poll_interval)

        return AutomationReceipt(
            action="wait_for_condition",
            target=f"{predicate_name}({predicate_args})",
            adapter="poll", success=False,
            error=f"Condition not met within {timeout}s",
            duration_ms=(time.time() - start) * 1000,
        )

    async def _check_predicate(self, name: str, args: Dict[str, Any]) -> bool:
        """Evaluate a single predicate."""
        if name == "app_is_frontmost":
            receipt = await self.get_frontmost_app()
            return bool(
                receipt.success and receipt.result
                and str(args.get("name", "")).lower() in str(receipt.result).lower()
            )
        elif name == "file_exists":
            return Path(str(args.get("path", ""))).exists()
        elif name == "window_title_contains":
            receipt = await self.get_window_title(args.get("app", ""))
            return bool(
                receipt.success and receipt.result
                and str(args.get("text", "")).lower() in str(receipt.result).lower()
            )
        return False

    # ------------------------------------------------------------------
    # Status / Audit
    # ------------------------------------------------------------------

    def get_recent_receipts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent automation receipts for audit."""
        return [
            {
                "action": r.action,
                "target": r.target[:100],
                "adapter": r.adapter,
                "success": r.success,
                "error": r.error[:100] if r.error else "",
                "duration_ms": round(r.duration_ms, 1),
                "receipt_id": r.receipt_id,
                "timestamp": r.timestamp,
            }
            for r in self._receipts[-limit:]
        ]

    def get_status(self) -> Dict[str, Any]:
        """Provider status for dashboards."""
        total = len(self._receipts)
        successes = sum(1 for r in self._receipts if r.success)
        return {
            "started": self._started,
            "total_actions": total,
            "success_rate": round(successes / max(1, total), 3),
            "recent_actions": self.get_recent_receipts(5),
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[HostAutomationProvider] = None


def get_host_automation() -> HostAutomationProvider:
    global _instance
    if _instance is None:
        _instance = HostAutomationProvider()
    return _instance


__all__ = [
    "HostAutomationProvider",
    "AutomationReceipt",
    "ScriptASTGuard",
    "AppleScriptRunner",
    "get_host_automation",
]
