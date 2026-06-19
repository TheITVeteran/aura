"""Governed natural-language OS automation.

This skill gives Aura a general desktop action lane without hardcoding one
demo task. The model may synthesize AppleScript for novel UI workflows, but
the script is bounded, statically guarded, authorized by AuthorityGateway, run
inside a governed receipt scope, and executed only through HostAutomation.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
import urllib.parse
from typing import Any, Dict

from pydantic import BaseModel, Field

from core.capabilities.host_automation import ScriptASTGuard, get_host_automation
from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.OSAutomation")

_OS_AUTOMATION_ENV_ERRORS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

_CODE_BLOCK_RE = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_+-]*)\s*\n(?P<body>.*?)```",
    re.DOTALL,
)


class OSAutomationInput(BaseModel):
    goal: str = Field(..., description="High-level desktop objective to accomplish.")
    script_type: str = Field(
        "applescript",
        description="Generated script type. AppleScript is the default governed desktop lane.",
    )
    execute: bool = Field(True, description="When false, compile and validate without executing.")


class OSAutomationCompilerSkill(BaseSkill):
    """Compile a desktop objective into a governed AppleScript action."""

    name = "os_automation"
    description = (
        "General governed macOS desktop automation. Use for app control, text entry, "
        "menus, browser/doc workflows, and visible UI actions when structured "
        "HostAutomation primitives are insufficient."
    )
    input_model = OSAutomationInput
    timeout_seconds = 45.0
    metabolic_cost = 3
    requires_approval = True

    async def execute(self, params: OSAutomationInput, context: Dict[str, Any]) -> Dict[str, Any]:
        goal = str(params.goal or "").strip()
        if not goal:
            return {"ok": False, "error": "OS automation goal is empty."}

        script_type = str(params.script_type or "applescript").strip().lower()
        if script_type not in {"applescript", "bash"}:
            return {"ok": False, "error": f"Unsupported script_type: {script_type}"}
        if script_type == "bash" and not bool(context.get("allow_shell_os_automation")):
            return {
                "ok": False,
                "error": "Dynamic shell OS automation requires explicit allow_shell_os_automation context.",
            }

        engine = ServiceContainer.get("cognitive_engine", default=None)
        if engine is None:
            return {"ok": False, "error": "Cognitive engine is not available."}

        # Query current macOS environment state for the compiler dynamically
        env_info = []
        host = get_host_automation()
        if host and script_type == "applescript" and context.get("source") != "unit" and context.get("origin") != "unit":
            # 1. Frontmost app
            try:
                frontmost_script = 'tell application "System Events" to name of first application process whose frontmost is true'
                res = await host.execute_applescript(frontmost_script)
                frontmost = str(getattr(res, "result", "") or "").strip()
                if frontmost:
                    env_info.append(f"Frontmost application: {frontmost}")
                    
                    # 2. Browser URL if browser is frontmost
                    if frontmost in {"Google Chrome", "Arc", "Microsoft Edge", "Safari"}:
                        browser_script = ""
                        if frontmost in {"Google Chrome", "Arc", "Microsoft Edge"}:
                            browser_script = f'''
tell application "{frontmost}"
    if (count of windows) is 0 then return ""
    set activeUrl to URL of active tab of front window
    set activeTitle to title of active tab of front window
    return activeUrl & " - " & activeTitle
end tell
'''
                        elif frontmost == "Safari":
                            browser_script = '''
tell application "Safari"
    if (count of windows) is 0 then return ""
    set activeUrl to URL of current tab of front window
    set activeTitle to name of current tab of front window
    return activeUrl & " - " & activeTitle
end tell
'''
                        if browser_script:
                            b_res = await host.execute_applescript(browser_script)
                            b_loc = str(getattr(b_res, "result", "") or "").strip()
                            if b_loc:
                                env_info.append(f"Active browser page/tab: {b_loc}")
            except _OS_AUTOMATION_ENV_ERRORS as exc:
                logger.debug("OSAutomation environment app query failed: %s", exc)

            # 3. Screen text
            try:
                screen_script = '''
tell application "System Events"
    try
        set frontApp to first application process whose frontmost is true
        set appName to name of frontApp
        set allText to entire contents of frontApp as string
        return appName & ": " & allText
    on error
        return ""
    end try
end tell
'''
                s_res = await host.execute_applescript(screen_script)
                screen_text = str(getattr(s_res, "result", "") or "").strip()
                if screen_text:
                    if len(screen_text) > 2000:
                        screen_text = screen_text[:1000] + "\n... [TRUNCATED] ...\n" + screen_text[-1000:]
                    env_info.append(f"Active window screen text:\n{screen_text}")
            except _OS_AUTOMATION_ENV_ERRORS as exc:
                logger.debug("OSAutomation environment screen text query failed: %s", exc)

            # 4. List of running application processes
            try:
                running_script = 'tell application "System Events" to name of every application process whose visible is true'
                r_res = await host.execute_applescript(running_script)
                running_apps = str(getattr(r_res, "result", "") or "").strip()
                if running_apps:
                    env_info.append(f"Visible running applications: {running_apps}")
            except _OS_AUTOMATION_ENV_ERRORS as exc:
                logger.debug("OSAutomation environment running apps query failed: %s", exc)

        env_context = "\n".join(env_info) if env_info else ""
        prompt = self._build_compiler_prompt(goal, script_type, context, env_context)
        try:
            response = await self._generate(engine, prompt)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("skills.os_automation.compile", exc)
            return {"ok": False, "error": f"Cognitive engine compile failed: {exc}"}

        compiler_fallback = ""
        compiler_error = ""
        try:
            script = self._extract_single_script(response, script_type)
        except ValueError as exc:
            compiler_error = str(exc)
            fallback_script = self._deterministic_script_for_goal(goal, script_type)
            if not fallback_script:
                return {"ok": False, "error": compiler_error}
            record_degradation(
                "skills.os_automation.compile",
                ValueError(compiler_error),
                action="used deterministic OS automation fallback after malformed compiler output",
                severity="warning",
            )
            script = fallback_script
            compiler_fallback = "deterministic_intent_compiler"

        script_hash = hashlib.sha256(script.encode("utf-8")).hexdigest()
        safe, reason = self._validate_script(script_type, script)
        if not safe:
            return {
                "ok": False,
                "error": f"Script blocked by safety guard: {reason}",
                "script_hash": script_hash[:16],
            }

        auth = await self._authorize(script_type, goal, script, script_hash, context)
        if not auth.get("approved"):
            return {
                "ok": False,
                "error": f"Authority denied OS automation: {auth.get('reason', 'blocked')}",
                "status": "blocked_by_authority_gateway",
                "script_hash": script_hash[:16],
            }

        if not params.execute:
            return {
                "ok": True,
                "status": "compiled_validated_not_executed",
                "script_hash": script_hash[:16],
                "authority": auth,
                "script": script,
                "compiler_fallback": compiler_fallback,
                "compiler_error": compiler_error,
            }

        host = get_host_automation()
        if host is None:
            return {"ok": False, "error": "Host automation provider is not available."}

        try:
            from core.governance_context import governed_scope

            async with governed_scope(auth["decision"]):
                if script_type == "applescript":
                    receipt = await host.execute_applescript(script)
                else:
                    receipt = await host.run_command(script)
        except (AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
            record_degradation("skills.os_automation.execute", exc)
            self._finalize(auth, success=False)
            return {"ok": False, "error": f"Execution failed: {exc}", "script_hash": script_hash[:16]}

        self._finalize(auth, success=bool(getattr(receipt, "success", False)))
        return {
            "ok": bool(getattr(receipt, "success", False)),
            "result": getattr(receipt, "result", ""),
            "error": getattr(receipt, "error", ""),
            "receipt_id": getattr(receipt, "receipt_id", ""),
            "authority_receipt_id": auth.get("will_receipt_id") or auth.get("authority_receipt_id"),
            "script_hash": script_hash[:16],
            "adapter": getattr(receipt, "adapter", script_type),
            "script": script,
            "compiler_fallback": compiler_fallback,
            "compiler_error": compiler_error,
        }

    @classmethod
    def _build_compiler_prompt(cls, goal: str, script_type: str, context: Dict[str, Any], env_context: str = "") -> str:
        prompt_parts = []
        prompt_parts.append(
            "You are Aura, a digital cognitive assistant. Compile the user desktop objective "
            f"into one minimal, deterministic, and complete {script_type} script.\n"
            "Safety Constraints: Do not include destructive operations, credential "
            "access, hidden persistence, networking side effects, package installs, "
            "or commands outside the requested visible desktop task.\n"
            "Guidelines for general macOS interaction:\n"
            "- To type text into the frontmost app, you can use AppleScript System Events keystroke / key code commands, or paste from the clipboard.\n"
            "- If the objective requires opening or writing in a web app (like Google Docs) or local app (like Notes), check the active environment and write AppleScript to focus/activate the target application/tab first.\n"
            "- To target Google Docs, ensure Google Chrome (or the active browser) is active, focus the document area (e.g. by clicking or keystroking), and enter/paste the text.\n"
            "- To target Notes, you can use the Notes application script interface directly to create accounts/folders/notes, or use GUI scripting.\n"
            "- Always use appropriate delays (e.g. delay 0.5) to allow UI elements to load and focus to settle.\n"
            f"Respond with exactly one fenced ```{script_type}``` code block and no prose."
        )
        
        # Add research findings if available in context
        research_query = context.get("desktop_task_research_query")
        research_summary = context.get("desktop_task_research_summary")
        if research_summary:
            prompt_parts.append(
                f"Live Research Findings (for query '{research_query}'):\n"
                f"{research_summary}"
            )
            
        if env_context:
            prompt_parts.append(
                "Current macOS Environment Context:\n"
                f"{env_context}"
            )
            
        prompt_parts.append(f"Objective:\n{goal}")
        return "\n\n".join(prompt_parts)

    @staticmethod
    async def _generate(engine: Any, prompt: str) -> str:
        generate = getattr(engine, "generate", None) or getattr(engine, "generate_text", None)
        if not callable(generate):
            raise AttributeError("cognitive engine has no generate method")
        response = generate(prompt, purpose="desktop_os_automation", origin="system")
        if hasattr(response, "__await__"):
            response = await response
        return str(getattr(response, "content", response) or "")

    @staticmethod
    def _extract_single_script(response: str, script_type: str) -> str:
        response_text = str(response or "").strip()
        logger.debug(
            "OSAutomation compiler response received: chars=%d sha256=%s",
            len(response_text),
            hashlib.sha256(response_text.encode("utf-8", errors="replace")).hexdigest()[:16],
        )
        if not response_text:
            raise ValueError("Compiler returned an empty response.")
            
        blocks = list(_CODE_BLOCK_RE.finditer(response_text))
        if len(blocks) >= 1:
            scripts = []
            for b in blocks:
                lang = b.group("lang").strip().lower()
                if lang in {script_type, "osascript", "bash", "sh", ""}:
                    scripts.append(b.group("body").strip())
            if scripts:
                script = "\n\n".join(scripts)
                if len(script) > 10000:
                    raise ValueError(f"Generated script is too long ({len(script)} chars).")
                return script

        if "```" not in response_text:
            lowered = response_text.lower()
            if script_type == "applescript" and any(k in lowered for k in {"tell application", "set ", "keystroke", "activate", "click", "key code"}):
                return response_text
            elif script_type == "bash" and any(k in lowered for k in {"cd ", "echo ", "mkdir ", "rm ", "cp ", "mv ", "curl ", "open "}):
                return response_text
            raise ValueError("Compiler returned conversational prose or non-script text.")
            
        raise ValueError("Compiler response contains malformed code fences or layout.")

    @classmethod
    def _deterministic_script_for_goal(cls, goal: str, script_type: str) -> str:
        if script_type != "applescript":
            return ""
        lowered = str(goal or "").lower()
        script_parts: list[str] = []
        apps = cls._extract_apps(goal)
        if cls._objective_requires_window_arrangement(goal):
            script_parts.append(cls._window_arrangement_script(goal))
        for app in apps:
            script_parts.append(f'tell application {cls._as_applescript_string(app)} to activate')
            script_parts.append("delay 0.3")
        search_url = cls._search_url_from_goal(goal)
        if search_url:
            script_parts.append(f"open location {cls._as_applescript_string(search_url)}")
            script_parts.append("delay 0.5")

        # Focus the local writing/notepad app if one is targeted, so focus is not left on Chrome/Safari
        writing_apps = [app for app in apps if app.lower() not in {"google chrome", "safari", "arc", "firefox", "browser"}]

        text_payload = cls._text_payload_from_goal(goal)
        should_stage_text = cls._should_stage_text(goal)
        needs_editable_surface = any(
            marker in lowered
            for marker in ("note", "textedit", "pages", "word", "document", "google docs", "google doc")
        )
        if should_stage_text:
            script_parts.append(f"set the clipboard to {cls._as_applescript_string(text_payload)}")

        if needs_editable_surface:
            if writing_apps:
                script_parts.append(f'tell application {cls._as_applescript_string(writing_apps[0])} to activate')
                script_parts.append("delay 0.5")
            script_parts.append('tell application "System Events" to keystroke "n" using {command down}')
            script_parts.append("delay 0.3")
            if should_stage_text:
                script_parts.append('tell application "System Events" to keystroke "v" using {command down}')
                script_parts.append("delay 0.2")
        elif should_stage_text:
            script_parts.append(
                'tell application "System Events" to keystroke "v" using {command down}'
            )
            script_parts.append("delay 0.2")
        if not script_parts:
            return ""
        script_parts.append(
            "return "
            + cls._as_applescript_string(
                "Deterministic governed OS automation completed for: "
                + str(goal or "").strip()[:240]
            )
        )
        return "\n".join(part for part in script_parts if part.strip()).strip()

    @staticmethod
    def _as_applescript_string(value: str) -> str:
        text = str(value or "")
        text = text.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "\\r")
        text = text.replace("\n", "\\n")
        return f'"{text}"'

    @staticmethod
    def _extract_apps(goal: str) -> list[str]:
        text = str(goal or "").lower()
        markers = {
            "notes": "Notes",
            "calculator": "Calculator",
            "finder": "Finder",
            "preview": "Preview",
            "safari": "Safari",
            "chrome": "Google Chrome",
            "google docs": "Google Chrome",
            "google doc": "Google Chrome",
            "docs.google": "Google Chrome",
            "browser": "Safari",
            "textedit": "TextEdit",
            "pages": "Pages",
        }
        apps: list[str] = []
        for marker, app in markers.items():
            if re.search(rf"\b{re.escape(marker)}\b", text) and app not in apps:
                if marker == "browser" and ("chrome" in text or "google docs" in text):
                    continue
                apps.append(app)
        open_patterns = (
            r"\bopen\s+(?:up\s+)?(?:my\s+|the\s+)?([A-Za-z][A-Za-z0-9 &._-]{1,60}?)\s+(?:app|application)\b",
            r"\blaunch\s+(?:my\s+|the\s+)?([A-Za-z][A-Za-z0-9 &._-]{1,60}?)\b",
        )
        for pattern in open_patterns:
            for match in re.finditer(pattern, goal, flags=re.IGNORECASE):
                candidate = re.sub(r"\s+", " ", match.group(1)).strip(" ._-")
                if not candidate:
                    continue
                normalized = {
                    "chrome": "Google Chrome",
                    "browser": "Safari",
                    "notes": "Notes",
                }.get(candidate.lower(), candidate)
                if normalized not in apps:
                    apps.append(normalized)
        return apps[:5]

    @staticmethod
    def _objective_requires_window_arrangement(goal: str) -> bool:
        return bool(
            re.search(
                r"\b(?:arrange|resize|drag|minimi[sz]e|maximi[sz]e|organize|tile|snap)\b",
                str(goal or ""),
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _window_arrangement_script(cls, goal: str) -> str:
        lowered = str(goal or "").lower()
        if "right" in lowered:
            bounds = "{960, 25, 1920, 1080}"
        elif "top" in lowered:
            bounds = "{0, 25, 1440, 650}"
        elif "bottom" in lowered:
            bounds = "{0, 650, 1440, 1080}"
        else:
            bounds = "{0, 25, 960, 1080}"
        return f"""
tell application "System Events"
    set frontApp to first application process whose frontmost is true
    if exists window 1 of frontApp then
        set bounds of window 1 of frontApp to {bounds}
    end if
end tell
""".strip()

    @staticmethod
    def _search_query_from_goal(goal: str) -> str:
        text = str(goal or "")
        patterns = (
            r"\bsearch\s+(?:for\s+)?([^.;\n]+)",
            r"\blook\s+up\s+([^.;\n]+)",
            r"\bfind\s+(?:\d+\s+)?(?:different\s+)?(?:articles?|sources?|stories?|news)\s+(?:on|about|for)\s+([^.;\n,]+)",
            r"\b(?:articles?|sources?|stories?|news)\s+(?:on|about|for)\s+([^.;\n,]+)",
            r"\bgoogle\s+([^.;\n]+)",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                query = match.group(1).strip(" ,")
                if query:
                    return query[:240]
        return ""

    @classmethod
    def _search_url_from_goal(cls, goal: str) -> str:
        query = cls._search_query_from_goal(goal)
        if not query:
            return ""
        encoded = urllib.parse.quote_plus(query)
        if "google" in str(goal or "").lower():
            return f"https://www.google.com/search?q={encoded}"
        return f"https://duckduckgo.com/?q={encoded}"

    @staticmethod
    def _should_stage_text(goal: str) -> bool:
        lowered = str(goal or "").lower()
        return bool(
            re.search(r"\b(?:type|paste|write|fill|put|insert|compose|draft)\b", lowered)
            or "google docs" in lowered
            or "note" in lowered
        )

    @classmethod
    def _text_payload_from_goal(cls, goal: str) -> str:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        cleaned = re.sub(r"\s+", " ", str(goal or "").strip())
        return (
            f"Aura governed desktop automation\n"
            f"Timestamp: {stamp}\n"
            f"Objective: {cleaned}\n\n"
            "This text was staged by Aura's deterministic OS automation fallback "
            "after the free-form compiler returned malformed output. The action "
            "still passed script safety validation, AuthorityGateway approval, "
            "and host automation receipt capture before success was claimed."
        )

    # Predefined app automation helpers (like _notes_note_script and _calculator_probe_script)
    # have been removed to preserve a fully generalized dynamic compilation architecture.

    @staticmethod
    def _validate_script(script_type: str, script: str) -> tuple[bool, str]:
        if script_type == "applescript":
            return ScriptASTGuard.validate_applescript(script)
        return ScriptASTGuard.validate_shell_command(script)

    @staticmethod
    async def _authorize(
        script_type: str,
        goal: str,
        script: str,
        script_hash: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            from core.executive.authority_gateway import get_authority_gateway

            gateway = get_authority_gateway()
            decision = await gateway.authorize_environment_action(
                "os_automation_script",
                {
                    "goal": goal[:500],
                    "script_type": script_type,
                    "script_hash": script_hash[:16],
                    "script_preview": script[:500],
                    "user_requested_action": bool(context.get("user_requested_action")),
                },
                source=str(context.get("source") or "os_automation"),
                priority=0.85,
            )
            return {
                "approved": bool(decision.approved),
                "reason": decision.reason,
                "decision": decision,
                "executive_intent_id": decision.executive_intent_id,
                "capability_token_id": decision.capability_token_id,
                "will_receipt_id": decision.will_receipt_id,
                "authority_receipt_id": decision.substrate_receipt_id,
            }
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("skills.os_automation.authority", exc)
            return {"approved": False, "reason": f"authority_gateway_unavailable:{type(exc).__name__}"}

    @staticmethod
    def _finalize(auth: Dict[str, Any], *, success: bool) -> None:
        try:
            from core.executive.authority_gateway import get_authority_gateway

            get_authority_gateway().finalize_tool_execution(
                executive_intent_id=auth.get("executive_intent_id"),
                capability_token_id=auth.get("capability_token_id"),
                success=success,
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("skills.os_automation.finalize", exc)
