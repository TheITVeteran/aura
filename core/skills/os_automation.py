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

        prompt = self._build_compiler_prompt(goal, script_type)
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

    @staticmethod
    def _build_compiler_prompt(goal: str, script_type: str) -> str:
        return (
            "Compile the user desktop objective into one minimal, deterministic "
            f"{script_type} script. Do not include destructive operations, credential "
            "access, hidden persistence, networking side effects, package installs, "
            "or commands outside the requested visible desktop task. Respond with "
            f"exactly one fenced ```{script_type}``` code block and no prose.\n\n"
            f"Objective:\n{goal[:1200]}"
        )

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
        blocks = list(_CODE_BLOCK_RE.finditer(response or ""))
        if len(blocks) != 1:
            raise ValueError("Compiler response must contain exactly one script code block.")
        lang = blocks[0].group("lang").strip().lower()
        if lang and lang != script_type:
            raise ValueError(f"Compiler returned {lang!r}, expected {script_type!r}.")
        script = blocks[0].group("body").strip()
        if not script:
            raise ValueError("Compiler returned an empty script.")
        if "```" in script:
            raise ValueError("Compiler returned nested code fences.")
        if len(script) > 10000:
            raise ValueError(f"Generated script is too long ({len(script)} chars).")
        return script

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
        text_payload = cls._text_payload_from_goal(goal)
        if cls._should_stage_text(goal):
            script_parts.append(f"set the clipboard to {cls._as_applescript_string(text_payload)}")
            if any(app.lower() in {"notes", "textedit", "pages", "google chrome", "safari"} for app in apps):
                script_parts.append(
                    'tell application "System Events" to keystroke "v" using {command down}'
                )
            script_parts.append("delay 0.2")
        if "notes" in lowered or "note" in lowered:
            script_parts.append(cls._notes_note_script(goal, text_payload))
        if "calculator" in lowered:
            script_parts.append(cls._calculator_probe_script(goal))
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

    @classmethod
    def _notes_note_script(cls, goal: str, body: str) -> str:
        lowered = str(goal or "").lower()
        title = "Aura Desktop Automation"
        title_match = re.search(
            r"\b(?:title|titled|called|named)\s+['\"]?([^'\".;\n]{2,80})",
            str(goal or ""),
            flags=re.IGNORECASE,
        )
        if title_match:
            title = title_match.group(1).strip()
        elif "journal" in lowered:
            title = "Aura Journal Entry"
        return f"""
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
    set newNote to make new note at targetFolder with properties {{name:{cls._as_applescript_string(title)}, body:{cls._as_applescript_string(body)}}}
end tell
""".strip()

    @classmethod
    def _calculator_probe_script(cls, goal: str) -> str:
        text = str(goal or "")
        match = re.search(r"\b(\d{1,4})\s*(?:\+|plus)\s*(\d{1,4})\b", text, flags=re.IGNORECASE)
        left = match.group(1) if match else "2"
        right = match.group(2) if match else "3"
        return f"""
tell application "Calculator" to activate
delay 0.3
tell application "System Events"
    keystroke "c"
    delay 0.1
    keystroke {cls._as_applescript_string(left)}
    delay 0.1
    keystroke "+"
    delay 0.1
    keystroke {cls._as_applescript_string(right)}
    delay 0.1
    keystroke "="
end tell
""".strip()

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
