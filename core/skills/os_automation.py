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

        try:
            script = self._extract_single_script(response, script_type)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

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

