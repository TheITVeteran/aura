"""Causally verified, governed natural-language macOS automation.

The skill compiles a bounded desktop objective to AppleScript, executes it
through HostAutomation, and reports success only when read-only observations
prove the objective-specific effect. A transport receipt is audit evidence,
never effect evidence.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import urllib.parse
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, Field

from core.capabilities.host_automation import ScriptASTGuard, get_host_automation
from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.os_automation_effects import (
    DesktopSnapshot,
    EffectContract,
    EffectVerdict,
    build_effect_contract,
    evaluate_effect_contract,
)
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.OSAutomation")

_OS_AUTOMATION_ERRORS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_STRICT_CODE_BLOCK_RE = re.compile(
    r"\A\s*```(?P<lang>[a-zA-Z0-9_+-]+)[ \t]*\r?\n"
    r"(?P<body>.*?)\r?\n```\s*\Z",
    re.DOTALL,
)
_SNAPSHOT_SEPARATOR = "\x1e"
_BASE_SNAPSHOT_SCRIPT = r'''
on replaceText(findText, replacementText, sourceText)
    set oldDelimiters to AppleScript's text item delimiters
    set AppleScript's text item delimiters to findText
    set textItems to text items of sourceText
    set AppleScript's text item delimiters to replacementText
    set sourceText to textItems as text
    set AppleScript's text item delimiters to oldDelimiters
    return sourceText
end replaceText

on cleanText(sourceValue)
    try
        set sourceText to sourceValue as text
    on error
        return ""
    end try
    set sourceText to my replaceText(return, " ", sourceText)
    set sourceText to my replaceText(linefeed, " ", sourceText)
    set sourceText to my replaceText(tab, " ", sourceText)
    return sourceText
end cleanText

set fieldSeparator to ASCII character 30
set appName to ""
set windowTitle to ""
set windowFrame to ""
set desktopFrame to ""
set minimizedValue to ""
set focusValue to ""
set runningText to ""

tell application "System Events"
    try
        set frontProcess to first application process whose frontmost is true
        set appName to my cleanText(name of frontProcess)
        try
            if exists window 1 of frontProcess then
                set windowTitle to my cleanText(name of window 1 of frontProcess)
                set windowPosition to position of window 1 of frontProcess
                set windowSize to size of window 1 of frontProcess
                set windowFrame to ((item 1 of windowPosition) as text) & "," & ((item 2 of windowPosition) as text) & "," & ((item 1 of windowSize) as text) & "," & ((item 2 of windowSize) as text)
                try
                    set minimizedValue to (value of attribute "AXMinimized" of window 1 of frontProcess) as text
                end try
            end if
        end try
        try
            set focusedElement to value of attribute "AXFocusedUIElement" of frontProcess
            set focusValue to my cleanText(value of attribute "AXValue" of focusedElement)
        end try
        try
            set runningNames to name of every application process whose visible is true
            set oldDelimiters to AppleScript's text item delimiters
            set AppleScript's text item delimiters to ", "
            set runningText to runningNames as text
            set AppleScript's text item delimiters to oldDelimiters
        end try
    end try
end tell

return appName & fieldSeparator & windowTitle & fieldSeparator & windowFrame & fieldSeparator & desktopFrame & fieldSeparator & minimizedValue & fieldSeparator & focusValue & fieldSeparator & runningText
'''.strip()


class OSAutomationInput(BaseModel):  # type: ignore[misc]
    goal: str = Field(..., min_length=1, description="High-level desktop objective to accomplish.")
    script_type: Literal["applescript"] = Field(
        "applescript",
        description="AppleScript only; shell execution uses a separate governed skill.",
    )
    execute: bool = Field(True, description="When false, compile and validate without executing.")


class OSAutomationCompilerSkill(BaseSkill):  # type: ignore[misc]
    """Compile, govern, execute, observe, and repair one desktop objective."""

    name = "os_automation"
    description = (
        "General governed macOS desktop automation with objective-specific effect "
        "verification and one bounded corrective attempt."
    )
    input_model = OSAutomationInput
    timeout_seconds = 90.0
    metabolic_cost = 3
    requires_approval = True

    async def execute(
        self,
        params: OSAutomationInput,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        goal = str(params.goal or "").strip()
        if not goal:
            return {"ok": False, "error": "OS automation goal is empty."}

        engine = ServiceContainer.get("cognitive_engine", default=None)
        if engine is None:
            return {"ok": False, "error": "Cognitive engine is not available."}

        text_payload = self._resolved_text_payload(goal, context)
        expected_url = self._search_url_from_goal(goal)
        contract = build_effect_contract(
            goal,
            text_payload=text_payload,
            expected_url=expected_url,
        )
        if not contract.verifiable:
            return {
                "ok": False,
                "status": "objective_not_verifiable",
                "error": (
                    "OS automation refused to act because the objective has no complete "
                    "observable acceptance contract."
                ),
                "effect_contract": contract.to_dict(),
                "effect_verified": False,
            }

        host = get_host_automation()
        if params.execute and host is None:
            return {"ok": False, "error": "Host automation provider is not available."}

        before = DesktopSnapshot()
        observation_errors: list[str] = []
        if params.execute and host is not None:
            before, observation_errors = await self._capture_desktop_snapshot(host, contract)
            pre_verdict = evaluate_effect_contract(contract, before, before)
            if pre_verdict.verified:
                return {
                    "ok": True,
                    "status": "already_satisfied",
                    "effect_verified": True,
                    "effect_evidence": "; ".join(pre_verdict.evidence),
                    "verified_effects": list(pre_verdict.evidence),
                    "verification_results": [check.to_dict() for check in pre_verdict.checks],
                    "effect_contract": contract.to_dict(),
                    "postconditions": self._postconditions(before),
                    "observation_errors": observation_errors,
                    "attempts": [],
                    "manual_reconciliation_required": False,
                }

        compile_context = dict(context)
        if text_payload:
            compile_context["os_automation_text_payload"] = text_payload
        if before.desktop_frame:
            compile_context["os_automation_desktop_frame"] = before.desktop_frame
        env_context = self._environment_context(before, observation_errors)
        try:
            script, compiler = await self._compile_script(
                engine=engine,
                goal=goal,
                context=compile_context,
                env_context=env_context,
                contract=contract,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("skills.os_automation.compile", exc)
            return {
                "ok": False,
                "status": "compiler_failed",
                "error": f"Cognitive engine compile failed: {exc}",
                "effect_contract": contract.to_dict(),
                "effect_verified": False,
            }

        script_hash = hashlib.sha256(script.encode("utf-8")).hexdigest()
        auth = await self._authority_for_script(goal, script, script_hash, context)
        if not auth.get("approved"):
            return self._authority_denial_result(auth, script_hash, contract, compiler)

        if not params.execute:
            closure = self._finalize(auth, success=True)
            closed = bool(closure.get("closed"))
            return {
                "ok": closed,
                "status": "compiled_validated_not_executed" if closed else "authority_closure_failed",
                "error": "" if closed else "Compile-only authority could not be closed cleanly.",
                "script_hash": script_hash[:16],
                "authority": self._public_authority(auth),
                "authority_closure": closure,
                "script": script,
                "compiler": compiler,
                "effect_contract": contract.to_dict(),
                "effect_verified": False,
            }

        if host is None:  # narrowed above; keeps type checkers honest
            return {"ok": False, "error": "Host automation provider is not available."}

        attempts: list[dict[str, Any]] = []
        current_script = script
        current_hash = script_hash
        current_auth = auth
        last_after = before
        last_verdict = evaluate_effect_contract(contract, before, before)
        last_receipt: Any = None
        last_closure: dict[str, Any] = {}

        for attempt_number in (1, 2):
            try:
                receipt = await self._execute_authorized_script(host, current_script, current_auth)
                execution_error = str(getattr(receipt, "error", "") or "")
            except _OS_AUTOMATION_ERRORS as exc:
                record_degradation("skills.os_automation.execute", exc)
                receipt = None
                execution_error = f"Execution raised {type(exc).__name__}: {exc}"

            after, after_errors = await self._capture_desktop_snapshot(host, contract)
            observation_errors.extend(after_errors)
            verdict = evaluate_effect_contract(contract, before, after)
            transport_success = bool(receipt is not None and getattr(receipt, "success", False))
            attempt_success = transport_success and verdict.verified
            closure = self._finalize(current_auth, success=attempt_success)
            closure_ok = bool(closure.get("closed"))
            attempts.append(
                {
                    "attempt": attempt_number,
                    "script_hash": current_hash[:16],
                    "transport_success": transport_success,
                    "transport_error": execution_error,
                    "receipt_id": str(getattr(receipt, "receipt_id", "") or ""),
                    "authority": self._public_authority(current_auth),
                    "authority_closure": closure,
                    "verification": verdict.to_dict(),
                }
            )
            last_after = after
            last_verdict = verdict
            last_receipt = receipt
            last_closure = closure

            if attempt_success and closure_ok:
                return self._success_result(
                    script=current_script,
                    script_hash=current_hash,
                    compiler=compiler,
                    contract=contract,
                    verdict=verdict,
                    before=before,
                    after=after,
                    receipt=receipt,
                    auth=current_auth,
                    closure=closure,
                    attempts=attempts,
                    observation_errors=observation_errors,
                )

            if not closure_ok:
                return self._failure_result(
                    status="authority_closure_failed",
                    error=(
                        "Desktop authority closure failed after the execution attempt; "
                        "manual reconciliation is required before retrying."
                    ),
                    script=current_script,
                    script_hash=current_hash,
                    compiler=compiler,
                    contract=contract,
                    verdict=verdict,
                    before=before,
                    after=after,
                    receipt=receipt,
                    closure=closure,
                    attempts=attempts,
                    observation_errors=observation_errors,
                    manual_reconciliation_required=transport_success,
                )

            if attempt_number == 1:
                try:
                    repaired_script = await self._compile_execution_repair(
                        engine=engine,
                        goal=goal,
                        failed_script=current_script,
                        verdict=verdict,
                        before=before,
                        after=after,
                        contract=contract,
                    )
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    return self._failure_result(
                        status="repair_compile_failed",
                        error=f"Effect verification and repair compilation failed: {exc}",
                        script=current_script,
                        script_hash=current_hash,
                        compiler=compiler,
                        contract=contract,
                        verdict=verdict,
                        before=before,
                        after=after,
                        receipt=receipt,
                        closure=closure,
                        attempts=attempts,
                        observation_errors=observation_errors,
                    )
                repaired_hash = hashlib.sha256(repaired_script.encode("utf-8")).hexdigest()
                if repaired_hash == current_hash and transport_success:
                    return self._failure_result(
                        status="repair_made_no_change",
                        error="Verification failed and the repair compiler returned the same script.",
                        script=current_script,
                        script_hash=current_hash,
                        compiler=compiler,
                        contract=contract,
                        verdict=verdict,
                        before=before,
                        after=after,
                        receipt=receipt,
                        closure=closure,
                        attempts=attempts,
                        observation_errors=observation_errors,
                    )
                current_script = repaired_script
                current_hash = repaired_hash
                current_auth = await self._authority_for_script(
                    goal,
                    current_script,
                    current_hash,
                    context,
                )
                if not current_auth.get("approved"):
                    denial = self._authority_denial_result(
                        current_auth,
                        current_hash,
                        contract,
                        compiler,
                    )
                    denial["attempts"] = attempts
                    return denial

        failure_reasons = "; ".join(last_verdict.failure_reasons)
        transport_error = str(getattr(last_receipt, "error", "") or "")
        return self._failure_result(
            status="effect_verification_failed",
            error=(
                "OS automation exhausted one bounded repair without proving the requested effect. "
                + (failure_reasons or transport_error or "No objective-specific effect was observed.")
            ),
            script=current_script,
            script_hash=current_hash,
            compiler=compiler,
            contract=contract,
            verdict=last_verdict,
            before=before,
            after=last_after,
            receipt=last_receipt,
            closure=last_closure,
            attempts=attempts,
            observation_errors=observation_errors,
        )

    @classmethod
    async def _compile_script(
        cls,
        *,
        engine: Any,
        goal: str,
        context: dict[str, Any],
        env_context: str,
        contract: EffectContract,
    ) -> tuple[str, dict[str, Any]]:
        prompt = cls._build_compiler_prompt(goal, context, env_context, contract)
        response = await cls._generate(engine, prompt)
        attempts: list[dict[str, str]] = []
        first_failure = ""
        try:
            script = cls._extract_single_script(response, "applescript")
            safe, reason = cls._validate_script("applescript", script)
            if not safe:
                raise ValueError(f"Script blocked by safety guard: {reason}")
        except ValueError as exc:
            first_failure = str(exc)
            attempts.append(cls._compiler_attempt("initial", response, first_failure))
            correction_prompt = cls._build_compiler_correction_prompt(
                goal=goal,
                failed_response=response,
                failure=first_failure,
                contract=contract,
            )
            corrected = await cls._generate(engine, correction_prompt)
            try:
                script = cls._extract_single_script(corrected, "applescript")
                safe, reason = cls._validate_script("applescript", script)
                if not safe:
                    raise ValueError(f"Script blocked by safety guard: {reason}")
            except ValueError as correction_exc:
                correction_failure = str(correction_exc)
                attempts.append(
                    cls._compiler_attempt("format_or_safety_repair", corrected, correction_failure)
                )
                fallback_context = dict(context)
                resolved_text = cls._resolved_text_payload(goal, context)
                if resolved_text:
                    fallback_context["text_payload"] = resolved_text
                fallback = cls._deterministic_script_for_goal(goal, fallback_context)
                fallback_safe, fallback_reason = cls._validate_script("applescript", fallback)
                if not fallback or not fallback_safe:
                    raise ValueError(
                        f"{first_failure}; correction failed: {correction_failure}; "
                        f"fallback unavailable: {fallback_reason}"
                    ) from correction_exc
                record_degradation(
                    "skills.os_automation.compile",
                    ValueError(first_failure),
                    action="used deterministic OS automation fallback after bounded compiler repair",
                    severity="warning",
                )
                return fallback, {
                    "fallback": "deterministic_intent_compiler",
                    "recovered": True,
                    "attempts": attempts,
                }
            attempts.append(cls._compiler_attempt("format_or_safety_repair", corrected, ""))
            return script, {"fallback": "", "recovered": True, "attempts": attempts}
        attempts.append(cls._compiler_attempt("initial", response, ""))
        return script, {"fallback": "", "recovered": False, "attempts": attempts}

    @classmethod
    async def _compile_execution_repair(
        cls,
        *,
        engine: Any,
        goal: str,
        failed_script: str,
        verdict: EffectVerdict,
        before: DesktopSnapshot,
        after: DesktopSnapshot,
        contract: EffectContract,
    ) -> str:
        prompt = (
            "Repair one governed AppleScript after objective-specific verification failed.\n"
            "Return exactly one fenced ```applescript``` block and no prose. Do not use "
            "`do shell script`. Preserve only actions needed for the objective and failed checks.\n\n"
            f"Objective:\n{goal}\n\n"
            f"Effect contract:\n{contract.to_dict()}\n\n"
            f"Failed checks:\n{list(verdict.failure_reasons)}\n\n"
            f"Before snapshot:\n{before.to_dict()}\n\n"
            f"After snapshot:\n{after.to_dict()}\n\n"
            f"Failed script:\n```applescript\n{failed_script[:10000]}\n```"
        )
        response = await cls._generate(engine, prompt)
        script = cls._extract_single_script(response, "applescript")
        safe, reason = cls._validate_script("applescript", script)
        if not safe:
            raise ValueError(f"Repair script blocked by safety guard: {reason}")
        return script

    @staticmethod
    def _compiler_attempt(stage: str, response: str, error: str) -> dict[str, str]:
        value = str(response or "")
        return {
            "stage": stage,
            "response_sha256": hashlib.sha256(
                value.encode("utf-8", errors="replace")
            ).hexdigest()[:16],
            "error": error,
        }

    @staticmethod
    def _build_compiler_correction_prompt(
        *,
        goal: str,
        failed_response: str,
        failure: str,
        contract: EffectContract,
    ) -> str:
        return (
            "Correct a malformed or unsafe AppleScript compiler response.\n"
            f"Failure: {failure}\n"
            f"Objective: {goal}\n"
            f"Acceptance contract: {contract.to_dict()}\n"
            "Return exactly one fenced ```applescript``` block and no prose. "
            "Do not use `do shell script`.\n\n"
            f"Prior response:\n{str(failed_response or '')[:10000]}"
        )

    @classmethod
    def _build_compiler_prompt(
        cls,
        goal: str,
        context: dict[str, Any],
        env_context: str = "",
        contract: EffectContract | None = None,
    ) -> str:
        prompt_parts = [
            (
                "Compile the desktop objective into one minimal, deterministic, complete "
                "AppleScript. Return exactly one fenced ```applescript``` code block and no prose.\n"
                "Constraints:\n"
                "- Do not use `do shell script`, destructive operations, credential access, "
                "hidden persistence, package installation, or unrelated actions.\n"
                "- Activate and verify the intended app before typing, clicking, or moving a window.\n"
                "- Use bounded delays only where focus or UI loading requires them.\n"
                "- Make the requested effect observable by the acceptance contract.\n"
                "- Never return success text as a substitute for changing the requested UI state."
            )
        ]
        if contract is not None:
            prompt_parts.append(f"Acceptance contract:\n{contract.to_dict()}")
        text_payload = str(context.get("os_automation_text_payload") or "").strip()
        if text_payload:
            prompt_parts.append(
                "Exact text payload to place in the requested editing surface:\n"
                + text_payload[:9000]
            )
        research_summary = str(context.get("desktop_task_research_summary") or "").strip()
        if research_summary:
            prompt_parts.append(
                "Bounded research context for the requested work product:\n"
                + research_summary[:6000]
            )
        if env_context:
            prompt_parts.append("Current read-only macOS observations:\n" + env_context)
        prompt_parts.append(f"Objective:\n{goal}")
        return "\n\n".join(prompt_parts)

    @staticmethod
    async def _generate(engine: Any, prompt: str) -> str:
        generate = getattr(engine, "generate", None) or getattr(engine, "generate_text", None)
        if not callable(generate):
            raise AttributeError("cognitive engine has no generate method")
        response = generate(
            prompt,
            purpose="desktop_os_automation",
            origin="user",
            is_background=False,
            prefer_tier="primary",
            use_strategies=False,
            max_tokens=1800,
            temperature=0.0,
        )
        if hasattr(response, "__await__"):
            response = await response
        return str(getattr(response, "content", response) or "")

    @staticmethod
    def _extract_single_script(response: str, script_type: str) -> str:
        if script_type != "applescript":
            raise ValueError("OS automation accepts AppleScript only.")
        response_text = str(response or "")
        logger.debug(
            "OSAutomation compiler response: chars=%d sha256=%s",
            len(response_text),
            hashlib.sha256(
                response_text.encode("utf-8", errors="replace")
            ).hexdigest()[:16],
        )
        if not response_text.strip():
            raise ValueError("Compiler returned an empty response.")
        match = _STRICT_CODE_BLOCK_RE.fullmatch(response_text)
        if match is None:
            raise ValueError(
                "Compiler must return exactly one fenced AppleScript block with no surrounding prose."
            )
        language = match.group("lang").strip().lower()
        if language != "applescript":
            raise ValueError(f"Compiler returned the wrong fenced language: {language or 'none'}.")
        script = match.group("body").strip()
        if not script:
            raise ValueError("Compiler returned an empty AppleScript block.")
        if "```" in script:
            raise ValueError("Compiler returned nested or multiple code fences.")
        if len(script) > 10000:
            raise ValueError(f"Generated script is too long ({len(script)} chars).")
        return script

    @staticmethod
    def _validate_script(script_type: str, script: str) -> tuple[bool, str]:
        if script_type != "applescript":
            return False, "OS automation accepts AppleScript only"
        if re.search(r"\bdo\s+shell\s+script\b", script, flags=re.IGNORECASE):
            return False, "Embedded shell execution belongs in the separately governed shell lane"
        safe, reason = ScriptASTGuard.validate_applescript(script)
        return bool(safe), str(reason)

    @classmethod
    async def _capture_desktop_snapshot(
        cls,
        host: Any,
        contract: EffectContract,
    ) -> tuple[DesktopSnapshot, list[str]]:
        errors: list[str] = []
        values: dict[str, object] = {}
        inspect_script = getattr(host, "inspect_applescript", None)
        if not callable(inspect_script):
            return DesktopSnapshot(), ["read_only_applescript_inspection_unavailable"]

        try:
            receipt = await inspect_script(
                _BASE_SNAPSHOT_SCRIPT,
                timeout_s=5.0,
                source="os_automation.desktop_snapshot",
            )
            if bool(getattr(receipt, "success", False)):
                raw_result = getattr(receipt, "result", "")
                if isinstance(raw_result, Mapping):
                    values.update(dict(raw_result))
                else:
                    fields = str(raw_result or "").split(_SNAPSHOT_SEPARATOR)
                    if len(fields) == 7:
                        values.update(
                            {
                                "frontmost_app": fields[0],
                                "frontmost_window": fields[1],
                                "window_frame": fields[2],
                                "desktop_frame": fields[3],
                                "window_minimized": fields[4],
                                "focused_value_excerpt": fields[5],
                                "running_apps": fields[6],
                            }
                        )
                    else:
                        errors.append(f"desktop_snapshot_field_count:{len(fields)}")
            else:
                errors.append(
                    "desktop_snapshot_failed:"
                    + str(getattr(receipt, "error", "unknown") or "unknown")[:240]
                )
        except _OS_AUTOMATION_ERRORS as exc:
            errors.append(f"desktop_snapshot_exception:{type(exc).__name__}")

        if not values.get("desktop_frame"):
            get_desktop_frame = getattr(host, "get_desktop_frame", None)
            if callable(get_desktop_frame):
                try:
                    desktop_receipt = await get_desktop_frame()
                    if bool(getattr(desktop_receipt, "success", False)):
                        values["desktop_frame"] = getattr(desktop_receipt, "result", None)
                    else:
                        errors.append("desktop_frame_snapshot_failed")
                except _OS_AUTOMATION_ERRORS as exc:
                    errors.append(f"desktop_frame_snapshot_exception:{type(exc).__name__}")

        snapshot = DesktopSnapshot.from_mapping(values)
        if contract.needs_browser_url:
            browser_script = cls._browser_url_probe(snapshot.frontmost_app)
            if browser_script:
                try:
                    browser_receipt = await inspect_script(
                        browser_script,
                        timeout_s=4.0,
                        source="os_automation.browser_url_snapshot",
                    )
                    if bool(getattr(browser_receipt, "success", False)):
                        values["browser_url"] = str(
                            getattr(browser_receipt, "result", "") or ""
                        )
                    else:
                        errors.append("browser_url_snapshot_failed")
                except _OS_AUTOMATION_ERRORS as exc:
                    errors.append(f"browser_url_snapshot_exception:{type(exc).__name__}")

        if contract.needs_screen_text:
            read_screen = getattr(host, "get_screen_text", None)
            if callable(read_screen):
                try:
                    screen_receipt = await read_screen(retain_screenshot=False)
                    screen_text = str(getattr(screen_receipt, "result", "") or "").strip()
                    if bool(getattr(screen_receipt, "success", False)) and not screen_text.startswith("["):
                        values["screen_text"] = screen_text[:4000]
                    elif not values.get("focused_value_excerpt"):
                        errors.append("screen_text_snapshot_unavailable")
                except _OS_AUTOMATION_ERRORS as exc:
                    errors.append(f"screen_text_snapshot_exception:{type(exc).__name__}")

        return DesktopSnapshot.from_mapping(values), errors

    @staticmethod
    def _browser_url_probe(frontmost_app: str) -> str:
        app = str(frontmost_app or "").strip()
        if app in {"Google Chrome", "Arc", "Microsoft Edge", "Brave Browser"}:
            quoted = OSAutomationCompilerSkill._as_applescript_string(app)
            return (
                f"tell application {quoted}\n"
                'if (count of windows) is 0 then return ""\n'
                "return URL of active tab of front window\n"
                "end tell"
            )
        if app == "Safari":
            return (
                'tell application "Safari"\n'
                'if (count of windows) is 0 then return ""\n'
                "return URL of current tab of front window\n"
                "end tell"
            )
        return ""

    @staticmethod
    def _environment_context(snapshot: DesktopSnapshot, errors: list[str]) -> str:
        lines: list[str] = []
        if snapshot.frontmost_app:
            lines.append(f"Frontmost application: {snapshot.frontmost_app}")
        if snapshot.frontmost_window:
            lines.append(f"Frontmost window: {snapshot.frontmost_window}")
        if snapshot.window_frame:
            lines.append(f"Window frame x,y,width,height: {snapshot.window_frame}")
        if snapshot.desktop_frame:
            lines.append(f"Desktop frame x,y,width,height: {snapshot.desktop_frame}")
        if snapshot.browser_url:
            lines.append(f"Active browser URL: {snapshot.browser_url}")
        if snapshot.focused_value_excerpt:
            lines.append(f"Focused value: {snapshot.focused_value_excerpt[:1200]}")
        if snapshot.running_apps:
            lines.append("Visible running applications: " + ", ".join(snapshot.running_apps))
        if errors:
            lines.append("Unavailable observations: " + ", ".join(dict.fromkeys(errors)))
        return "\n".join(lines)

    @classmethod
    def _resolved_text_payload(cls, goal: str, context: Mapping[str, Any]) -> str:
        for key in (
            "desktop_task_document_body",
            "document_body",
            "body",
            "content",
            "draft",
            "text_payload",
        ):
            candidate = str(context.get(key) or "").strip()
            if candidate and not cls._looks_like_automation_narration(candidate):
                return candidate[:9000]
        quoted = re.search(
            r"\b(?:type|paste|write|fill|insert|enter)\s+(?:the\s+text\s+)?[\"']([^\"']{1,9000})[\"']",
            goal,
            flags=re.IGNORECASE,
        )
        if quoted:
            return quoted.group(1).strip()
        direct = re.search(
            r"\b(?:type|paste|enter)\s+(.+?)(?=\s+\b(?:into|in|to|and then|then)\b|[.;]|$)",
            goal,
            flags=re.IGNORECASE,
        )
        if direct:
            return direct.group(1).strip(" \"'")[:9000]
        if cls._extract_writing_topic(goal) or re.search(
            r"\b(?:note|document|paragraph|essay|summary|report|journal\s+entry)\b",
            goal,
            flags=re.IGNORECASE,
        ):
            return cls._text_payload_from_goal(goal, context)
        return ""

    @classmethod
    def _deterministic_script_for_goal(
        cls,
        goal: str,
        script_type_or_context: str | Mapping[str, Any] = "applescript",
        context: Mapping[str, Any] | None = None,
    ) -> str:
        if isinstance(script_type_or_context, Mapping):
            context = script_type_or_context
            script_type = "applescript"
        else:
            script_type = str(script_type_or_context or "applescript").lower()
        if script_type != "applescript":
            return ""

        context = context or {}
        lowered = str(goal or "").lower()
        script_parts: list[str] = []
        apps = cls._extract_apps(goal)
        for app in apps:
            script_parts.append(f"tell application {cls._as_applescript_string(app)} to activate")
            script_parts.append("delay 0.4")

        search_url = cls._search_url_from_goal(goal)
        if search_url:
            script_parts.append(f"open location {cls._as_applescript_string(search_url)}")
            script_parts.append("delay 0.8")

        if cls._objective_requires_window_arrangement(goal):
            raw_frame = context.get("os_automation_desktop_frame")
            desktop_frame: tuple[int, int, int, int] | None = None
            if isinstance(raw_frame, (list, tuple)) and len(raw_frame) == 4:
                try:
                    converted = tuple(int(value) for value in raw_frame)
                    desktop_frame = (
                        converted[0],
                        converted[1],
                        converted[2],
                        converted[3],
                    )
                except (TypeError, ValueError):
                    desktop_frame = None
            arrangement = cls._window_arrangement_script(goal, desktop_frame)
            if not arrangement:
                return ""
            script_parts.append(arrangement)

        text_payload = cls._resolved_text_payload(goal, context)
        requests_text = bool(
            re.search(r"\b(?:type|paste|write|fill|insert|compose|draft|enter)\b", lowered)
            or "google docs" in lowered
        )
        if requests_text and text_payload:
            writing_apps = [
                app
                for app in apps
                if app.lower()
                not in {"google chrome", "safari", "arc", "firefox", "brave browser"}
            ]
            script_parts.append(
                f"set the clipboard to {cls._as_applescript_string(text_payload)}"
            )
            if writing_apps:
                script_parts.append(
                    f"tell application {cls._as_applescript_string(writing_apps[0])} to activate"
                )
                script_parts.append("delay 0.5")
            if re.search(r"\b(?:note|document|google docs?|textedit|pages|word)\b", lowered):
                script_parts.append(
                    'tell application "System Events" to keystroke "n" using {command down}'
                )
                script_parts.append("delay 0.4")
            script_parts.append(
                'tell application "System Events" to keystroke "v" using {command down}'
            )
            script_parts.append("delay 0.4")

        if not script_parts:
            return ""
        script_parts.append('return "OS automation action dispatched; verify observable state."')
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
            "google docs": "Google Chrome",
            "google chrome": "Google Chrome",
            "chrome": "Google Chrome",
            "calculator": "Calculator",
            "textedit": "TextEdit",
            "microsoft word": "Microsoft Word",
            "preview": "Preview",
            "finder": "Finder",
            "safari": "Safari",
            "notes": "Notes",
            "pages": "Pages",
        }
        apps: list[str] = []
        for marker, app in markers.items():
            if re.search(rf"\b{re.escape(marker)}\b", text) and app not in apps:
                apps.append(app)
        patterns = (
            r"\bopen\s+(?:up\s+)?(?:my\s+|the\s+)?([A-Za-z][A-Za-z0-9 &._-]{1,60}?)\s+(?:app|application)\b",
            r"\blaunch\s+(?:my\s+|the\s+)?([A-Za-z][A-Za-z0-9 &._-]{1,60}?)(?=\s*(?:,|\.|;|\band\b|$))",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, goal, flags=re.IGNORECASE):
                candidate = re.sub(r"\s+", " ", match.group(1)).strip(" ._-")
                normalized = {
                    "chrome": "Google Chrome",
                    "notes": "Notes",
                }.get(candidate.lower(), candidate)
                if normalized and normalized.lower() != "browser" and normalized not in apps:
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
    def _window_arrangement_script(
        cls,
        goal: str,
        desktop_frame: tuple[int, int, int, int] | None = None,
    ) -> str:
        lowered = str(goal or "").lower()
        if re.search(r"\bminimi[sz](?:e|ed|ing)?\b", lowered):
            return '''
tell application "System Events"
    set frontProcess to first application process whose frontmost is true
    if exists window 1 of frontProcess then set value of attribute "AXMinimized" of window 1 of frontProcess to true
end tell
'''.strip()

        if desktop_frame is None:
            return ""
        screen_x, screen_y, screen_width, screen_height = desktop_frame
        if min(screen_width, screen_height) <= 0:
            return ""
        if "right" in lowered:
            position_x = screen_x + screen_width // 2
            position_y = screen_y
            width = screen_width - screen_width // 2
            height = screen_height
        elif "top" in lowered:
            position_x = screen_x
            position_y = screen_y
            width = screen_width
            height = screen_height // 2
        elif "bottom" in lowered:
            position_x = screen_x
            position_y = screen_y + screen_height // 2
            width = screen_width
            height = screen_height - screen_height // 2
        elif re.search(r"\bmaximi[sz](?:e|ed|ing)?\b", lowered):
            position_x = screen_x
            position_y = screen_y
            width = screen_width
            height = screen_height
        elif "left" in lowered:
            position_x = screen_x
            position_y = screen_y
            width = screen_width // 2
            height = screen_height
        else:
            position_x = screen_x + screen_width // 8
            position_y = screen_y + screen_height // 8
            width = (screen_width * 3) // 4
            height = (screen_height * 3) // 4
        return f'''
tell application "System Events"
    set frontProcess to first application process whose frontmost is true
    if exists window 1 of frontProcess then
        set position of window 1 of frontProcess to {{{position_x}, {position_y}}}
        set size of window 1 of frontProcess to {{{width}, {height}}}
    end if
end tell
'''.strip()

    @staticmethod
    def _search_query_from_goal(goal: str) -> str:
        patterns = (
            r"\bsearch\s+(?:google\s+)?(?:for\s+)?([^.;\n]+)",
            r"\blook\s+up\s+([^.;\n]+)",
            r"\bgoogle\s+([^.;\n]+)",
        )
        for pattern in patterns:
            match = re.search(pattern, str(goal or ""), flags=re.IGNORECASE)
            if match:
                query = match.group(1).strip(" ,")
                if query:
                    return query[:240]
        return ""

    @classmethod
    def _search_url_from_goal(cls, goal: str) -> str:
        explicit = re.search(r"https?://[^\s<>\"']+", str(goal or ""), flags=re.IGNORECASE)
        if explicit:
            return explicit.group(0).rstrip(".,);]")
        query = cls._search_query_from_goal(goal)
        if not query:
            return ""
        encoded = urllib.parse.quote_plus(query)
        if "google" in str(goal or "").lower():
            return f"https://www.google.com/search?q={encoded}"
        return f"https://duckduckgo.com/?q={encoded}"

    @classmethod
    def _text_payload_from_goal(
        cls,
        goal: str,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        context = context or {}
        for key in (
            "desktop_task_document_body",
            "document_body",
            "body",
            "content",
            "draft",
            "text_payload",
            "os_automation_text_payload",
        ):
            candidate = str(context.get(key) or "").strip()
            if candidate and not cls._looks_like_automation_narration(candidate):
                return candidate[:9000]

        topic = cls._extract_writing_topic(goal)
        timestamp = ""
        if re.search(r"\b(?:timestamp|time stamp|date stamp|dated)\b", goal, re.IGNORECASE):
            timestamp = time.strftime("[%Y-%m-%d %H:%M:%S %Z] ")
        if topic:
            return (
                f"{timestamp}{topic[:1].upper() + topic[1:]}. "
                f"This note captures the requested subject clearly and keeps the central point "
                f"visible for later use: {topic}."
            )[:9000]
        return (
            f"{timestamp}Status note: the requested desktop work is ready for review. "
            "The visible content is recorded here so its completion can be read back directly."
        )[:9000]

    @staticmethod
    def _looks_like_automation_narration(text: str) -> bool:
        lowered = str(text or "").lower()
        return any(
            marker in lowered
            for marker in (
                "aura governed desktop automation",
                "aura desktop task receipt",
                "canonical computer-use gateway",
                "deterministic os automation fallback",
                "host automation receipt",
                "authoritygateway approval",
            )
        )

    @staticmethod
    def _extract_writing_topic(goal: str) -> str:
        text = " ".join(str(goal or "").strip().split())
        patterns = (
            r"\b(?:write|draft|compose|type|create)\s+(?:me\s+)?(?:a\s+|an\s+)?(?:short\s+|full\s+|one\s+)?(?:paragraph|note|document|essay|summary|report|journal\s+entry)\s+(?:about|on|describing|explaining)\s+(.+)$",
            r"\b(?:write|draft|compose|type)\s+(.+?)\s+(?:in|into|to)\s+(?:notes|google docs|docs|a note|the note)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            topic = re.split(
                r"\b(?:and then|then|after that|also|export|save|create a folder|make a folder)\b",
                str(match.group(1) or ""),
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip(" .,:;?!\"'")
            if topic:
                return topic[:220]
        return ""

    @classmethod
    async def _authority_for_script(
        cls,
        goal: str,
        script: str,
        script_hash: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if bool(context.get("_capability_token_verified")):
            return {
                "approved": True,
                "reason": "delegated_capability_engine_authority",
                "delegated": True,
                "capability_token_id": context.get("capability_token_id"),
                "will_receipt_id": context.get("will_receipt_id"),
                "authority_receipt_id": context.get("authority_receipt_id"),
            }
        return await cls._authorize(goal, script, script_hash, context)

    @staticmethod
    async def _authorize(
        goal: str,
        script: str,
        script_hash: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            from core.executive.authority_gateway import get_authority_gateway

            decision = await get_authority_gateway().authorize_environment_action(
                "os_automation_script",
                {
                    "goal": goal[:500],
                    "script_type": "applescript",
                    "script_hash": script_hash[:16],
                    "script_preview": script[:500],
                    "user_requested_action": bool(context.get("user_requested_action")),
                },
                source=str(context.get("source") or context.get("origin") or "os_automation"),
                priority=0.85,
            )
            return {
                "approved": bool(decision.approved),
                "reason": decision.reason,
                "decision": decision,
                "delegated": False,
                "executive_intent_id": decision.executive_intent_id,
                "capability_token_id": decision.capability_token_id,
                "will_receipt_id": decision.will_receipt_id,
                "authority_receipt_id": decision.substrate_receipt_id,
            }
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("skills.os_automation.authority", exc)
            return {
                "approved": False,
                "reason": f"authority_gateway_unavailable:{type(exc).__name__}",
                "delegated": False,
            }

    @staticmethod
    def _finalize(auth: dict[str, Any], *, success: bool) -> dict[str, Any]:
        if auth.get("delegated"):
            return {
                "closed": True,
                "mode": "delegated_to_capability_engine",
                "pending_outer_closure": True,
                "success": success,
            }
        try:
            from core.executive.authority_gateway import get_authority_gateway

            result = get_authority_gateway().finalize_tool_execution(
                executive_intent_id=auth.get("executive_intent_id"),
                capability_token_id=auth.get("capability_token_id"),
                success=success,
            )
            if isinstance(result, dict):
                return dict(result)
            return {
                "closed": False,
                "mode": "direct",
                "success": success,
                "errors": ["authority gateway returned no closure receipt"],
            }
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("skills.os_automation.finalize", exc)
            return {
                "closed": False,
                "mode": "direct",
                "success": success,
                "errors": [f"{type(exc).__name__}:{exc}"],
            }

    @staticmethod
    async def _execute_authorized_script(
        host: Any,
        script: str,
        auth: dict[str, Any],
    ) -> Any:
        if auth.get("delegated"):
            return await host.execute_applescript(script)
        decision = auth.get("decision")
        if decision is None:
            raise RuntimeError("Direct OS automation authority is missing its decision scope.")
        from core.governance_context import governed_scope

        async with governed_scope(decision):
            return await host.execute_applescript(script)

    @staticmethod
    def _postconditions(snapshot: DesktopSnapshot) -> dict[str, object]:
        result: dict[str, object] = dict(snapshot.to_dict())
        if snapshot.window_frame:
            result["frontmost_window_bounds"] = ",".join(
                str(value) for value in snapshot.window_frame
            )
        return result

    @staticmethod
    def _public_authority(auth: dict[str, Any]) -> dict[str, object]:
        return {
            "approved": bool(auth.get("approved")),
            "reason": str(auth.get("reason") or ""),
            "mode": "delegated" if auth.get("delegated") else "direct",
            "capability_token_id": str(auth.get("capability_token_id") or ""),
            "will_receipt_id": str(auth.get("will_receipt_id") or ""),
            "authority_receipt_id": str(auth.get("authority_receipt_id") or ""),
        }

    @classmethod
    def _authority_denial_result(
        cls,
        auth: dict[str, Any],
        script_hash: str,
        contract: EffectContract,
        compiler: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "error": f"Authority denied OS automation: {auth.get('reason', 'blocked')}",
            "status": "blocked_by_authority_gateway",
            "script_hash": script_hash[:16],
            "authority": cls._public_authority(auth),
            "compiler": compiler,
            "effect_contract": contract.to_dict(),
            "effect_verified": False,
        }

    @classmethod
    def _success_result(
        cls,
        *,
        script: str,
        script_hash: str,
        compiler: dict[str, Any],
        contract: EffectContract,
        verdict: EffectVerdict,
        before: DesktopSnapshot,
        after: DesktopSnapshot,
        receipt: Any,
        auth: dict[str, Any],
        closure: dict[str, Any],
        attempts: list[dict[str, Any]],
        observation_errors: list[str],
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "status": "completed_verified",
            "result": getattr(receipt, "result", ""),
            "error": "",
            "receipt_id": getattr(receipt, "receipt_id", ""),
            "authority_receipt_id": auth.get("will_receipt_id")
            or auth.get("authority_receipt_id"),
            "authority": cls._public_authority(auth),
            "authority_closure": closure,
            "script_hash": script_hash[:16],
            "adapter": getattr(receipt, "adapter", "applescript"),
            "script": script,
            "compiler": compiler,
            "effect_contract": contract.to_dict(),
            "effect_verified": True,
            "effect_evidence": "; ".join(verdict.evidence),
            "verified_effects": list(verdict.evidence),
            "verification_results": [check.to_dict() for check in verdict.checks],
            "preconditions": cls._postconditions(before),
            "postconditions": cls._postconditions(after),
            "observation_errors": list(dict.fromkeys(observation_errors)),
            "attempts": attempts,
            "manual_reconciliation_required": False,
        }

    @classmethod
    def _failure_result(
        cls,
        *,
        status: str,
        error: str,
        script: str,
        script_hash: str,
        compiler: dict[str, Any],
        contract: EffectContract,
        verdict: EffectVerdict,
        before: DesktopSnapshot,
        after: DesktopSnapshot,
        receipt: Any,
        closure: dict[str, Any],
        attempts: list[dict[str, Any]],
        observation_errors: list[str],
        manual_reconciliation_required: bool = False,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "status": status,
            "error": error,
            "result": getattr(receipt, "result", "") if receipt is not None else "",
            "receipt_id": getattr(receipt, "receipt_id", "") if receipt is not None else "",
            "script_hash": script_hash[:16],
            "script": script,
            "compiler": compiler,
            "effect_contract": contract.to_dict(),
            "effect_verified": False,
            "effect_evidence": "; ".join(verdict.evidence),
            "verified_effects": list(verdict.evidence),
            "verification_results": [check.to_dict() for check in verdict.checks],
            "preconditions": cls._postconditions(before),
            "postconditions": cls._postconditions(after),
            "authority_closure": closure,
            "observation_errors": list(dict.fromkeys(observation_errors)),
            "attempts": attempts,
            "manual_reconciliation_required": manual_reconciliation_required,
        }
