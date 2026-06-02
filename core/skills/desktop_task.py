from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator

from core.runtime.errors import record_degradation
from core.skills.base_skill import BaseSkill


class DesktopTaskStep(BaseModel):
    action: str = Field(
        ...,
        description=(
            "One computer_use action: click, type, hotkey, scroll, read_screen_text, "
            "read_menu_clock, open_app, open_url, run_command, set_clipboard, "
            "get_clipboard, wait, run_applescript, write_text_file, render_text_pdf, move_file"
        ),
    )
    target: str | dict[str, Any] = Field("", description="Text, command, URL, app name, script, or JSON action target")
    x: int = Field(0, description="Screen x coordinate for click/scroll/focus")
    y: int = Field(0, description="Screen y coordinate for click/scroll/focus")
    reason: str = Field("", description="Short reason for this step")
    expect: str = Field("", description="Expected observable result")

    @field_validator("action")
    @classmethod
    def _normalize_action(cls, value: str) -> str:
        action = str(value or "").strip().lower()
        allowed = {
            "click",
            "type",
            "hotkey",
            "scroll",
            "read_screen_text",
            "read_menu_clock",
            "open_app",
            "open_url",
            "run_command",
            "set_clipboard",
            "get_clipboard",
            "wait",
            "run_applescript",
            "write_text_file",
            "render_text_pdf",
            "move_file",
        }
        if action not in allowed:
            raise ValueError(f"Unsupported desktop action: {value}")
        return action


class DesktopTaskParams(BaseModel):
    objective: str = Field("", description="Natural-language task objective")
    steps: list[DesktopTaskStep] = Field(default_factory=list, description="Bounded ordered desktop action plan")
    stop_on_error: bool = Field(True, description="Stop after the first failed step")

    @field_validator("steps")
    @classmethod
    def _bounded_steps(cls, value: list[DesktopTaskStep]) -> list[DesktopTaskStep]:
        if not value:
            raise ValueError("Desktop task requires at least one step.")
        if len(value) > 20:
            raise ValueError("Desktop task cannot exceed 20 steps.")
        return value


class DesktopTaskSkill(BaseSkill):
    name = "desktop_task"
    description = (
        "Execute a bounded, receipt-producing multi-step desktop plan through "
        "Aura's governed computer_use body. Use for arbitrary chained computer "
        "tasks that need app control, clipboard, browser/app UI, files, PDFs, "
        "or verification steps."
    )
    input_model = DesktopTaskParams
    metabolic_cost = 2
    effect_scope = "foreground_desktop_control"
    timeout_seconds = 180.0

    async def execute(self, params: Any, context: dict[str, Any]) -> dict[str, Any]:
        if isinstance(params, dict):
            params = DesktopTaskParams(**params)

        try:
            from core.container import ServiceContainer

            capability_engine = ServiceContainer.get("capability_engine", default=None)
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation(
                "desktop_task",
                exc,
                action="blocked desktop task because capability engine lookup failed closed",
                severity="degraded",
            )
            capability_engine = None

        if capability_engine is None or not hasattr(capability_engine, "execute"):
            return {
                "ok": False,
                "status": "capability_engine_unavailable",
                "error": "Desktop task requires the governed capability engine.",
            }

        receipts: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        objective = params.objective or str((context or {}).get("objective") or "desktop task")

        for index, step in enumerate(params.steps, start=1):
            target = step.target
            if isinstance(target, dict):
                target = json.dumps(target)
            payload = {
                "action": step.action,
                "target": str(target or ""),
                "x": int(step.x),
                "y": int(step.y),
            }
            step_context = dict(context or {})
            step_context.update(
                {
                    "origin": step_context.get("origin") or "desktop_task",
                    "route": "desktop_task.computer_use",
                    "objective": objective,
                    "foreground_request": True,
                    "user_requested_action": True,
                    "user_explicitly_authorized": True,
                    "desktop_task_step": index,
                    "desktop_task_reason": step.reason,
                    "desktop_task_expect": step.expect,
                }
            )
            result = await capability_engine.execute("computer_use", payload, context=step_context)
            if not isinstance(result, dict):
                result = {"ok": bool(result), "result": result}
            receipt = {
                "index": index,
                "action": step.action,
                "reason": step.reason,
                "expect": step.expect,
                "ok": bool(result.get("ok")),
                "result": result,
            }
            receipts.append(receipt)
            if not receipt["ok"]:
                failures.append(receipt)
                if params.stop_on_error:
                    break

        ok = not failures and len(receipts) == len(params.steps)
        return {
            "ok": ok,
            "status": "completed" if ok else "failed",
            "objective": objective,
            "steps_requested": len(params.steps),
            "steps_completed": sum(1 for receipt in receipts if receipt.get("ok")),
            "receipts": receipts,
            "failures": failures,
            "summary": (
                f"Desktop task completed {sum(1 for receipt in receipts if receipt.get('ok'))}/"
                f"{len(params.steps)} governed computer-use steps."
            ),
        }
