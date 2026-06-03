from __future__ import annotations

import json
import re
import time
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
            "get_clipboard, wait, run_applescript, write_text_file, render_text_pdf, "
            "move_file, create_folder"
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
            "create_folder",
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

    @staticmethod
    def _json_target(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _safe_filename(text: str, *, default: str = "aura_desktop_task") -> str:
        stem = re.sub(r"[^A-Za-z0-9._ -]+", "", str(text or "")).strip(" ._-")
        stem = re.sub(r"\s+", "_", stem).strip("_")
        return (stem or default)[:80]

    @staticmethod
    def _extract_folder_name(objective: str) -> str:
        text = str(objective or "")
        match = re.search(
            r"\b(?:folder|directory)\s+(?:named|called|titled)\s+(?:['\"]([^'\"]+)['\"]|([^.,;\n]+))",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return str(match.group(1) or match.group(2) or "").strip()[:100]
        if "journal" in text.lower():
            return "Aura's Journal"
        return f"Aura Desktop Task {int(time.time())}"

    @staticmethod
    def _extract_search_query(objective: str) -> str:
        text = str(objective or "").strip()
        patterns = (
            r"\bsearch\s+(?:for\s+)?([^.;\n]+)",
            r"\blook\s+up\s+([^.;\n]+)",
            r"\bgoogle\s+([^.;\n]+)",
            r"\bopen\s+(?:a\s+)?(?:browser\s+)?tab\s+(?:on\s+google\s+)?(?:for\s+)?([^.;\n]+)",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                query = match.group(1).strip(" ,")
                if query:
                    return query[:240]
        if "news" in text.lower():
            return text[:240]
        return ""

    @staticmethod
    def _extract_apps(objective: str) -> list[str]:
        text = str(objective or "").lower()
        apps: list[str] = []
        app_markers = {
            "notes": "Notes",
            "calculator": "Calculator",
            "finder": "Finder",
            "preview": "Preview",
            "safari": "Safari",
            "chrome": "Google Chrome",
            "browser": "Safari",
        }
        for marker, app in app_markers.items():
            if marker in text and app not in apps:
                if marker == "browser" and "chrome" in text:
                    continue
                apps.append(app)
        return apps[:4]

    @staticmethod
    def _document_body(objective: str, context: dict[str, Any] | None) -> str:
        context = context or {}
        for key in ("desktop_task_document_body", "draft_response", "cognitive_reply", "response"):
            value = str(context.get(key) or "").strip()
            if value:
                return value[:9000]
        stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        return (
            "Aura desktop task receipt\n\n"
            f"Timestamp: {stamp}\n"
            f"Objective: {str(objective or '').strip()}\n\n"
            "This document was created through Aura's governed desktop_task lane. "
            "It records the requested objective and the actions Aura attempted through her "
            "canonical computer-use gateway."
        )

    @classmethod
    def _steps_from_payload(cls, payload: Any) -> list[DesktopTaskStep]:
        if isinstance(payload, dict):
            payload = payload.get("steps")
        if not isinstance(payload, list):
            return []
        steps: list[DesktopTaskStep] = []
        for item in payload[:20]:
            try:
                steps.append(item if isinstance(item, DesktopTaskStep) else DesktopTaskStep(**dict(item)))
            except (TypeError, ValueError):
                continue
        return steps

    @classmethod
    def _steps_from_plan_text(cls, text: str) -> list[DesktopTaskStep]:
        source = str(text or "").strip()
        if not source:
            return []
        candidates: list[str] = []
        candidates.extend(
            match.group(1).strip()
            for match in re.finditer(r"```(?:json)?\s*(.*?)```", source, flags=re.IGNORECASE | re.DOTALL)
        )
        for open_char, close_char in (("{", "}"), ("[", "]")):
            start = source.find(open_char)
            end = source.rfind(close_char)
            if start >= 0 and end > start:
                candidates.append(source[start : end + 1])
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            steps = cls._steps_from_payload(parsed)
            if steps:
                return steps
        return []

    @classmethod
    def _steps_from_context(cls, context: dict[str, Any] | None) -> list[DesktopTaskStep]:
        context = context or {}
        for key in ("desktop_task_steps", "desktop_task_plan"):
            steps = cls._steps_from_payload(context.get(key))
            if steps:
                return steps
            steps = cls._steps_from_plan_text(str(context.get(key) or ""))
            if steps:
                return steps
        for key in ("cognitive_reply", "draft_response", "response"):
            steps = cls._steps_from_plan_text(str(context.get(key) or ""))
            if steps:
                return steps
        return []

    @staticmethod
    def _generic_open_app_mentions(objective: str) -> list[str]:
        text = str(objective or "")
        apps: list[str] = []
        patterns = (
            r"\bopen\s+(?:up\s+)?(?:my\s+|the\s+)?([A-Za-z][A-Za-z0-9 &._-]{1,60}?)\s+(?:app|application)\b",
            r"\blaunch\s+(?:my\s+|the\s+)?([A-Za-z][A-Za-z0-9 &._-]{1,60}?)\b",
        )
        stopwords = {"a", "an", "the", "my", "new"}
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                candidate = re.sub(r"\s+", " ", match.group(1)).strip(" ._-")
                if not candidate or candidate.lower() in stopwords:
                    continue
                if candidate.lower() == "notes":
                    candidate = "Notes"
                elif candidate.lower() == "chrome":
                    candidate = "Google Chrome"
                elif candidate.lower() == "browser":
                    candidate = "Safari"
                if candidate not in apps:
                    apps.append(candidate)
        return apps[:4]

    def _derive_steps_from_objective(
        self,
        objective: str,
        context: dict[str, Any] | None,
    ) -> list[DesktopTaskStep]:
        text = str(objective or "").strip()
        lowered = text.lower()
        steps: list[DesktopTaskStep] = []
        folder_name = self._extract_folder_name(text)
        folder_path = folder_name
        wants_folder = any(token in lowered for token in ("folder", "directory", "journal"))
        wants_document = any(
            token in lowered
            for token in ("write", "summary", "summarize", "note", "document", "pdf", "save", "journal")
        )
        wants_pdf = "pdf" in lowered or wants_document
        wants_search = any(token in lowered for token in ("search", "look up", "google", "news", "article"))

        if wants_folder or wants_document:
            steps.append(
                DesktopTaskStep(
                    action="create_folder",
                    target={"path": folder_path},
                    reason="Create the requested artifact folder inside an allowed desktop root.",
                    expect="Folder exists.",
                )
            )

        apps = self._extract_apps(text)
        for app in self._generic_open_app_mentions(text):
            if app not in apps:
                apps.append(app)

        for app in apps[:4]:
            steps.append(
                DesktopTaskStep(
                    action="open_app",
                    target=app,
                    reason=f"Open {app} because the objective names that app or surface.",
                    expect=f"{app} accepts focus or reports a launch error.",
                )
            )

        query = self._extract_search_query(text)
        if wants_search and query:
            steps.append(
                DesktopTaskStep(
                    action="open_url",
                    target=query,
                    reason="Open a browser/search tab for the requested live research topic.",
                    expect="Default browser accepts the search URL.",
                )
            )

        if wants_document:
            body = self._document_body(text, context)
            filename_stem = self._safe_filename("aura_journal_entry" if "journal" in lowered else "aura_desktop_summary")
            text_path = f"{folder_path}/{filename_stem}.txt"
            steps.append(
                DesktopTaskStep(
                    action="write_text_file",
                    target={
                        "path": text_path,
                        "content": body,
                        "overwrite": False,
                    },
                    reason="Write a durable text artifact before PDF rendering.",
                    expect="Text artifact exists with the composed body.",
                )
            )
            if wants_pdf:
                steps.append(
                    DesktopTaskStep(
                        action="render_text_pdf",
                        target={
                            "path": f"{folder_path}/{filename_stem}.pdf",
                            "title": "Aura Desktop Task",
                            "body": body,
                            "overwrite": False,
                        },
                        reason="Render the same verified text body into a PDF artifact.",
                        expect="PDF artifact exists and starts with a PDF header.",
                    )
                )

        if not steps:
            steps.append(
                DesktopTaskStep(
                    action="read_screen_text",
                    target="",
                    reason="Observe the current desktop before attempting an underspecified action.",
                    expect="Foreground screen text or an explicit permission failure is returned.",
                )
            )
        return steps[:20]

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

        steps = list(params.steps)
        if not steps:
            steps = self._steps_from_context(context)
        if not steps:
            steps = self._derive_steps_from_objective(objective, context)

        for index, step in enumerate(steps, start=1):
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

        ok = not failures and len(receipts) == len(steps)
        return {
            "ok": ok,
            "status": "completed" if ok else "failed",
            "objective": objective,
            "steps_requested": len(steps),
            "steps_completed": sum(1 for receipt in receipts if receipt.get("ok")),
            "receipts": receipts,
            "failures": failures,
            "summary": (
                f"Desktop task completed {sum(1 for receipt in receipts if receipt.get('ok'))}/"
                f"{len(steps)} governed computer-use steps."
            ),
        }
