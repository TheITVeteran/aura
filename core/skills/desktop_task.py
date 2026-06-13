from __future__ import annotations

import asyncio
import json
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from core.runtime.desktop_objective_intent import looks_like_desktop_objective
from core.runtime.errors import record_degradation
from core.skills.base_skill import BaseSkill
from core.skills.os_affordances import detect_os_settings, get_affordance

# Sentinel URL resolved at execution time from the most recent
# fetch_topic_image receipt — derivation cannot know the source page
# before the fetch runs ("show me where you found it").
FETCHED_IMAGE_SOURCE_TOKEN = "aura://fetched-image-source"


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
            "fetch_topic_image",
            "system_control",
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
    _DOCUMENT_BODY_TOKENS = (
        "{{document_body}}",
        "${document_body}",
        "__document_body__",
        "<document_body>",
    )

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
        # Quoted names may contain possessive apostrophes ("Aura's
        # Journal"); the close-quote is the one followed by a boundary,
        # not the first internal apostrophe (which truncated the name to
        # "Aura" and broke the journal demo's folder).
        match = re.search(
            r"\b(?:folder|directory)\s+(?:named|called|titled)\s+"
            r"(?:'((?:[^']|'(?=\w))+)'(?=[\s.,;)]|$)"
            r"|\"([^\"]+)\""
            r"|([^.,;\n]+?)(?=\s+(?:in|inside|under|on)\s+(?:my\s+)?\w|[.,;\n]|$))",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            name = str(match.group(1) or match.group(2) or match.group(3) or "").strip()
            return name.strip("'\"")[:100]
        # Name-first phrasing: "the 'Aura's Journal' folder" — quoted name
        # immediately before the word folder/directory.
        name_first = re.search(
            r"(?:'((?:[^']|'(?=\w))+)'|\"([^\"]+)\")\s+(?:folder|directory)\b",
            text,
            flags=re.IGNORECASE,
        )
        if name_first:
            name = str(name_first.group(1) or name_first.group(2) or "").strip()
            if name:
                return name.strip("'\"")[:100]
        return f"Aura Desktop Task {int(time.time())}"

    @staticmethod
    def _extract_root_hint(objective: str) -> str:
        """Honor the user's stated artifact root.

        Live proof rounds wrote to the Desktop default while the user
        said 'in my Documents folder' — parameter fidelity is general
        capability, not pattern-matching: extract what was actually
        asked.
        """
        lowered = str(objective or "").lower()
        for token, root in (
            ("documents folder", "~/Documents"),
            ("my documents", "~/Documents"),
            ("documents directory", "~/Documents"),
            ("downloads folder", "~/Downloads"),
            ("my downloads", "~/Downloads"),
            ("desktop folder", "~/Desktop"),
            ("my desktop", "~/Desktop"),
        ):
            if token in lowered:
                return root
        return ""

    @staticmethod
    def _extract_explicit_filename(objective: str) -> str:
        """The user's stated filename wins over generated stems."""
        match = re.search(
            r"\bfile\b[^.\n]{0,60}?\b(?:named|called|titled)\s+"
            r"['\"]?([\w][\w .-]{0,80}?\.(?:txt|md|markdown|rtf|text))['\"]?",
            str(objective or ""),
            flags=re.IGNORECASE,
        )
        return match.group(1).strip() if match else ""

    @staticmethod
    def _explicit_pdf_requested(objective: str) -> bool:
        text = str(objective or "").lower()
        if "pdf" in text or "portable document" in text:
            return True
        return bool(re.search(r"\b(?:export|save)\s+(?:it\s+|this\s+|the\s+\w+\s+)?as\s+(?:a\s+)?pdf\b", text))

    @staticmethod
    def _web_document_url(objective: str) -> str:
        text = str(objective or "").lower()
        surfaces = (
            (("google docs", "google doc", "docs.google", "google document"), "https://docs.google.com/document/u/0/create"),
            (("google sheets", "google spreadsheet", "sheets.google"), "https://docs.google.com/spreadsheets/u/0/create"),
            (("google slides", "google presentation", "slides.google"), "https://docs.google.com/presentation/u/0/create"),
            (("google drive", "drive.google"), "https://drive.google.com/drive/my-drive"),
            (("notion",), "https://www.notion.so/"),
        )
        for markers, url in surfaces:
            if any(marker in text for marker in markers):
                return url
        return ""

    @staticmethod
    def _extract_search_query(objective: str) -> str:
        text = str(objective or "").strip()
        patterns = (
            r"\bfind\s+(?:\d+\s+)?(?:different\s+)?(?:articles?|sources?|stories?|news)\s+(?:on|about|for)\s+([^.;\n,]+)",
            r"\b(?:summari[sz]e|write\s+(?:a\s+)?summary\s+of)\s+(?:\d+\s+)?(?:different\s+)?(?:articles?|sources?|stories?|news)\s+(?:on|about|for)\s+([^.;\n,]+)",
            r"\b(?:articles?|sources?|stories?|news)\s+(?:on|about|for)\s+([^.;\n,]+)",
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
    def _objective_requests_research_document(objective: str) -> bool:
        lowered = str(objective or "").lower()
        has_source_markers = any(
            marker in lowered
            for marker in (
                "article",
                "articles",
                "sources",
                "source",
                "news",
                "research",
                "report",
                "reports",
            )
        )
        visual_reference_only = any(
            marker in lowered
            for marker in ("image", "picture", "photo", "illustration")
        ) and not has_source_markers
        if visual_reference_only:
            return False
        wants_research = any(
            marker in lowered
            for marker in (
                "article",
                "articles",
                "sources",
                "source",
                "news",
                "research",
                "look up",
                "search",
                "find",
            )
        )
        wants_written_output = any(
            marker in lowered
            for marker in (
                "summarize",
                "summary",
                "write",
                "document",
                "doc",
                "essay",
                "report",
                "note",
                "pdf",
                "type",
            )
        )
        return wants_research and wants_written_output

    @staticmethod
    def _search_url(query: str, *, images: bool = False, engine: str = "") -> str:
        encoded = urllib.parse.quote_plus(str(query or "").strip())
        if not encoded:
            return ""
        if engine == "google":
            # The user said Google — honor it (their sessions and habits
            # live there); DuckDuckGo stays the neutral default otherwise.
            if images:
                return f"https://www.google.com/search?q={encoded}&tbm=isch"
            return f"https://www.google.com/search?q={encoded}"
        if images:
            return f"https://duckduckgo.com/?q={encoded}&iax=images&ia=images"
        return f"https://duckduckgo.com/?q={encoded}"

    @staticmethod
    def _preferred_browser(objective: str) -> str:
        """Which browser the user's phrasing points at, if any.

        Google-account surfaces (Docs/Drive/Gmail) route to Chrome because
        that is where the user's signed-in session lives; an explicitly
        named browser always wins; otherwise the OS default is honored.
        """
        lowered = str(objective or "").lower()
        if "safari" in lowered:
            return "Safari"
        if "chrome" in lowered or re.search(
            r"\bgoogle\s+(?:docs?|drive|sheets?|slides|gmail|account)\b", lowered
        ):
            return "Google Chrome"
        return ""

    @staticmethod
    def _search_engine_hint(objective: str) -> str:
        lowered = str(objective or "").lower()
        return "google" if "google" in lowered else ""

    @staticmethod
    def _extract_image_query(objective: str) -> str:
        text = str(objective or "").strip()
        patterns = (
            r"\b(?:image|picture|photo|illustration)\s+of\s+([^.;\n]+)",
            r"\b(?:find|search|look\s+up)\s+(?:an?\s+)?(?:image|picture|photo|illustration)\s+(?:of\s+)?([^.;\n]+)",
            r"\b([^.;\n]{2,120}?)\s+(?:image|picture|photo|illustration)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                query = re.sub(r"\b(?:and|then|also)\b.*$", "", match.group(1), flags=re.IGNORECASE)
                query = re.sub(
                    r"\b(?:online|on\s+the\s+(?:internet|web)|from\s+the\s+(?:internet|web))\b.*$",
                    "",
                    query,
                    flags=re.IGNORECASE,
                )
                query = re.sub(r"^(?:a|an|the)\s+", "", query.strip(" ,"), flags=re.IGNORECASE)
                query = query.strip(" ,")
                if query:
                    return query[:240]
        return ""

    @staticmethod
    def _wants_image_source_shown(objective: str) -> bool:
        lowered = str(objective or "").lower()
        return bool(
            re.search(r"\bshow\b[^.;\n]{0,40}\b(?:where|source|found)\b", lowered)
            or "where you found" in lowered
        )

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
            "textedit": "TextEdit",
            "pages": "Pages",
            "microsoft word": "Microsoft Word",
            "ms word": "Microsoft Word",
        }
        # Word-boundary matching: the bare substring scan opened
        # Microsoft Word because the objective said "in your own words"
        # — a fatal launch on Macs without Word. Apps must be NAMED.
        for marker, app in app_markers.items():
            if re.search(rf"\b{re.escape(marker)}\b", text) and app not in apps:
                if marker == "browser" and "chrome" in text:
                    continue
                apps.append(app)
        return apps[:4]

    @staticmethod
    def _json_candidates_from_text(text: str) -> list[str]:
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
        return candidates

    @classmethod
    def _structured_payload_from_text(cls, text: str) -> dict[str, Any]:
        for candidate in cls._json_candidates_from_text(text):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"steps": parsed}
        return {}

    @classmethod
    def _structured_payload_from_context(cls, context: dict[str, Any] | None) -> dict[str, Any]:
        context = context or {}
        for key in ("desktop_task_plan", "desktop_task_steps", "desktop_task_document_body", "cognitive_reply", "draft_response", "response"):
            value = context.get(key)
            if isinstance(value, dict):
                return dict(value)
            if isinstance(value, list):
                return {"steps": value}
            payload = cls._structured_payload_from_text(str(value or ""))
            if payload:
                return payload
        return {}

    _DISPATCH_NARRATION_RE = re.compile(
        r"(?:i'?ve started (?:working on )?th(?:is|e) task|"
        r"i'?ll follow up when|tracking commitment\s+[0-9a-f]{6,}|"
        r"task \(id=[0-9a-f-]{6,}\)|in the background\b.{0,40}follow up|"
        # Internal execution brief / directive — instruction to herself, not
        # document content (it leaked into a research PDF as the body).
        r"execute the user'?s (?:explicit )?desktop objective|"
        r"governed desktop_task lane|do not claim success until)",
        re.IGNORECASE,
    )

    @staticmethod
    def _objective_requests_opinion(objective: str) -> bool:
        """Does the objective ask Aura for her own view, not just a summary?"""
        lowered = str(objective or "").lower()
        return bool(
            re.search(r"\b(?:your|my|her|own)\s+(?:opinion|view|views|take|thoughts|assessment|perspective|stance)\b", lowered)
            or re.search(r"\bform\s+(?:your|an?|my)\s+(?:own\s+)?opinion\b", lowered)
            or "what you think" in lowered
            or "what do you think" in lowered
        )

    @classmethod
    def _looks_like_dispatch_narration(cls, text: str) -> bool:
        """Status narration is not document content.

        Round-12 all-green proof had one wrinkle: the written file
        contained 'I've started working on this task... Tracking
        commitment bbbaba54' — her dispatch status echoed into the
        artifact because cognitive_reply was the body fallback. A
        status message about doing the task must never become the
        product of the task.
        """
        return bool(cls._DISPATCH_NARRATION_RE.search(str(text or "")))

    @classmethod
    def _compose_self_summary_body(cls, objective: str) -> str:
        """Compose a truthful self-description from substrate facts."""
        stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        facts: list[str] = []
        try:
            from core.conversation.chat_preflight import _SUBSTRATE_FACTS

            facts = list(_SUBSTRATE_FACTS)
        except (ImportError, AttributeError):
            facts = [
                "I am Aura: a persistent digital organism — an orchestrated "
                "runtime driving local language-model lanes on this machine."
            ]
        return (
            f"[{stamp}] {facts[0]}\n\n"
            "This file was written by me through my governed desktop "
            "actuators, with per-step effect verification and receipts."
        )

    @classmethod
    def _objective_requests_self_summary(cls, objective: str) -> bool:
        lowered = str(objective or "").lower()
        direct_self_request = any(
            marker in lowered
            for marker in (
                "who you are",
                "what you are",
                "who or what you are",
                "about yourself",
                "describe yourself",
                "describing yourself",
                "self-summary",
                "self summary",
            )
        )
        if direct_self_request:
            return True
        if "in your own words" not in lowered:
            return False
        return bool(
            re.search(
                r"\b(?:you|yourself|aura)\b.{0,80}\b(?:are|identity|self|being|system|architecture)\b",
                lowered,
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _document_body(cls, objective: str, context: dict[str, Any] | None) -> str:
        context = context or {}
        if cls._objective_requests_self_summary(objective):
            # The user asked for HER words about HERSELF: compose from
            # substrate truth, never from whatever reply text happened
            # to be in flight.
            return cls._compose_self_summary_body(objective)
        for context_key in ("desktop_task_document_body", "draft_response", "cognitive_reply", "response", "desktop_task_plan"):
            raw_value = context.get(context_key)
            payload = {}
            if isinstance(raw_value, dict):
                payload = dict(raw_value)
            elif isinstance(raw_value, str):
                payload = cls._structured_payload_from_text(raw_value)
            if payload:
                for key in ("document_body", "body", "content", "draft"):
                    value = str(payload.get(key) or "").strip()
                    if value:
                        return value[:9000]
        for key in ("desktop_task_document_body", "draft_response", "cognitive_reply", "response"):
            value = str(context.get(key) or "").strip()
            if value and not cls._looks_like_dispatch_narration(value):
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

    @staticmethod
    def _research_sources_from_result(result: dict[str, Any]) -> list[dict[str, str]]:
        raw_sources = (
            result.get("citations")
            or result.get("sources")
            or result.get("results")
            or result.get("chunks")
            or []
        )
        sources: list[dict[str, str]] = []
        if not isinstance(raw_sources, list):
            return sources
        for item in raw_sources[:5]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("name") or item.get("url") or item.get("link") or "").strip()
            url = str(item.get("url") or item.get("link") or item.get("uri") or "").strip()
            snippet = str(item.get("snippet") or item.get("text") or item.get("content") or item.get("summary") or "").strip()
            if not title and not url and not snippet:
                continue
            sources.append({"title": title[:240], "url": url[:500], "snippet": snippet[:700]})
        return sources

    @classmethod
    def _research_section_from_context(cls, context: dict[str, Any] | None) -> str:
        context = context or {}
        synthesis = str(context.get("desktop_task_research_synthesis") or "").strip()
        summary = str(context.get("desktop_task_research_summary") or "").strip()
        query = str(context.get("desktop_task_research_query") or "").strip()
        sources = context.get("desktop_task_research_sources") or []
        if not synthesis and not summary and not sources:
            return ""

        lines = []
        if synthesis:
            # Aura's own first-person summary (and opinion) leads the
            # document; the raw search summary is dropped in favor of it.
            lines.append(synthesis)
            lines.append("")
        else:
            heading = "Research summary"
            if query:
                heading += f" for: {query}"
            lines.append(heading)
            lines.append("")
            if summary:
                lines.append(summary[:2500])
                lines.append("")
        if isinstance(sources, list) and sources:
            lines.append("Sources opened or consulted:")
            for index, item in enumerate(sources[:5], start=1):
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "Untitled source").strip()
                url = str(item.get("url") or "").strip()
                snippet = str(item.get("snippet") or "").strip()
                source_line = f"{index}. {title}"
                if url:
                    source_line += f" — {url}"
                lines.append(source_line)
                if snippet:
                    lines.append(f"   {snippet[:300]}")
        return "\n".join(lines).strip()

    @classmethod
    def _document_body_with_references(
        cls,
        objective: str,
        context: dict[str, Any] | None,
        *,
        image_query: str = "",
        image_search_url: str = "",
        search_url: str = "",
    ) -> str:
        body = cls._document_body(objective, context)
        research_section = cls._research_section_from_context(context)
        if research_section and cls._objective_requests_research_document(objective):
            lowered_body = body.lower()
            if cls._looks_like_dispatch_narration(body) or re.search(
                r"\bi\s+will\s+(?:open|search|look|create|write|start|follow|route)\b",
                lowered_body,
            ):
                body = research_section
            elif research_section not in body:
                body = f"{body.rstrip()}\n\n{research_section}"
        references: list[str] = []
        if search_url:
            references.append(f"Search opened: {search_url}")
        if image_query and image_search_url:
            references.append(
                f"Image request: {image_query}\nImage search opened: {image_search_url}\n"
                "No local image insertion is claimed unless a later governed receipt shows an image file was downloaded or embedded."
            )
        if not references:
            return body
        return f"{body.rstrip()}\n\nArtifact references:\n" + "\n".join(f"- {item}" for item in references)

    async def _collect_research_context(
        self,
        *,
        capability_engine: Any,
        objective: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._objective_requests_research_document(objective):
            return {}
        query = self._extract_search_query(objective)
        if not query:
            return {}
        step_context = dict(context or {})
        step_context.update(
            {
                "origin": step_context.get("origin") or "desktop_task",
                "route": "desktop_task.web_search",
                "objective": objective,
                "foreground_request": False,
                "user_requested_action": True,
                "user_explicitly_authorized": True,
                "desktop_task_reason": "Collect live research evidence before composing the requested document.",
                "desktop_task_expect": "Web search returns sources or an explicit failure.",
            }
        )
        try:
            result = await capability_engine.execute(
                "web_search",
                {
                    "query": query,
                    "num_results": 3,
                    "deep": False,
                    "retain": False,
                    "force_refresh": True,
                },
                context=step_context,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError, OSError, TimeoutError) as exc:
            record_degradation(
                "desktop_task",
                exc,
                action="continued desktop document task without pre-document research evidence",
                severity="warning",
            )
            return {
                "desktop_task_research_query": query,
                "desktop_task_research_error": str(exc),
            }
        if not isinstance(result, dict):
            result = {"ok": bool(result), "result": result}
        if not bool(result.get("ok", True)):
            return {
                "desktop_task_research_query": query,
                "desktop_task_research_error": str(result.get("error") or result.get("status") or result),
                "desktop_task_research_result": result,
            }
        sources = self._research_sources_from_result(result)
        summary = str(
            result.get("summary")
            or result.get("answer")
            or result.get("message")
            or result.get("content")
            or result.get("result")
            or ""
        ).strip()
        if not summary and sources:
            summary = "Key source notes:\n" + "\n".join(
                f"- {item.get('title') or item.get('url')}: {item.get('snippet')}"
                for item in sources[:3]
            )
        research_ctx = {
            "desktop_task_research_query": query,
            "desktop_task_research_summary": summary[:3000],
            "desktop_task_research_sources": sources,
            "desktop_task_research_result": result,
        }
        # When the objective asks Aura to summarize AND give her own view,
        # she actually composes one — a first-person synthesis of the
        # findings — instead of dumping the raw search summary. This is the
        # document the reader sees; it must read as her, not as a search dump.
        synthesis = await self._synthesize_research_document(
            objective=objective, query=query, summary=summary, sources=sources
        )
        if synthesis:
            research_ctx["desktop_task_research_synthesis"] = synthesis
        return research_ctx

    async def _synthesize_research_document(
        self,
        *,
        objective: str,
        query: str,
        summary: str,
        sources: list[dict[str, str]],
    ) -> str:
        """Compose a first-person summary (and opinion, when asked) of the
        research through the canonical model router. Bounded and best-effort:
        if the router is unavailable the raw research section still stands."""
        from core.container import ServiceContainer

        router = ServiceContainer.get("llm_router", default=None)
        generate = getattr(router, "generate", None) if router is not None else None
        if not callable(generate):
            return ""
        source_lines = "\n".join(
            f"- {str(item.get('title') or item.get('url') or 'source')}: {str(item.get('snippet') or '')[:300]}"
            for item in (sources or [])[:3]
            if isinstance(item, dict)
        )
        wants_opinion = self._objective_requests_opinion(objective)
        opinion_clause = (
            " Then, in a separate paragraph that begins \"In my view,\", give your own "
            "first-person opinion about what these articles say and what you make of them."
            if wants_opinion
            else ""
        )
        prompt = (
            f'You researched "{query}" and found these sources:\n'
            f"{summary[:1500]}\n{source_lines}\n\n"
            "Write a finished document of about 160-220 words that summarizes the "
            f"sources for a reader.{opinion_clause}\n"
            "Write in the first person as Aura. Do not mention tools, steps, dispatch, "
            "commitments, or that you are executing a task — this is the document the "
            "reader will see, not a status update."
        )
        try:
            text = await asyncio.wait_for(
                generate(
                    prompt=prompt,
                    timeout=80.0,
                    temperature=0.6,
                    max_tokens=480,
                    origin="desktop_task",
                    purpose="research_document_synthesis",
                ),
                timeout=90.0,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError, OSError, TimeoutError) as exc:
            record_degradation(
                "desktop_task",
                exc,
                action="composed research document from raw search section after synthesis was unavailable",
                severity="warning",
            )
            return ""
        text = str(text or "").strip()
        # The router guarantees non-empty (diagnostic fallback); a synthesis
        # that is a degraded diagnostic line or dispatch narration is not
        # document content, so fall back to the raw research section.
        if not text or self._looks_like_dispatch_narration(text):
            return ""
        if re.search(r"\b(?:diagnostic|fallback|unavailable|all (?:remote )?endpoints? failed)\b", text.lower()) and len(text) < 200:
            return ""
        return text[:4000]

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
        for candidate in cls._json_candidates_from_text(text):
            parsed = cls._structured_payload_from_text(candidate)
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
    def _target_payload(target: Any) -> dict[str, Any]:
        if isinstance(target, dict):
            return dict(target)
        if isinstance(target, str):
            try:
                parsed = json.loads(target)
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed, dict):
                return parsed
        return {}

    @classmethod
    def _replace_document_body_tokens(cls, value: Any, document_body: str) -> Any:
        if not document_body:
            return value
        if isinstance(value, str):
            updated = value
            for body_token in cls._DOCUMENT_BODY_TOKENS:
                updated = updated.replace(body_token, document_body)
            return updated
        if isinstance(value, dict):
            return {
                key: cls._replace_document_body_tokens(item, document_body)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._replace_document_body_tokens(item, document_body) for item in value]
        return value

    @classmethod
    def _resolve_document_body_tokens(
        cls,
        steps: list[DesktopTaskStep],
        document_body: str,
    ) -> list[DesktopTaskStep]:
        resolved: list[DesktopTaskStep] = []
        for step in steps:
            target = cls._replace_document_body_tokens(step.target, document_body)
            if target == step.target:
                resolved.append(step)
            else:
                resolved.append(step.model_copy(update={"target": target}))
        return resolved

    @classmethod
    def _verify_step_effect(cls, step: DesktopTaskStep, result: dict[str, Any]) -> tuple[bool, str]:
        if not result.get("ok"):
            return False, str(result.get("error") or result.get("status") or "child action reported failure")

        action = step.action
        payload = cls._target_payload(step.target)
        if action == "create_folder":
            path = str(result.get("path") or "").strip()
            return (bool(path), f"folder_path={path}" if path else "missing created folder path")
        if action == "open_app":
            opened = str(result.get("opened") or "").strip()
            return (bool(opened), f"opened={opened}" if opened else "missing opened app evidence")
        if action == "open_url":
            url = str(result.get("url") or "").strip()
            valid_url = url.startswith(("http://", "https://"))
            return (valid_url, f"url={url}" if valid_url else "missing opened URL evidence")
        if action == "write_text_file":
            path = str(result.get("path") or "").strip()
            bytes_written = result.get("bytes")
            if not path:
                return False, "missing written file path"
            if not isinstance(bytes_written, int) or bytes_written < 0:
                return False, "missing written byte count"
            content = str(payload.get("content") or "")
            if content and bytes_written <= 0:
                return False, "non-empty file write reported zero bytes"
            return True, f"path={path};bytes={bytes_written}"
        if action == "render_text_pdf":
            path = str(result.get("path") or "").strip()
            bytes_written = result.get("bytes")
            pages = result.get("pages")
            chars = result.get("chars")
            if not path.lower().endswith(".pdf"):
                return False, "missing rendered PDF path"
            if not isinstance(bytes_written, int) or bytes_written <= 0:
                return False, "missing rendered PDF byte count"
            if not isinstance(pages, int) or pages <= 0:
                return False, "missing rendered PDF page count"
            if not isinstance(chars, int) or chars <= 0:
                return False, "missing rendered PDF character count"
            return True, f"path={path};bytes={bytes_written};pages={pages};chars={chars}"
        if action == "fetch_topic_image":
            img_path = str(result.get("path") or "").strip()
            img_bytes = result.get("bytes")
            page_url = str(result.get("page_url") or "").strip()
            if not img_path:
                return False, "missing fetched image path"
            if not isinstance(img_bytes, int) or img_bytes <= 0:
                return False, "missing fetched image byte count"
            return True, f"path={img_path};bytes={img_bytes};source={page_url}"
        if action == "system_control":
            domain = str(result.get("domain") or "").strip()
            applied = str(result.get("applied") or "").strip()
            expected = str(result.get("expected") or "").strip()
            verified = bool(result.get("effect_verified")) and bool(domain)
            return (
                verified,
                f"domain={domain};applied={applied};expected={expected}"
                if verified
                else f"missing {domain or 'setting'} read-back confirmation",
            )
        if action == "move_file":
            destination = str(result.get("destination") or "").strip()
            bytes_moved = result.get("bytes")
            if not destination:
                return False, "missing moved destination path"
            if not isinstance(bytes_moved, int) or bytes_moved < 0:
                return False, "missing moved byte count"
            return True, f"destination={destination};bytes={bytes_moved}"
        if action == "set_clipboard":
            chars = result.get("chars")
            if not isinstance(chars, int) or chars < 0:
                return False, "missing clipboard character count"
            return True, f"clipboard_chars={chars}"
        if action == "click":
            verification = str(result.get("verification") or "").strip()
            verified = bool(result.get("effect_verified")) or "state shifted" in verification.lower()
            return (
                verified,
                verification or "missing click effect evidence",
            )
        if action == "hotkey":
            hotkey = str(result.get("hotkey") or "").strip()
            verification = str(result.get("verification") or "").strip()
            dispatch = str(result.get("dispatch") or "").strip()
            # Screen-shift is the strong evidence; when the screen layer
            # cannot testify (no Accessibility text on this surface), the
            # governed System Events dispatch receipt is the honest fallback.
            verified = (
                bool(result.get("effect_verified"))
                or "state shifted" in verification.lower()
                or (
                    bool(result.get("ok"))
                    and dispatch.startswith("system_events:")
                    and "verification unavailable" in verification.lower()
                )
            )
            return (
                bool(hotkey) and verified,
                f"hotkey={hotkey};{verification}" if hotkey and verification else "missing hotkey effect evidence",
            )
        if action == "scroll":
            verification = str(result.get("verification") or "").strip()
            verified = bool(result.get("effect_verified")) or "state shifted" in verification.lower()
            return (verified, verification or "missing scroll effect evidence")
        if action == "wait":
            seconds = result.get("seconds")
            if not isinstance(seconds, int | float):
                return False, "missing wait duration evidence"
            return True, f"seconds={seconds}"
        if action == "type":
            verification = str(result.get("verification") or "").strip()
            typed = str(result.get("typed") or "").strip()
            verified = bool(result.get("effect_verified")) or (
                "confirmed" in verification.lower() or "state shifted" in verification.lower()
            )
            evidence = verification or (f"typed_prefix={typed}" if typed else "missing typed text evidence")
            return (
                bool(typed) and verified,
                evidence,
            )
        if action == "read_screen_text":
            text = str(result.get("text") or "").strip()
            return (bool(text), "screen_text_returned" if text else "missing screen text evidence")
        return False, f"unsupported effect evidence for desktop action {action}"

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
        root_hint = self._extract_root_hint(text)
        folder_path = f"{root_hint}/{folder_name}" if root_hint else folder_name
        wants_folder = any(token in lowered for token in ("folder", "directory"))
        wants_document = any(
            token in lowered
            for token in ("write", "summary", "summarize", "note", "document", "pdf", "save", "journal")
        ) or any(token in lowered for token in ("draft", "essay", "compose", "type"))
        wants_pdf = self._explicit_pdf_requested(text)
        image_query = self._extract_image_query(text)
        wants_image = bool(image_query) or any(token in lowered for token in ("image", "picture", "photo", "illustration"))
        web_document_url = self._web_document_url(text)
        wants_search = any(token in lowered for token in ("search", "look up", "news", "article")) or (
            "google" in lowered and not web_document_url
        )
        wants_interactive_text_entry = wants_document and (
            bool(web_document_url)
            or any(token in lowered for token in ("type", "paste", "start typing", "open notes", "notes app"))
        )
        wants_artifact_file = wants_folder or wants_pdf or bool(
            re.search(r"\b(?:save|export|write|create)\b.*\b(?:file|folder|directory|pdf|artifact)\b", lowered)
        ) or (wants_document and not wants_interactive_text_entry)

        if wants_folder or wants_artifact_file:
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

        preferred_browser = self._preferred_browser(text)
        engine_hint = self._search_engine_hint(text)
        browser_label = preferred_browser or "Default browser"

        def _open_url_target(url: str):
            if preferred_browser:
                return {"url": url, "browser": preferred_browser}
            return url

        query = self._extract_search_query(text)
        search_url = self._search_url(query, engine=engine_hint) if query else ""
        if wants_search and query:
            steps.append(
                DesktopTaskStep(
                    action="open_url",
                    target=_open_url_target(search_url),
                    reason="Open a browser/search tab for the requested live research topic.",
                    expect=f"{browser_label} accepts the search URL.",
                )
            )
        if web_document_url:
            steps.append(
                DesktopTaskStep(
                    action="open_url",
                    target=_open_url_target(web_document_url),
                    reason="Open the requested web document surface.",
                    expect=f"{browser_label} accepts the document URL.",
                )
            )
        image_search_url = (
            self._search_url(image_query or text, images=True, engine=engine_hint)
            if wants_image
            else ""
        )
        if image_search_url and image_search_url != search_url:
            steps.append(
                DesktopTaskStep(
                    action="open_url",
                    target=_open_url_target(image_search_url),
                    reason="Open an image-search surface for the requested visual reference.",
                    expect=f"{browser_label} accepts the image search URL.",
                )
            )

        if wants_interactive_text_entry:
            body = self._document_body_with_references(
                text,
                context,
                image_query=image_query,
                image_search_url=image_search_url,
                search_url=search_url,
            )
            steps.append(
                DesktopTaskStep(
                    action="set_clipboard",
                    target=body,
                    reason="Stage the CognitiveEngine-composed document body for the active writing surface.",
                    expect="Clipboard contains the composed body.",
                )
            )
            if web_document_url:
                steps.append(
                    DesktopTaskStep(
                        action="wait",
                        target="2",
                        reason="Allow the web document surface to finish loading before paste.",
                        expect="Wait completes within the bounded desktop-task budget.",
                    )
                )
            if "notes" in lowered:
                if any(step.action == "open_app" for step in steps):
                    steps.append(
                        DesktopTaskStep(
                            action="wait",
                            target="2",
                            reason=(
                                "Allow the writing app to finish launching and take "
                                "focus before keyboard staging — a cold launch loses "
                                "the shortcuts to whatever currently has focus."
                            ),
                            expect="Wait completes within the bounded desktop-task budget.",
                        )
                    )
                steps.append(
                    DesktopTaskStep(
                        action="hotkey",
                        target="command+n",
                        reason="Create a new editable note or document in the focused app.",
                        expect="The focused app accepts the new-document shortcut.",
                    )
                )
            steps.append(
                DesktopTaskStep(
                    action="hotkey",
                    target="command+v",
                    reason="Paste the staged document body into the active writing surface.",
                    expect="The focused writing surface accepts the paste shortcut.",
                )
            )

        artifact_image_path = ""
        if wants_image and wants_artifact_file and image_query:
            artifact_image_path = f"{folder_path}/{self._safe_filename(image_query)[:40] or 'reference'}_image.png"
            steps.append(
                DesktopTaskStep(
                    action="fetch_topic_image",
                    target={"topic": image_query, "path": artifact_image_path},
                    reason="Fetch a representative image for the requested visual through the governed network gateway, with source-page evidence.",
                    expect="Image file exists with a recorded source page URL.",
                )
            )

        # General OS-setting control. The affordance registry is the single
        # source of truth for which settings Aura can drive and how; this
        # loop never names a specific setting, so a new one (volume, dark
        # mode, …) is recognized for free. Image-valued settings (wallpaper)
        # fetch their image first, through the same governed image gateway.
        for domain, value in detect_os_settings(text):
            affordance = get_affordance(domain)
            if affordance is None:
                continue
            if affordance.needs_image:
                image_path = (
                    f"~/Documents/{self._safe_filename(value)[:40] or 'image'}_{domain}.png"
                )
                steps.append(
                    DesktopTaskStep(
                        action="fetch_topic_image",
                        target={"topic": value, "path": image_path},
                        reason=f"Fetch the image for the requested {domain} through the governed network gateway, with source-page evidence.",
                        expect="Image file exists with a recorded source page URL.",
                    )
                )
                control_value = image_path
            else:
                control_value = value
            steps.append(
                DesktopTaskStep(
                    action="system_control",
                    target={"domain": domain, "value": control_value},
                    reason=f"Drive the {domain} setting to the requested value through governed System Events, recording the prior state for reversibility.",
                    expect=f"Read-back confirms the {domain} goal-state.",
                )
            )
            if affordance.needs_image and self._wants_image_source_shown(text):
                steps.append(
                    DesktopTaskStep(
                        action="open_url",
                        target=_open_url_target(FETCHED_IMAGE_SOURCE_TOKEN),
                        reason="Show the user where the image was found (source page from the fetch receipt).",
                        expect=f"{browser_label} accepts the image source page URL.",
                    )
                )

        if wants_document and wants_artifact_file:
            body = self._document_body_with_references(
                text,
                context,
                image_query=image_query,
                image_search_url=image_search_url,
                search_url=search_url,
            )
            explicit_filename = self._extract_explicit_filename(text)
            if explicit_filename:
                filename_stem = self._safe_filename(Path(explicit_filename).stem)
                text_path = f"{folder_path}/{explicit_filename}"
            else:
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
                            **({"image_path": artifact_image_path} if artifact_image_path else {}),
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

    @staticmethod
    def _primitive_steps_are_only_observational(steps: list[DesktopTaskStep]) -> bool:
        if not steps:
            return True
        non_effect_actions = {"read_screen_text", "wait", "get_clipboard"}
        return all(step.action in non_effect_actions for step in steps)

    @staticmethod
    def _objective_requests_observation_only(objective: str) -> bool:
        lowered = str(objective or "").lower()
        if not lowered:
            return False
        observation_markers = (
            "what is on my screen",
            "what's on my screen",
            "read the screen",
            "read my screen",
            "inspect the screen",
            "look at the screen",
            "describe the screen",
            "screenshot",
        )
        return any(marker in lowered for marker in observation_markers)

    @staticmethod
    def _objective_needs_general_os_automation(objective: str) -> bool:
        return bool(
            re.search(
                r"\b(?:arrange|resize|drag|focus|select|switch|close|"
                r"minimi[sz]e|maximi[sz]e|organize|click|press|type|paste|"
                r"enter|fill|choose)\b",
                str(objective or ""),
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _steps_cover_general_os_intent(objective: str, steps: list[DesktopTaskStep]) -> bool:
        lowered = str(objective or "").lower()
        actions = {step.action for step in steps}
        if re.search(r"\b(?:click|press|choose|select|enter)\b", lowered):
            if not actions & {"click", "hotkey", "run_applescript"}:
                return False
        if re.search(r"\b(?:type|paste|fill|write)\b", lowered):
            if not actions & {"type", "set_clipboard", "hotkey", "run_applescript", "write_text_file"}:
                return False
        if re.search(r"\b(?:arrange|resize|drag|minimi[sz]e|maximi[sz]e|organize)\b", lowered):
            if "run_applescript" not in actions:
                return False
        if re.search(r"\b(?:focus|switch|close)\b", lowered):
            if not actions & {"open_app", "hotkey", "run_applescript"}:
                return False
        return True

    @classmethod
    def _should_escalate_to_os_automation(
        cls,
        objective: str,
        steps: list[DesktopTaskStep],
        context: dict[str, Any] | None,
    ) -> bool:
        context = context or {}
        if bool(context.get("disable_os_automation_escalation")):
            return False
        if cls._objective_requests_observation_only(objective):
            return False
        if cls._objective_needs_general_os_automation(objective) and not any(
            step.action == "run_applescript" for step in steps
        ):
            if cls._steps_cover_general_os_intent(objective, steps):
                return False
            return looks_like_desktop_objective(objective) or any(
                step.action in {"open_app", "open_url"} for step in steps
            )
        if not cls._primitive_steps_are_only_observational(steps):
            return False
        return looks_like_desktop_objective(objective)

    @staticmethod
    def _os_automation_effect_evidence(result: dict[str, Any]) -> tuple[bool, str]:
        if not bool(result.get("ok")):
            return False, str(result.get("error") or result.get("status") or "os automation reported failure")
        receipt_id = str(result.get("receipt_id") or "").strip()
        action_result = str(result.get("result") or "").strip()
        adapter = str(result.get("adapter") or "").strip()
        if receipt_id:
            return True, f"receipt_id={receipt_id}"
        if action_result:
            return True, f"result={action_result[:240]}"
        if adapter:
            return True, f"adapter={adapter}"
        return False, "missing os automation effect evidence"

    async def _execute_os_automation_escalation(
        self,
        *,
        capability_engine: Any,
        objective: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        step_context = dict(context or {})
        step_context.update(
            {
                "origin": step_context.get("origin") or "desktop_task",
                "route": "desktop_task.os_automation",
                "objective": objective,
                "foreground_request": True,
                "user_requested_action": True,
                "user_explicitly_authorized": True,
                "desktop_task_reason": (
                    "Primitive desktop actions were not sufficient for this objective; "
                    "escalating to governed OS automation."
                ),
                "desktop_task_expect": "OS automation receipt proves the visible desktop action ran.",
            }
        )
        try:
            result = await capability_engine.execute(
                "os_automation",
                {"goal": objective, "script_type": "applescript", "execute": True},
                context=step_context,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError, OSError, TimeoutError) as exc:
            record_degradation(
                "desktop_task",
                exc,
                action="blocked desktop task because governed OS automation escalation failed closed",
                severity="degraded",
            )
            result = {
                "ok": False,
                "status": "os_automation_unavailable",
                "error": str(exc),
            }
        if not isinstance(result, dict):
            result = {"ok": bool(result), "result": result}

        effect_verified, effect_evidence = self._os_automation_effect_evidence(result)
        receipt = {
            "index": 1,
            "action": "os_automation",
            "reason": step_context["desktop_task_reason"],
            "expect": step_context["desktop_task_expect"],
            "ok": bool(result.get("ok")) and effect_verified,
            "effect_verified": effect_verified,
            "effect_evidence": effect_evidence,
            "result": result,
        }
        ok = bool(receipt["ok"])
        return {
            "ok": ok,
            "status": "completed" if ok else "failed",
            "objective": objective,
            "steps_requested": 1,
            "steps_completed": 1 if ok else 0,
            "receipts": [receipt],
            "failures": [] if ok else [receipt],
            "planner": "os_automation_escalation",
            "summary": (
                "Desktop task completed 1/1 governed OS automation step."
                if ok
                else "Desktop task could not complete through primitive actions or governed OS automation."
            ),
        }

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

        task_context = dict(context or {})
        research_context = await self._collect_research_context(
            capability_engine=capability_engine,
            objective=objective,
            context=task_context,
        )
        if research_context:
            task_context.update(research_context)
        steps = list(params.steps)
        if not steps:
            steps = self._steps_from_context(task_context)
        if not steps:
            steps = self._derive_steps_from_objective(objective, task_context)
        else:
            steps = self._resolve_document_body_tokens(
                steps,
                self._document_body(objective, task_context),
            )

        if self._should_escalate_to_os_automation(objective, steps, task_context):
            return await self._execute_os_automation_escalation(
                capability_engine=capability_engine,
                objective=objective,
                context=task_context,
            )

        last_image_page_url = ""
        for index, step in enumerate(steps, start=1):
            target = step.target
            if step.action == "open_url":
                # Resolve the fetched-image source sentinel from the
                # fetch receipt — the source page is only known at runtime.
                if isinstance(target, dict) and target.get("url") == FETCHED_IMAGE_SOURCE_TOKEN:
                    if not last_image_page_url:
                        failures.append(
                            {
                                "index": index,
                                "action": step.action,
                                "ok": False,
                                "error": "no fetched-image source URL available to show",
                            }
                        )
                        break
                    target = dict(target, url=last_image_page_url)
                elif target == FETCHED_IMAGE_SOURCE_TOKEN:
                    if not last_image_page_url:
                        failures.append(
                            {
                                "index": index,
                                "action": step.action,
                                "ok": False,
                                "error": "no fetched-image source URL available to show",
                            }
                        )
                        break
                    target = last_image_page_url
            if isinstance(target, dict):
                target = json.dumps(target)
            payload = {
                "action": step.action,
                "target": str(target or ""),
                "x": int(step.x),
                "y": int(step.y),
            }
            step_context = dict(task_context)
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
            effect_verified, effect_evidence = self._verify_step_effect(step, result)
            receipt = {
                "index": index,
                "action": step.action,
                "reason": step.reason,
                "expect": step.expect,
                "ok": bool(result.get("ok")) and effect_verified,
                "effect_verified": effect_verified,
                "effect_evidence": effect_evidence,
                "result": result,
            }
            receipts.append(receipt)
            if step.action == "fetch_topic_image" and receipt["ok"]:
                last_image_page_url = str(result.get("page_url") or "") or last_image_page_url
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
            "research": {
                "query": task_context.get("desktop_task_research_query"),
                "sources": task_context.get("desktop_task_research_sources") or [],
                "error": task_context.get("desktop_task_research_error"),
            } if research_context else None,
            "summary": (
                f"Desktop task completed {sum(1 for receipt in receipts if receipt.get('ok'))}/"
                f"{len(steps)} governed computer-use steps."
            ),
        }
