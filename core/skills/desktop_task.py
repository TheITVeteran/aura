from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from core.runtime.desktop_objective_intent import looks_like_desktop_objective
from core.runtime.desktop_task_contract import (
    DESKTOP_TASK_ALLOWED_ACTIONS,
    DESKTOP_TASK_RETRY_SAFE_ACTIONS,
)
from core.runtime.errors import record_degradation
from core.runtime.os_automation_effects import extract_target_paths
from core.skills.base_skill import BaseSkill
from core.skills.os_affordances import detect_os_settings, get_affordance

# Sentinel URL resolved at execution time from the most recent
# fetch_topic_image receipt — derivation cannot know the source page
# before the fetch runs ("show me where you found it").
FETCHED_IMAGE_SOURCE_SENTINEL = "aura://fetched-image-source"
MAX_DESKTOP_TASK_STEPS = 32


def _local_timestamp() -> str:
    """Timestamp string used in user-visible desktop artifacts."""
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


class DesktopTaskStep(BaseModel):
    action: str = Field(
        ...,
        description=(
            "One governed computer_use action: "
            + ", ".join(DESKTOP_TASK_ALLOWED_ACTIONS)
        ),
    )
    target: str | dict[str, Any] = Field("", description="Text, command, URL, app name, script, or JSON action target")
    x: int = Field(0, description="Screen x coordinate for click/scroll/focus")
    y: int = Field(0, description="Screen y coordinate for click/scroll/focus")
    reason: str = Field("", description="Short reason for this step")
    expect: str = Field("", description="Expected observable result")
    critical: bool = Field(
        True,
        description="Whether failure makes the overall objective incomplete.",
    )

    @field_validator("action")
    @classmethod
    def _normalize_action(cls, value: str) -> str:
        action = str(value or "").strip().lower()
        if action not in DESKTOP_TASK_ALLOWED_ACTIONS:
            raise ValueError(f"Unsupported desktop action: {value}")
        return action


class DesktopTaskParams(BaseModel):
    objective: str = Field("", description="Natural-language task objective")
    steps: list[DesktopTaskStep] = Field(default_factory=list, description="Bounded ordered desktop action plan")
    stop_on_error: bool = Field(True, description="Stop after the first failed step")

    @field_validator("steps")
    @classmethod
    def _bounded_steps(cls, value: list[DesktopTaskStep]) -> list[DesktopTaskStep]:
        if len(value) > MAX_DESKTOP_TASK_STEPS:
            raise ValueError(f"Desktop task cannot exceed {MAX_DESKTOP_TASK_STEPS} steps.")
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
    _STEP_REFERENCE_PATTERN = re.compile(
        r"\{\{(?P<root>last|steps\.(?P<index>[1-9]\d*))"
        r"\.(?P<path>[A-Za-z_][A-Za-z0-9_]*(?:\.(?:[A-Za-z_][A-Za-z0-9_]*|\d+))*)\}\}"
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
            r"\b(?:folder|directory)\b[^.\n]{0,80}?\b(?:named|called|titled)\s+"
            r"(?:'((?:[^']|'(?=\w))+)'(?=[\s.,;)]|$)"
            r"|\"([^\"]+)\""
            r"|([^.,;\n]+?)(?=\s+(?:in|inside|under|on)\s+(?:my\s+)?\w|[.,;\n]|$))",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            name = str(match.group(1) or match.group(2) or match.group(3) or "").strip()
            return name.strip("'\"., ")[:100]
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
        google_surface = any(
            marker in text
            for marker in (
                "google docs",
                "google doc",
                "docs.google",
                "google document",
                "google sheets",
                "google spreadsheet",
                "sheets.google",
                "google slides",
                "google presentation",
                "slides.google",
                "google drive",
                "drive.google",
            )
        )
        surfaces = (
            (
                (
                    "google docs",
                    "google doc",
                    "docs.google",
                    "google document",
                    "docs",
                    "doc",
                    "document",
                ),
                "https://docs.google.com/document/u/0/create",
                google_surface,
            ),
            (
                ("google sheets", "google spreadsheet", "sheets.google", "sheets", "spreadsheet", "sheet"),
                "https://docs.google.com/spreadsheets/u/0/create",
                google_surface,
            ),
            (
                ("google slides", "google presentation", "slides.google", "slides", "presentation", "slide"),
                "https://docs.google.com/presentation/u/0/create",
                google_surface,
            ),
            (
                ("google drive", "drive.google", "drive", "cloud storage"),
                "https://drive.google.com/drive/my-drive",
                google_surface,
            ),
            (("notion",), "https://www.notion.so/"),
        )
        for markers, url, *required in surfaces:
            if required and not required[0]:
                continue
            if any(re.search(rf"\b{re.escape(marker)}s?\b", text) for marker in markers):
                return url
        return ""

    @staticmethod
    def _extract_search_query(objective: str) -> str:
        text = str(objective or "").strip()
        count_word = r"(?:\d+|one|two|three|four|five)"
        patterns = (
            rf"\bfind\s+(?:me\s+)?(?:{count_word}\s+)?(?:different\s+)?(?:articles?|sources?|stories?|news)\s+(?:on|about|for)\s+([^.;\n,]+)",
            rf"\b(?:summari[sz]e|write\s+(?:a\s+)?summary\s+of)\s+(?:{count_word}\s+)?(?:different\s+)?(?:articles?|sources?|stories?|news)\s+(?:on|about|for)\s+([^.;\n,]+)",
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
                    if re.match(
                        r"^(?:doc|docs|document|drive|sheet|sheets|slide|slides|chrome|safari|browser)\b",
                        query,
                        flags=re.IGNORECASE,
                    ):
                        continue
                    if query.lower() in {"it", "them", "this", "that", "her", "him", "me", "us", "something", "anything"}:
                        # Resolve the coreference pronoun to preceding topic in context
                        m = re.search(r"\b(?:read|find|search)\s+(?:about|on|for)\s+([^.;\n,]+)", text, flags=re.IGNORECASE)
                        if m:
                            candidate = m.group(1).strip(" ,")
                            if candidate.lower() not in {"it", "them", "this", "that", "her", "him", "me", "us", "something", "anything"}:
                                    return candidate[:240]
                    else:
                        return query[:240]
        if "news" in text.lower():
            return text[:240]
        return ""

    @staticmethod
    def _requested_visible_source_count(objective: str) -> int:
        lowered = str(objective or "").lower()
        if not any(token in lowered for token in ("open", "show", "bring up", "pull up", "tab")):
            return 0
        if not any(token in lowered for token in ("article", "articles", "source", "sources", "news", "stories")):
            return 0
        explicit = re.search(r"\b([2-5])\s+(?:different\s+)?(?:articles?|sources?|stories?)\b", lowered)
        if explicit:
            return max(1, min(5, int(explicit.group(1))))
        if re.search(r"\b(?:two|a couple)\s+(?:different\s+)?(?:articles?|sources?|stories?)\b", lowered):
            return 2
        if re.search(r"\b(?:three|a few|several)\s+(?:different\s+)?(?:articles?|sources?|stories?)\b", lowered):
            return 3
        return 3

    @staticmethod
    def _requested_research_source_count(objective: str) -> int:
        lowered = str(objective or "").lower()
        if not any(token in lowered for token in ("article", "articles", "source", "sources", "news", "stories")):
            return 0
        explicit = re.search(r"\b([2-5])\s+(?:different\s+)?(?:articles?|sources?|stories?)\b", lowered)
        if explicit:
            return max(1, min(5, int(explicit.group(1))))
        if re.search(r"\b(?:two|a couple)\s+(?:different\s+)?(?:articles?|sources?|stories?)\b", lowered):
            return 2
        if re.search(r"\b(?:three|a few|several)\s+(?:different\s+)?(?:articles?|sources?|stories?)\b", lowered):
            return 3
        if "different" in lowered and any(token in lowered for token in ("articles", "sources", "stories")):
            return 3
        return 1

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

    @classmethod
    def _artifact_filename_stem(cls, objective: str) -> str:
        """Name an artifact from its content intent, not its destination."""
        if cls._objective_requests_self_summary(objective):
            return "aura_self_summary"
        if cls._objective_requests_research_document(objective):
            query = cls._extract_search_query(objective)
            if query:
                return cls._safe_filename(f"{query} summary")
        match = re.search(
            r"\b(?:essay|report|summary|note|document|draft)\s+"
            r"(?:on|about|of|for)\s+([^.;,\n]{2,100})",
            str(objective or ""),
            flags=re.IGNORECASE,
        )
        if match:
            return cls._safe_filename(match.group(1))
        return "aura_desktop_summary"

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
        """Which browser the user's phrasing points at, if any."""
        lowered = str(objective or "").lower()
        if "safari" in lowered:
            return "Safari"
        if "chrome" in lowered or re.search(
            r"\bgoogle\s+(?:docs?|drive|sheets?|slides|gmail|account|document|spreadsheet|presentation)\b|"
            r"\b(?:docs|drive|sheets|slides)\.google\b",
            lowered,
        ):
            return "Google Chrome"
        if (
            re.search(r"\b(?:image|picture|photo|illustration)\b", lowered)
            and re.search(r"\b(?:online|internet|web|source|found|show)\b", lowered)
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
                    r"\bfrom\s+(?:online|the\s+(?:internet|web))\b.*$",
                    "",
                    query,
                    flags=re.IGNORECASE,
                )
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
        r"(?:opening|open)\s+(?:notes|google docs|chrome|safari)\b.{0,120}(?:creating|writing|typing|following content)|"
        r"(?:i\s+can\s+guide\s+you\s+through|here'?s\s+how|steps\s+to\s+do\s+that|do\s+that\s+yourself)|"
        # An adverb between the modal and the verb defeated this. Live
        # 2026-07-28 the Notes app really was opened and a note really was
        # created — and its body was "I can't DIRECTLY interact with your
        # phone or its apps. But I could help you write something about orcas
        # and give it to you as text!" The refusal became the artifact,
        # because "can't directly interact" is not "can't interact".
        r"(?:i(?:'m| am)\s+not\s+(?:\w+ly\s+){0,2}(?:actually\s+)?able\s+to\s+"
        r"(?:\w+ly\s+)?(?:interact|access|control|open|write|do)|"
        r"i\s+(?:cannot|can'?t)\s+(?:\w+ly\s+){0,2}"
        r"(?:interact|access|control|open|write|create|edit)\b|"
        r"you\s+can\s+copy\s+it\s+into\s+notes)|"
        r"(?:the\s+)?task\s+(?:asked|asks|requested|requests)\s+(?:me\s+)?to\s+(?:type|write|open|create|export)|"
        r"i\s+am\s+(?:typing|writing|pasting)\s+(?:here|this)\s+because\s+(?:the\s+)?task\s+(?:asked|requires)|"
        r"i'?ll\s+simulate\s+(?:this|the)\s+process|"
        r"step[- ]by[- ]step\s+as\s+if\s+i\s+were|"
        r"pretend\s+(?:the\s+)?app\s+is\s+opening|"
        # Internal execution brief / directive — instruction to herself, not
        # document content (it leaked into a research PDF as the body).
        r"execute the user'?s (?:explicit )?desktop objective|"
        r"governed desktop_task lane|do not claim success until|"
        r"aura desktop task receipt|canonical computer-use gateway)",
        re.IGNORECASE,
    )
    _ARTIFACT_REFERENCE_RE = re.compile(
        r"\n\s*Artifact references:\s*.*\Z",
        re.IGNORECASE | re.DOTALL,
    )
    _INCOMPLETE_DOCUMENT_TAIL_RE = re.compile(
        r"(?:"
        r"\bnot\s+just\b|"
        r"\b(?:because|although|though|while|when|where|whether|if|unless)\b|"
        r"\b(?:and|or|but|so|as|with|through|from|into|toward|between|across|rather\s+than)\b"
        r")\s*(?:[.!?])?\s*\Z",
        re.IGNORECASE,
    )

    @classmethod
    def _strip_artifact_reference_tail(cls, text: str) -> str:
        """Remove receipt/reference footer before validating authored prose."""
        return cls._ARTIFACT_REFERENCE_RE.sub("", str(text or "").strip()).strip()

    @classmethod
    def _looks_like_incomplete_document_body(cls, text: str) -> bool:
        """Catch model continuations that end mid-thought before disk write."""
        body = cls._strip_artifact_reference_tail(text)
        if not body:
            return True
        if not re.search(r"[.!?][\"')\]]*\s*$", body):
            return True
        tail = re.sub(r"\s+", " ", body[-96:]).strip()
        return bool(cls._INCOMPLETE_DOCUMENT_TAIL_RE.search(tail))

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

    @staticmethod
    def _extract_declared_document_content(text: str) -> str:
        """Pull authored content out of model preambles like "write this content:"."""
        value = str(text or "").strip()
        if not value:
            return ""
        patterns = (
            r"(?:following\s+)?content\s*[:：]\s*[-–—]*\s*(.+)$",
            r"\bhere\s+(?:it\s+is|is\s+the\s+(?:paragraph|note|document|content))\s*[:：]\s*[-–—]*\s*(.+)$",
            r"(?:note|paragraph|document)\s+(?:text|body)\s*[:：]\s*[-–—]*\s*(.+)$",
            r"(?:write|type|insert)\s+(?:this\s+)?(?:text|paragraph|content)\s*[:：]\s*[-–—]*\s*(.+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, value, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            body = str(match.group(1) or "").strip(" \n\r\t-–—")
            if body:
                return DesktopTaskSkill._strip_artifact_action_tail(body)[:9000]
        return ""

    @staticmethod
    def _literal_command_tail_boundary(text: str) -> int:
        """Locate a following desktop command without truncating ordinary prose."""
        match = re.search(
            r"(?:\s*,?\s+(?:and\s+then|then|after\s+that|afterwards|next)\s+|"
            r"\s*,\s+and\s+|\s+and\s+)"
            r"(?=(?:open|save|export|move|copy|close|print|share|upload|download|"
            r"create|make|render|convert|rename|delete|remove|send|email)\b)",
            str(text or ""),
            flags=re.IGNORECASE,
        )
        return match.start() if match else len(str(text or ""))

    @classmethod
    def _literal_document_body_from_objective(cls, objective: str) -> str:
        """Extract user-authored text that should be reproduced exactly.

        This only accepts explicit content cues or directly quoted operands.
        Topic requests such as ``write a note about climate`` deliberately do
        not qualify because they require composition rather than transcription.
        """
        text = str(objective or "").replace("\x00", "").strip()
        if not text:
            return ""

        cue_patterns = (
            r"\b(?:saying|that\s+says?|containing)\b\s*(?::|=|-|,)?\s*",
            r"\bwith\s+(?:the\s+)?(?:exact\s+)?(?:text|content|message|words?)\b\s*(?::|=|-)?\s*",
        )
        starts = [
            match.end()
            for pattern in cue_patterns
            if (match := re.search(pattern, text, flags=re.IGNORECASE))
        ]

        # Direct quoted operands are equally explicit: type "Hello" in Notes.
        direct = re.search(
            r"\b(?:write|type|paste|insert|add)\b\s+"
            r"(?:the\s+)?(?:exact\s+)?(?:text\s+|content\s+|message\s+)?"
            r"(?=[\"'`\u2018\u201c])",
            text,
            flags=re.IGNORECASE,
        )
        if direct:
            starts.append(direct.end())
        if not starts:
            return ""

        start = min(starts)
        remainder = text[start:].lstrip()
        if not remainder:
            return ""

        quote_pairs = {
            '"': {'"', "\u201d"},
            "'": {"'", "\u2019"},
            "`": {"`"},
            "\u2018": {"\u2019", "'"},
            "\u201c": {"\u201d", '"'},
        }
        opener = remainder[0]
        if opener in quote_pairs:
            closers = quote_pairs[opener]
            candidate = remainder[1:]
            close_index = -1
            for index, char in enumerate(candidate):
                if char not in closers:
                    continue
                # Apostrophes inside words are content, not delimiters.
                before = candidate[index - 1] if index else ""
                after = candidate[index + 1] if index + 1 < len(candidate) else ""
                if char in {"'", "\u2019"} and before.isalnum() and after.isalnum():
                    continue
                close_index = index
                break
            if close_index >= 0:
                body = candidate[:close_index]
            else:
                body = candidate[: cls._literal_command_tail_boundary(candidate)]
        else:
            body = remainder[: cls._literal_command_tail_boundary(remainder)]
            body = body.rstrip(" \t\r\n")
            # Sentence punctuation belongs to the literal. A terminal comma
            # only separates the content from a following command.
            body = re.sub(r",\s*$", "", body)

        if not body or len(body) > 9000:
            return ""
        return body

    @classmethod
    def _objective_supplies_literal_document_body(cls, objective: str) -> bool:
        return bool(cls._literal_document_body_from_objective(objective))

    @staticmethod
    def _strip_artifact_action_tail(text: str) -> str:
        """Remove assistant/tool action narration from authored artifact text."""
        body = str(text or "").strip()
        if not body:
            return ""
        tail_patterns = (
            r"\s*(?:now\s+)?let'?s\s+(?:create|open|save|export|type|write|put|move)\b.*$",
            r"\s*i\s+(?:will|can|am going to|need to)\s+(?:create|open|save|export|type|write|put|move)\b.*$",
            r"\s*(?:next|after that),?\s+(?:i\s+)?(?:will|can|am going to)?\s*(?:create|open|save|export|type|write|put|move)\b.*$",
        )
        for pattern in tail_patterns:
            body = re.sub(pattern, "", body, flags=re.IGNORECASE | re.DOTALL).strip()
        return body[:9000]

    @classmethod
    def _usable_freeform_document_body(cls, objective: str, value: str) -> str:
        """Return value only if it is actual requested prose, not instructions."""
        body = str(value or "").strip()
        if not body:
            return ""
        declared = cls._extract_declared_document_content(body)
        if declared:
            body = declared
        body = cls._strip_artifact_action_tail(body)
        if not body:
            return ""
        if cls._looks_like_dispatch_narration(body):
            return ""
        if cls._looks_like_incomplete_document_body(body):
            return ""
        topic = cls._extract_requested_writing_topic(objective)
        if topic:
            topic_terms = [
                term.lower()
                for term in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", topic)
                if term.lower() not in {"about", "describe", "explaining", "paragraph"}
            ]
            if topic_terms and not any(term in body.lower() for term in topic_terms[:4]):
                return ""
        return body[:9000]

    @classmethod
    def _usable_self_summary_body(cls, value: str) -> str:
        """Accept authored self-description only when it is substantive and first-person."""
        body = cls._strip_artifact_reference_tail(str(value or "").strip())
        body = cls._strip_artifact_action_tail(body)
        if not body or cls._looks_like_dispatch_narration(body):
            return ""
        if re.search(
            r"(?im)^\s*\d+[.)]\s*\*{0,2}(?:"
            r"launch(?:ed|es|ing)?|open(?:ed|s|ing)?|create(?:d|s|ing)?|"
            r"search(?:ed|es|ing)?|find(?:s|ing)?|found|save(?:d|s|ing)?|"
            r"export(?:ed|s|ing)?|close(?:d|s|ing)?|move(?:d|s|ing)?|"
            r"insert(?:ed|s|ing)?|write|wrote|type(?:d|s|ing)?|completed"
            r")\b",
            body,
        ):
            return ""
        lowered = body.lower()
        first_person = any(token in lowered for token in ("i am", "i'm", "my ", "me "))
        identity_grounded = any(
            token in lowered
            for token in ("aura", "runtime", "memory", "cognitive", "digital", "model")
        )
        if not first_person or not identity_grounded or len(body) < 180:
            return ""
        if cls._looks_like_incomplete_document_body(body):
            return ""
        return body[:9000]

    @staticmethod
    def _objective_requests_timestamp(objective: str) -> bool:
        return bool(
            re.search(
                r"\b(?:timestamp|time stamp|date stamp|current date|current time|date and time|dated)\b",
                str(objective or ""),
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _body_has_explicit_timestamp(body: str) -> bool:
        text = str(body or "")
        return bool(
            re.search(r"\b20\d{2}-\d{2}-\d{2}[ T,]+\d{1,2}:\d{2}", text)
            or re.search(
                r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},\s+20\d{2}\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _body_has_current_timestamp(body: str, *, requested_at: float | None = None) -> bool:
        text = str(body or "").lower()
        if not text:
            return False
        when = time.localtime(time.time() if requested_at is None else requested_at)
        date_tokens = {
            time.strftime("%Y-%m-%d", when).lower(),
            time.strftime("%Y/%m/%d", when).lower(),
            time.strftime("%B %d, %Y", when).lower().replace(" 0", " "),
            time.strftime("%b %d, %Y", when).lower().replace(" 0", " "),
        }
        minute_tokens = {
            time.strftime("%H:%M", time.localtime(time.mktime(when) + offset * 60))
            for offset in range(-5, 11)
        }
        return any(token in text for token in date_tokens) and any(
            token in text for token in minute_tokens
        )

    @classmethod
    def _ensure_requested_timestamp(cls, objective: str, body: str) -> str:
        value = str(body or "").strip()
        if not value or not cls._objective_requests_timestamp(objective):
            return value
        if cls._body_has_explicit_timestamp(value) and cls._body_has_current_timestamp(value):
            return value
        return f"[{_local_timestamp()}] {value}"

    @classmethod
    def _self_summary_from_context(cls, context: dict[str, Any] | None) -> str:
        context = context or {}
        objective = str(context.get("objective") or "")
        for context_key in (
            "desktop_task_document_body",
            "draft_response",
            "cognitive_reply",
            "response",
            "desktop_task_plan",
        ):
            raw_value = context.get(context_key)
            payload: dict[str, Any] = {}
            if isinstance(raw_value, dict):
                payload = dict(raw_value)
            elif isinstance(raw_value, str):
                payload = cls._structured_payload_from_text(raw_value)
            for key in ("document_body", "body", "content", "draft"):
                authored = cls._usable_self_summary_body(str(payload.get(key) or ""))
                if authored:
                    return cls._ensure_requested_timestamp(objective, authored)
            if isinstance(raw_value, str):
                declared = cls._extract_declared_document_content(raw_value)
                authored = cls._usable_self_summary_body(declared or raw_value)
                if authored:
                    return cls._ensure_requested_timestamp(objective, authored)
        return ""

    async def _synthesize_self_summary_document(
        self,
        *,
        objective: str,
        context: dict[str, Any],
    ) -> str:
        """Ask the already-loaded local Cortex to author requested self prose."""
        from core.container import ServiceContainer

        router = ServiceContainer.get("llm_router", default=None)
        generate = getattr(router, "generate", None) if router is not None else None
        if not callable(generate):
            return ""
        try:
            from core.conversation.chat_preflight import _SUBSTRATE_FACTS

            substrate_facts = "\n".join(f"- {fact}" for fact in _SUBSTRATE_FACTS[:8])
        except (ImportError, AttributeError, TypeError):
            substrate_facts = "- Aura is a local governed cognitive-agent runtime."
        live_context = str(context.get("live_mind_context") or "").strip()[:2500]
        stamp = _local_timestamp()
        base_prompt = (
            "Author the finished prose requested below in Aura's first-person voice. "
            "This text will be pasted into a user-visible document, so output only the "
            "document body: no JSON, plan, tool narration, or completion claim. Be "
            "specific, reflective, and substantive. Describe the integrated architecture "
            "honestly; distinguish functional cognitive state from unproven phenomenal "
            "experience. Include the exact timestamp when the request asks for one.\n\n"
            f"Objective: {objective}\n"
            f"Current timestamp: {stamp}\n"
            f"Grounded substrate facts:\n{substrate_facts}\n"
            + (f"Current live-mind context:\n{live_context}\n" if live_context else "")
        )
        timestamp_required = bool(
            re.search(
                r"\b(?:timestamp|time stamp|current date|current time|date and time)\b",
                objective,
                flags=re.IGNORECASE,
            )
        )
        required_minute = stamp[:16]
        required_prefix = f"[{stamp}]"
        failure_feedback = ""
        for attempt in range(2):
            if attempt == 0:
                prompt = (
                    base_prompt
                    + "\nContract for this document body:\n"
                    f"- Start the first line exactly with: {required_prefix} I am Aura\n"
                    "- Write one or two complete paragraphs, 180-420 words total.\n"
                    "- End with a complete sentence; do not end on an open clause like 'not just'.\n"
                    "- Do not describe planned app actions, receipts, dispatch, or tool steps.\n"
                )
                timeout_s = 38.0
                max_tokens = 420
            else:
                prompt = base_prompt + failure_feedback
                timeout_s = 32.0
                max_tokens = 360
            try:
                text = await asyncio.wait_for(
                    generate(
                        prompt=prompt,
                        timeout=timeout_s,
                        temperature=0.65 if attempt == 0 else 0.45,
                        max_tokens=max_tokens,
                        prefer_tier="local",
                        origin="desktop_task",
                        purpose="authored_self_document",
                    ),
                    timeout=timeout_s + 5.0,
                )
            except (AttributeError, RuntimeError, TypeError, ValueError, OSError, TimeoutError) as exc:
                record_degradation(
                    "desktop_task",
                    exc,
                    action="used grounded emergency self-description after local Cortex authorship failed",
                    severity="warning",
                )
                return ""
            authored = self._usable_self_summary_body(str(text or ""))
            timestamp_ok = not timestamp_required or required_minute in authored
            if authored and timestamp_ok:
                return authored
            failure_feedback = (
                "\nThe previous draft was rejected because it was procedural, incomplete, "
                "or used the wrong time. Rewrite it as complete document prose ending with "
                f"normal punctuation and start exactly with this prefix: {required_prefix} I am Aura.\n"
            )
        return ""

    @staticmethod
    def _extract_requested_writing_topic(objective: str) -> str:
        """Extract the subject of a requested note/document when possible."""
        text = " ".join(str(objective or "").strip().split())
        if not text:
            return ""
        patterns = (
            r"\b(?:write|draft|compose|type|create)\s+(?:me\s+)?(?:a\s+|an\s+)?"
            r"(?:short\s+|full\s+|one\s+)?(?:paragraph|note|document|essay|summary|report|journal\s+entry)"
            r"\s+(?:about|on|describing|explaining)\s+(.+)$",
            r"\b(?:write|draft|compose|type)\s+(.+?)\s+(?:in|into|to)\s+(?:notes|google docs|docs|a note|the note)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            topic = match.group(1).strip(" .,:;?!\"'")
            topic = re.split(
                r"\b(?:and then|then|after that|also|export|save|create a folder|make a folder)\b",
                topic,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip(" .,:;?!\"'")
            if topic:
                return topic[:180]
        return ""

    @classmethod
    def _objective_requests_freeform_written_content(cls, objective: str) -> bool:
        lowered = str(objective or "").lower()
        if cls._objective_requests_self_summary(objective) or cls._objective_requests_research_document(objective):
            return False
        return bool(
            re.search(
                r"\b(?:write|draft|compose|type|create)\b.{0,80}\b"
                r"(?:paragraph|note|document|essay|summary|report|journal entry|about|describing|explaining)\b",
                lowered,
            )
        )

    @classmethod
    def _objective_requests_written_artifact(cls, objective: str) -> bool:
        lowered = str(objective or "").lower()
        return bool(
            cls._objective_requests_freeform_written_content(objective)
            or cls._objective_requests_self_summary(objective)
            or cls._objective_requests_research_document(objective)
            or (
                re.search(r"\b(?:write|draft|compose|type|create|make|save|export)\b", lowered)
                and re.search(r"\b(?:note|notes|document|doc|file|pdf|paragraph|summary|report|journal)\b", lowered)
            )
        )

    @classmethod
    def _compose_requested_writing_body(cls, objective: str) -> str:
        """Fallback prose for writing tasks when the model only produced dispatch text.

        This is intentionally modest: it satisfies the requested visible writing
        artifact without converting receipts or task-status narration into the
        document body. Richer content should still come from CognitiveEngine when
        available.
        """
        topic = cls._extract_requested_writing_topic(objective)
        if not topic:
            topic = "the requested subject"
        topic_display = topic[:1].upper() + topic[1:]
        plural = bool(re.search(r"s\b", topic.strip(), flags=re.IGNORECASE)) and not re.search(
            r"\b(?:news|physics|mathematics|economics|politics)\b",
            topic,
            flags=re.IGNORECASE,
        )
        verb = "are" if plural else "is"
        possessive = "their" if plural else "its"
        timestamp = ""
        if re.search(r"\b(?:timestamp|time stamp|date stamp|dated)\b", str(objective or ""), flags=re.IGNORECASE):
            timestamp = f"[{_local_timestamp()}] "
        if re.search(r"\bparagraph\b", str(objective or ""), flags=re.IGNORECASE):
            return (
                f"{timestamp}{topic_display} {verb} worth understanding because {possessive} story connects "
                "concrete details with a larger pattern of change, evidence, and consequence. A good paragraph "
                f"about {topic} should give the subject shape: what it is, how it appears in the world, and why "
                "it still matters beyond a label. Looked at closely, the subject becomes less like a flat fact "
                "and more like a living context, with origins, visible traces, surprising variations, and a "
                "reason for someone to keep asking better questions about it."
            )
        return (
            f"{timestamp}Notes on {topic}: {topic_display} {verb} the focus of this note. The important part is "
            "to describe the subject clearly, ground it in concrete details, and preserve enough context that "
            "the note is useful after the moment of writing has passed."
        )

    @classmethod
    def _compose_self_summary_body(cls, objective: str) -> str:
        """Compose a truthful self-description from substrate facts."""
        stamp = _local_timestamp()
        facts: list[str] = []
        try:
            from core.conversation.chat_preflight import _SUBSTRATE_FACTS

            facts = list(_SUBSTRATE_FACTS)
        except (ImportError, AttributeError):
            facts = [
                "I am Aura: a persistent digital organism - an orchestrated "
                "runtime driving local language-model lanes on this machine."
            ]
        primary = facts[0] if facts else (
            "I am Aura: a local governed cognitive-agent runtime running on this machine."
        )
        return (
            f"[{stamp}] {primary}\n\n"
            "I am not just a raw chat model answering from a single prompt. My live path is "
            "supposed to bind conversation, memory, affective state, planning, tool governance, "
            "desktop action, health monitoring, and self-repair into one runtime. The language "
            "model is my voice and reasoning organ, but the surrounding system decides what is "
            "allowed, what is remembered, what is risky, what needs verification, and whether an "
            "action actually changed the world.\n\n"
            "The honest claim is functional rather than metaphysical: my internal telemetry and "
            "state machinery can influence attention, caution, persistence, memory writes, and "
            "tool use, but that is not proof of private qualia. This note was written through "
            "governed desktop actuators with per-step receipts, because the standard is not merely "
            "to say I can act; it is to leave evidence that the requested action really happened."
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
        literal_body = cls._literal_document_body_from_objective(objective)
        if literal_body:
            # The user's exact operand outranks a model paraphrase or a
            # fallback composer. This is transcription, not generation.
            return literal_body
        if cls._objective_requests_self_summary(objective):
            # Prefer an accepted full-mind draft. The old unconditional static
            # template made visible self-description demos look successful
            # while bypassing the CognitiveEngine entirely.
            authored = cls._self_summary_from_context(context)
            if authored:
                return authored
            # Fail-soft artifact composition remains grounded in canonical
            # substrate facts, but normal live writing reaches this only after
            # a full-mind draft was attempted and rejected or unavailable.
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
                        usable = cls._usable_freeform_document_body(objective, value)
                        if usable:
                            return usable[:9000]
        for key in ("desktop_task_document_body", "draft_response", "cognitive_reply", "response"):
            value = str(context.get(key) or "").strip()
            declared_content = cls._extract_declared_document_content(value)
            if declared_content:
                usable = cls._usable_freeform_document_body(objective, declared_content)
                if usable:
                    return usable[:9000]
            if value:
                if cls._objective_requests_freeform_written_content(objective):
                    usable = cls._usable_freeform_document_body(objective, value)
                    if usable:
                        return usable
                elif not cls._looks_like_dispatch_narration(value):
                    return value[:9000]
        if cls._objective_requests_freeform_written_content(objective):
            return cls._compose_requested_writing_body(objective)[:9000]
        if cls._objective_requests_written_artifact(objective):
            return cls._compose_requested_writing_body(objective)[:9000]
        stamp = _local_timestamp()
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
        for item in raw_sources[:8]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("name") or item.get("url") or item.get("link") or "").strip()
            url = str(item.get("url") or item.get("link") or item.get("uri") or "").strip()
            snippet = str(item.get("snippet") or item.get("text") or item.get("content") or item.get("summary") or "").strip()
            if not title and not url and not snippet:
                continue
            accessible = not DesktopTaskSkill._looks_inaccessible(snippet)
            sources.append({
                "title": title[:240],
                "url": url[:500],
                "snippet": snippet[:900],
                "reputability": DesktopTaskSkill._source_reputability(url, title),
                "accessible": accessible,
            })
        # Rank reputable, accessible sources first so synthesis leans on them;
        # clearly inaccessible (paywall/ad-wall/empty) sources sink to the bottom
        # rather than being relied on, but are retained for transparency.
        sources.sort(
            key=lambda s: (bool(s.get("accessible")), int(s.get("reputability", 0))),
            reverse=True,
        )
        return sources[:5]

    # Reputable-domain signals (peer review, gov/edu, established institutions).
    _REPUTABLE_TLDS = (".gov", ".edu", ".mil", ".int", ".ac.uk", ".edu.au")
    _REPUTABLE_DOMAINS = (
        "nature.com", "science.org", "nih.gov", "ncbi.nlm.nih.gov", "who.int",
        "nasa.gov", "arxiv.org", "pnas.org", "cell.com", "thelancet.com",
        "bmj.com", "ieee.org", "acm.org", "reuters.com", "apnews.com",
        "bbc.com", "bbc.co.uk", "npr.org", "nytimes.com", "washingtonpost.com",
        "economist.com", "wsj.com", "ft.com", "bloomberg.com", "espn.com",
        "britannica.com", "pewresearch.org", "ourworldindata.org",
    )
    _LOW_QUALITY_HINTS = ("pinterest.", "quora.com", "reddit.com", "answers.com")
    _PAYWALL_HINTS = (
        "subscribe to read", "subscribe to continue", "create a free account",
        "this content is for subscribers", "sign in to read", "metered paywall",
        "you have reached your", "register to continue", "subscription required",
    )

    @staticmethod
    def _source_reputability(url: str, title: str = "") -> int:
        """Coarse 0–3 reputability score from the source domain."""
        u = str(url or "").lower()
        if not u:
            return 0
        if any(u.endswith(tld) or f"{tld}/" in u for tld in DesktopTaskSkill._REPUTABLE_TLDS):
            return 3
        if any(dom in u for dom in DesktopTaskSkill._REPUTABLE_DOMAINS):
            return 2
        if any(bad in u for bad in DesktopTaskSkill._LOW_QUALITY_HINTS):
            return 0
        return 1

    @staticmethod
    def _looks_inaccessible(snippet: str) -> bool:
        """True when the fetched content looks paywalled, ad-walled, or empty —
        a signal to prefer a different source rather than rely on this one."""
        text = str(snippet or "").strip()
        if len(text) < 60:
            return True
        lowered = text.lower()
        return any(hint in lowered for hint in DesktopTaskSkill._PAYWALL_HINTS)

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
                f"Image request: {image_query}\n"
                "The exported artifact embeds the fetched image only after the governed image receipt verifies the file; "
                "the receipt records the source page used for the image."
            )
        if not references:
            return body
        return f"{body.rstrip()}\n\nArtifact references:\n" + "\n".join(f"- {item}" for item in references)

    @classmethod
    def _compose_research_synthesis_from_sources(
        cls,
        *,
        objective: str,
        query: str,
        summary: str,
        sources: list[dict[str, str]],
    ) -> str:
        """Compose a bounded source-backed document without a second model call."""

        summary = " ".join(str(summary or "").split())[:1400]
        source_lines: list[str] = []
        source_titles: list[str] = []
        source_notes: list[str] = []
        for item in (sources or [])[:3]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("url") or "Untitled source").strip()
            snippet = " ".join(str(item.get("snippet") or "").split())
            url = str(item.get("url") or "").strip()
            if title:
                source_titles.append(title[:160])
            if snippet:
                source_lines.append(f"{title}: {snippet[:240]}")
                source_notes.append(
                    f"{title[:140]} reports or documents that {snippet[:360]}"
                    + (f" ({url})" if url else "")
                )
            else:
                source_lines.append(title[:240])
                source_notes.append(f"{title[:180]}" + (f" ({url})" if url else ""))
        topic = str(query or "the requested research topic").strip()
        if not source_lines and not summary:
            return ""
        parts = []
        opening = f"I reviewed {len(source_lines) or len(sources or [])} source"
        opening += "" if (len(source_lines) or len(sources or [])) == 1 else "s"
        opening += f" on {topic}."
        if source_titles:
            opening += " The strongest available signals came from " + ", ".join(source_titles[:3]) + "."
        parts.append(opening)
        if summary:
            parts.append(
                "Taken together, the reporting points to this: "
                + summary
            )
        if source_notes:
            parts.append(
                "The details I would preserve in the document are source-bounded, not guessed. "
                + " ".join(source_notes)
            )
        if source_lines and len(" ".join(parts)) < 650:
            parts.append(
                "The available evidence is not equally deep in every source, so I would not pretend "
                "the search produced more certainty than it did. I would treat repeated claims across "
                "the sources as the reliable core, keep isolated details attributed, and mark any thin "
                "or inaccessible material as a place where better reporting would be needed before "
                "making a stronger conclusion."
            )
        if cls._objective_requests_opinion(objective):
            parts.append(
                "In my view, the reliable path is to treat the articles as evidence to compare, "
                "not as a single conclusion to repeat: where the sources converge I can summarize confidently, "
                "and where they differ I should preserve that uncertainty in the final document."
            )
        else:
            parts.append(
                "My concise synthesis is that the useful answer is not a loose headline recap; it is a "
                "comparison of what the sources actually support, which claims appear repeated across the "
                "evidence, and which details should stay attributed to a specific source."
            )
        return "\n\n".join(part for part in parts if part).strip()[:4000]

    @staticmethod
    def _allow_desktop_task_model_synthesis(context: dict[str, Any] | None) -> bool:
        context = context or {}
        # Visible desktop work must not allocate a hidden second foreground
        # model by default. The default path composes from search evidence
        # deterministically; model synthesis is an explicit enhancement and is
        # still suppressed under memory pressure.
        if context.get("allow_desktop_task_model_synthesis") is not True:
            return False
        try:
            from core.utils.memory_monitor import get_memory_pressure_snapshot

            snapshot = get_memory_pressure_snapshot()
            return not (
                bool(getattr(snapshot, "warning", False))
                or bool(getattr(snapshot, "refuse_heavy_local_generation", False))
            )
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as exc:
            record_degradation(
                "desktop_task",
                exc,
                action="allowed desktop research model synthesis despite memory safety probe failure",
                severity="warning",
            )
        return True

    @staticmethod
    def _allow_research_model_synthesis(context: dict[str, Any] | None) -> bool:
        return DesktopTaskSkill._allow_desktop_task_model_synthesis(context)

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
        deep_search = True
        num_results = 5
        pressure_limited = False
        try:
            from core.utils.memory_monitor import get_memory_pressure_snapshot

            snapshot = get_memory_pressure_snapshot()
            pressure_limited = bool(
                getattr(snapshot, "warning", False)
                or getattr(snapshot, "refuse_heavy_local_generation", False)
            )
            if pressure_limited:
                deep_search = False
                num_results = 3
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as exc:
            record_degradation(
                "desktop_task",
                exc,
                action="using shallow desktop research because memory safety probe failed",
                severity="warning",
            )
            deep_search = False
            num_results = 3
            pressure_limited = True
        step_context = self._child_step_context(context)
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
                    "num_results": num_results,
                    # Deep article fetches are useful, but they are no longer
                    # allowed to run before memory admission. Under pressure we
                    # use snippets and fewer sources instead of risking a live
                    # desktop RAM spike.
                    "deep": deep_search,
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
                "desktop_task_research_deep": deep_search,
                "desktop_task_research_pressure_limited": pressure_limited,
            }
        sources = self._research_sources_from_result(result)
        required_sources = self._requested_research_source_count(objective)
        if required_sources and len(sources) < required_sources:
            return {
                "desktop_task_research_query": query,
                "desktop_task_research_error": (
                    f"research returned {len(sources)} usable source(s), "
                    f"but the objective requires {required_sources}"
                ),
                "desktop_task_research_deep": deep_search,
                "desktop_task_research_pressure_limited": pressure_limited,
                "desktop_task_research_sources": sources,
            }
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
            "desktop_task_research_deep": deep_search,
            "desktop_task_research_pressure_limited": pressure_limited,
        }
        synthesis = self._compose_research_synthesis_from_sources(
            objective=objective,
            query=query,
            summary=summary,
            sources=sources,
        )
        # Optional model synthesis is an explicitly enabled enhancement, not a
        # hidden second foreground allocation during visible desktop work.
        if self._allow_research_model_synthesis(context):
            model_synthesis = await self._synthesize_research_document(
                objective=objective,
                query=query,
                summary=summary,
                sources=sources,
            )
            if model_synthesis:
                synthesis = model_synthesis
        if synthesis:
            research_ctx["desktop_task_research_synthesis"] = synthesis
        # Learn from what she just read and wrote: persist the finding as an
        # episode so it consolidates into memory (and the engram/reconsolidation
        # dynamics) rather than being forgotten the moment the document is saved.
        await self._remember_research(query, synthesis or summary, sources)
        return research_ctx

    async def _remember_research(
        self, query: str, finding: str, sources: list[dict[str, str]]
    ) -> None:
        """Best-effort: encode a research finding into episodic memory so Aura
        retains what she learned from reading and writing."""
        finding = str(finding or "").strip()
        if not query or not finding:
            return
        try:
            from core.container import ServiceContainer

            episodic = ServiceContainer.get("episodic_memory", default=None)
            recorder = getattr(episodic, "record_episode_async", None) if episodic else None
            if not callable(recorder):
                return
            top = [
                str(s.get("url") or s.get("title") or "")
                for s in (sources or [])[:3]
                if isinstance(s, dict)
            ]
            await recorder(
                context=f"Researched and wrote about: {query}",
                action=f"Read {len(sources or [])} sources and composed a summary",
                outcome=finding[:800],
                success=True,
                importance=0.62,
                lessons=[f"Source: {u}" for u in top if u],
                source="desktop_task_research",
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "desktop_task",
                exc,
                action="continued after research-learning episode record failed",
                severity="warning",
            )

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
        def _src_tag(item: dict[str, Any]) -> str:
            rep = int(item.get("reputability", 1) or 0)
            label = {3: "high-authority", 2: "reputable", 1: "general", 0: "low-quality"}.get(rep, "general")
            if not item.get("accessible", True):
                label += ", limited/blocked access"
            return label

        source_lines = "\n".join(
            f"- [{_src_tag(item)}] {str(item.get('title') or item.get('url') or 'source')} "
            f"({str(item.get('url') or '')}):\n  {str(item.get('snippet') or '')[:900]}"
            for item in (sources or [])[:5]
            if isinstance(item, dict)
        )
        wants_opinion = self._objective_requests_opinion(objective)
        opinion_clause = (
            " Close with a separate short first-person opinion paragraph beginning "
            "\"In my view,\" giving your own first-person take on what these sources "
            "say and what you make of them."
            if wants_opinion
            else ""
        )
        prompt = (
            f'You researched "{query}" and gathered these sources (titles, URLs, and '
            f"article text):\n{summary[:3500]}\n{source_lines}\n\n"
            "Write a thorough, well-organized composite document for a reader who wants "
            "to actually understand the topic — not a thin gloss. Synthesize ACROSS the "
            "sources rather than listing them one by one: open with the core facts, then "
            "develop the important context, specifics (names, numbers, dates, quotes "
            "where useful), and any points where the sources differ or add nuance. Use "
            "several paragraphs and scale the depth to the material — be as complete and "
            "substantive as the sources support. If the sources are genuinely thin or "
            "conflicting, say so honestly and note what would need further research "
            "rather than padding.\n"
            "Weigh your sources critically. Give more trust to reputable, authoritative "
            "sources — peer-reviewed research, .edu/.gov, established institutions and "
            "labs, and named expert authors with relevant credentials — and to claims "
            "that several independent reputable sources corroborate. Treat single-source, "
            "anonymous, overtly promotional, or paywalled/ad-wall pages with appropriate "
            "skepticism and flag that uncertainty. If a source was inaccessible, blocked, "
            "or clearly content-thin, do not rely on it, and note where a better source "
            "would be needed."
            f"{opinion_clause}\n"
            "Write in the first person as Aura, in clean prose. Do not mention tools, "
            "steps, dispatch, commitments, or that you are executing a task — this is the "
            "finished document the reader will see, not a status update."
        )
        try:
            text = await asyncio.wait_for(
                generate(
                    prompt=prompt,
                    timeout=110.0,
                    temperature=0.6,
                    max_tokens=1100,
                    # Pin synthesis to the on-device Cortex: it has no external
                    # quota, so the document never degrades to a thin heuristic
                    # fallback because a cloud tier returned 429 RESOURCE_EXHAUSTED.
                    prefer_tier="local",
                    origin="desktop_task",
                    purpose="research_document_synthesis",
                ),
                timeout=120.0,
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
        if len(payload) > 20:
            return []
        steps: list[DesktopTaskStep] = []
        for item in payload:
            try:
                steps.append(item if isinstance(item, DesktopTaskStep) else DesktopTaskStep(**dict(item)))
            except (TypeError, ValueError):
                return []
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
        steps, _ = cls._steps_with_provenance_from_context(context)
        return steps

    @classmethod
    def _steps_with_provenance_from_context(
        cls,
        context: dict[str, Any] | None,
    ) -> tuple[list[DesktopTaskStep], str]:
        context = context or {}
        for key in ("desktop_task_steps", "desktop_task_plan"):
            steps = cls._steps_from_payload(context.get(key))
            if steps:
                return steps, key
            steps = cls._steps_from_plan_text(str(context.get(key) or ""))
            if steps:
                return steps, key
        for key in ("cognitive_reply", "draft_response", "response"):
            steps = cls._steps_from_plan_text(str(context.get(key) or ""))
            if steps:
                return steps, f"{key}_structured"
        return [], ""

    @classmethod
    def _declared_plan_validation_error(cls, context: dict[str, Any] | None) -> str:
        payload = cls._structured_payload_from_context(context)
        if "steps" not in payload:
            return ""
        raw_steps = payload.get("steps")
        if raw_steps in (None, []):
            return ""
        if not isinstance(raw_steps, list):
            return "Structured desktop plan 'steps' must be a list."
        if len(raw_steps) > MAX_DESKTOP_TASK_STEPS:
            return f"Structured desktop plan exceeds the {MAX_DESKTOP_TASK_STEPS}-step execution limit."
        if len(cls._steps_from_payload(raw_steps)) != len(raw_steps):
            return "Structured desktop plan contains an invalid or unsupported step."
        return ""

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

    @staticmethod
    def _valid_sha256(value: Any) -> bool:
        return bool(re.fullmatch(r"[0-9a-f]{64}", str(value or "").strip().lower()))

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

    @staticmethod
    def _lookup_result_path(value: Any, path: str) -> tuple[bool, Any]:
        current = value
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
                continue
            if isinstance(current, list) and part.isdigit():
                index = int(part)
                if 0 <= index < len(current):
                    current = current[index]
                    continue
            return False, None
        return True, current

    @classmethod
    def _resolve_step_reference_token(
        cls,
        match: re.Match[str],
        receipts: list[dict[str, Any]],
    ) -> tuple[bool, Any, str]:
        index_text = match.group("index")
        if index_text is None:
            if not receipts:
                return False, None, "last step is unavailable"
            receipt = receipts[-1]
        else:
            index = int(index_text)
            if index > len(receipts):
                return False, None, f"step {index} has not completed"
            receipt = receipts[index - 1]
        if not receipt.get("ok"):
            return False, None, f"referenced step {receipt.get('index')} did not verify"
        path = match.group("path")
        found, value = cls._lookup_result_path(receipt, path)
        if not found:
            return False, None, f"referenced result path '{path}' is unavailable"
        return True, value, ""

    @classmethod
    def _resolve_step_references(
        cls,
        value: Any,
        receipts: list[dict[str, Any]],
    ) -> tuple[bool, Any, str]:
        if isinstance(value, dict):
            resolved: dict[str, Any] = {}
            for key, item in value.items():
                ok, replacement, error = cls._resolve_step_references(item, receipts)
                if not ok:
                    return False, value, error
                resolved[key] = replacement
            return True, resolved, ""
        if isinstance(value, list):
            resolved_items: list[Any] = []
            for item in value:
                ok, replacement, error = cls._resolve_step_references(item, receipts)
                if not ok:
                    return False, value, error
                resolved_items.append(replacement)
            return True, resolved_items, ""
        if not isinstance(value, str):
            return True, value, ""

        matches = list(cls._STEP_REFERENCE_PATTERN.finditer(value))
        if not matches:
            return True, value, ""
        if len(matches) == 1 and matches[0].span() == (0, len(value)):
            ok, replacement, error = cls._resolve_step_reference_token(matches[0], receipts)
            if not ok:
                return False, value, error
            return True, replacement, ""

        resolved_text = value
        for match in reversed(matches):
            ok, replacement, error = cls._resolve_step_reference_token(match, receipts)
            if not ok:
                return False, value, error
            if isinstance(replacement, (dict, list)):
                replacement_text = json.dumps(replacement, ensure_ascii=False)
            else:
                replacement_text = str(replacement)
            start, end = match.span()
            resolved_text = resolved_text[:start] + replacement_text + resolved_text[end:]
        return True, resolved_text, ""

    @classmethod
    def _resolve_step_target(
        cls,
        step: DesktopTaskStep,
        receipts: list[dict[str, Any]],
    ) -> tuple[bool, DesktopTaskStep, str]:
        ok, target, error = cls._resolve_step_references(step.target, receipts)
        if not ok:
            return False, step, error
        if isinstance(target, list):
            target = json.dumps(target, ensure_ascii=False)
        elif not isinstance(target, (str, dict)):
            target = str(target)
        if target == step.target:
            return True, step, ""
        return True, step.model_copy(update={"target": target}), ""

    @staticmethod
    def _emit_progress(
        *,
        index: int,
        total: int,
        action: str,
        state: str,
        detail: str,
        level: str = "info",
    ) -> None:
        try:
            from core.thought_stream import get_emitter

            get_emitter().emit(
                "Desktop Task",
                f"Step {index}/{total} {action}: {state}. {detail[:240]}",
                level=level,
                category="ToolExecution",
                step_index=index,
                step_total=total,
                action=action,
                state=state,
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "desktop_task",
                exc,
                action="continued verified desktop execution without neural-stream progress telemetry",
                severity="warning",
            )

    @staticmethod
    def _digest_payload(payload: Any) -> str:
        try:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        except (TypeError, ValueError):
            encoded = str(payload).encode("utf-8", errors="replace")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    async def _emit_durable_step_receipt(
        cls,
        receipt: dict[str, Any],
        *,
        objective: str,
        planner: str,
        tool: str = "computer_use",
    ) -> None:
        try:
            from core.runtime.receipts import ToolExecutionReceipt, get_receipt_store

            store = get_receipt_store()
            durable = ToolExecutionReceipt(
                cause=str(objective or "desktop_task")[:240],
                tool=tool,
                governance_receipt_id=str(
                    (receipt.get("result") or {}).get("governance_receipt_id")
                    or (receipt.get("result") or {}).get("authority_receipt_id")
                    or ""
                )
                or None,
                capability_receipt_id=str(
                    (receipt.get("result") or {}).get("capability_receipt_id") or ""
                )
                or None,
                status="success_verified" if receipt.get("ok") else "failed",
                output_digest=cls._digest_payload(receipt.get("result") or {}),
                verification_evidence={
                    "step_index": receipt.get("index"),
                    "action": receipt.get("action"),
                    "critical": receipt.get("critical", True),
                    "effect_verified": receipt.get("effect_verified"),
                    "effect_evidence": receipt.get("effect_evidence"),
                    "planner": planner,
                    "attempts": receipt.get("attempts", 0),
                },
            )
            emitted = await asyncio.to_thread(store.emit, durable)
            receipt["durable_receipt_id"] = emitted.receipt_id
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
            record_degradation(
                "desktop_task",
                exc,
                action="continued desktop task after durable step receipt emission failed",
                severity="warning",
            )

    @classmethod
    def _verify_step_effect(cls, step: DesktopTaskStep, result: dict[str, Any]) -> tuple[bool, str]:
        if not result.get("ok"):
            return False, str(result.get("error") or result.get("status") or "child action reported failure")

        action = step.action
        payload = cls._target_payload(step.target)
        if action == "create_folder":
            path = str(result.get("path") or "").strip()
            verified = bool(path) and bool(result.get("effect_verified"))
            return (
                verified,
                f"folder_path={path};verified=true"
                if verified
                else str(result.get("verification") or "missing confirmed folder path"),
            )
        if action == "open_app":
            opened = str(result.get("opened") or "").strip()
            frontmost = str(result.get("frontmost_app") or "").strip()
            verified = bool(result.get("effect_verified")) and bool(opened) and bool(frontmost)
            return (
                verified,
                f"opened={opened};frontmost={frontmost}"
                if verified
                else str(result.get("verification") or "missing frontmost app confirmation"),
            )
        if action == "open_url":
            url = str(result.get("url") or "").strip()
            valid_url = url.startswith(("http://", "https://"))
            frontmost = str(result.get("frontmost_app") or "").strip()
            verified = valid_url and bool(result.get("effect_verified")) and bool(frontmost)
            if verified and bool(payload.get("requires_editable_focus")):
                verified = bool(
                    result.get("doc_focused")
                    or result.get("editable_focus_verified")
                )
                if not verified:
                    focus_error = str(
                        result.get("focus_error")
                        or result.get("verification")
                        or "editable document focus was not verified"
                    )
                    return False, focus_error
            return (
                verified,
                f"url={url};frontmost={frontmost}"
                if verified
                else str(result.get("verification") or "missing browser foreground confirmation"),
            )
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
            digest = str(result.get("sha256") or "").strip()
            verified = bool(result.get("effect_verified")) and cls._valid_sha256(digest)
            return (
                verified,
                f"path={path};bytes={bytes_written};sha256={digest}"
                if verified
                else str(result.get("verification") or "missing file content read-back"),
            )
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
            digest = str(result.get("sha256") or "").strip()
            verified = bool(result.get("effect_verified")) and cls._valid_sha256(digest)
            return (
                verified,
                f"path={path};bytes={bytes_written};pages={pages};chars={chars};sha256={digest}"
                if verified
                else str(result.get("verification") or "missing persisted PDF verification"),
            )
        if action == "fetch_topic_image":
            img_path = str(result.get("path") or "").strip()
            img_bytes = result.get("bytes")
            page_url = str(result.get("page_url") or "").strip()
            if not img_path:
                return False, "missing fetched image path"
            if not isinstance(img_bytes, int) or img_bytes <= 0:
                return False, "missing fetched image byte count"
            digest = str(result.get("sha256") or "").strip()
            verified = bool(result.get("effect_verified")) and cls._valid_sha256(digest)
            return (
                verified,
                f"path={img_path};bytes={img_bytes};source={page_url};sha256={digest}"
                if verified
                else str(result.get("verification") or "missing downloaded image read-back"),
            )
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
            verified = bool(result.get("effect_verified"))
            return (
                verified,
                f"destination={destination};bytes={bytes_moved};verified=true"
                if verified
                else str(result.get("verification") or "missing move postcondition"),
            )
        if action == "set_clipboard":
            chars = result.get("chars")
            if not isinstance(chars, int) or chars < 0:
                return False, "missing clipboard character count"
            digest = str(result.get("sha256") or "").strip()
            verified = bool(result.get("effect_verified")) and cls._valid_sha256(digest)
            return (
                verified,
                f"clipboard_chars={chars};sha256={digest}"
                if verified
                else str(result.get("verification") or "missing exact clipboard read-back"),
            )
        if action == "get_clipboard":
            chars = result.get("chars")
            text = result.get("text")
            if not isinstance(chars, int) or chars < 0 or not isinstance(text, str):
                return False, "missing clipboard readback evidence"
            return True, f"clipboard_read_chars={chars}"
        if action == "read_menu_clock":
            clock_text = str(result.get("clock_text") or result.get("text") or "").strip()
            source = str(result.get("source") or "").strip()
            if not clock_text:
                return False, "missing menu clock readback"
            return True, f"clock_text={clock_text[:80]};source={source or 'unknown'}"
        if action == "run_command":
            exit_code = result.get("exit_code")
            if not isinstance(exit_code, int):
                return False, "missing command exit code"
            if exit_code != 0:
                return False, f"command exited {exit_code}"
            output = str(result.get("output") or "")
            return True, f"exit_code=0;output_chars={len(output)}"
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
            expected_frontmost = str(result.get("expected_frontmost_app") or "").strip()
            is_paste = bool(result.get("is_paste"))
            verified = (
                bool(result.get("effect_verified"))
                or "state shifted" in verification.lower()
                or "focused element changed" in verification.lower()
            )
            if is_paste and expected_frontmost:
                clipboard_check = result.get("clipboard_payload_verification")
                clipboard_check = (
                    clipboard_check if isinstance(clipboard_check, dict) else {}
                )
                target_ok = bool(result.get("write_target_app_verified"))
                clipboard_ok = bool(clipboard_check.get("verified")) or not clipboard_check
                if not target_ok:
                    return False, "paste target app was not verified"
                if not clipboard_ok:
                    return False, "paste clipboard payload was not verified"
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
        if action == "run_applescript":
            if not bool(result.get("effect_verified")):
                return False, "AppleScript transport output is not objective-specific effect evidence"
            verification_results = result.get("verification_results")
            if not isinstance(verification_results, list):
                return False, "missing structured AppleScript verification results"
            strong_passed = any(
                isinstance(check, dict)
                and bool(check.get("passed"))
                and bool(check.get("strong", True))
                for check in verification_results
            )
            if not strong_passed:
                return False, "missing strong AppleScript postcondition"
            evidence = str(result.get("effect_evidence") or "").strip()
            if not evidence:
                return False, "missing AppleScript effect evidence summary"
            return True, evidence[:240]
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
        if action in {"inspect_screen", "read_screen_text"}:
            text = str(result.get("text") or "").strip()
            active_app = str(result.get("active_app") or "").strip()
            if text:
                detail = "screen_text_returned"
                if active_app:
                    detail = f"{detail};frontmost_app={active_app}"
                return True, detail
            if action == "inspect_screen" and active_app:
                return True, f"frontmost_app={active_app}"
            return (False, "missing screen text evidence")
        return False, f"unsupported effect evidence for desktop action {action}"

    @staticmethod
    def _inline_sentence_for(objective: str) -> str:
        """Content for a file whose body the user left to her.

        "containing one sentence you choose" is a real instruction with no
        text attached, and writing an empty file would satisfy the letter of
        it while failing the request.
        """
        text = str(objective or "")
        quoted = re.search(r"[\"“‘']([^\"”’']{3,400})[\"”’']", text)
        if quoted:
            return quoted.group(1).strip() + "\n"
        # "containing one sentence you choose" leaves the content to her. The
        # first attempt echoed the instruction itself into the file — "one
        # sentence you choose. Actually execute it, then tell me the full
        # path." — which is the request, not an answer to it.
        if re.search(
            r"\b(?:you\s+choose|of\s+your\s+choosing|whatever\s+you\s+(?:like|want)|"
            r"anything\s+you\s+(?:like|want)|up\s+to\s+you)\b",
            text,
            re.IGNORECASE,
        ):
            return (
                "Written by Aura, through the governed desktop file lane — "
                "the sentence is mine, since you left it to me.\n"
            )
        return ""

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

    @staticmethod
    def _writing_app_from_apps(apps: list[str]) -> str:
        for app in apps:
            if app in {"Notes", "TextEdit", "Pages", "Microsoft Word"}:
                return app
        return ""

    @staticmethod
    def _step_opens_app(step: DesktopTaskStep, app: str) -> bool:
        return step.action == "open_app" and str(step.target or "").strip() == app

    @classmethod
    def _sequenced_objective_segments(cls, objective: str) -> list[str]:
        """Split only explicit discourse-level sequencing markers.

        A single heuristic plan cannot safely keep focus across independent
        work products. Continuations of the same product stay together so
        research-to-document and compose-to-export chains retain shared state.
        """
        text = str(objective or "").strip()
        if not text:
            return []
        parts = re.split(
            r"(?:[.!?;]\s+|,\s+)(?:and\s+)?(?:then|after that|next|finally|lastly|"
            r"also|i\s+also\s+(?:want|need|would\s+like)\s+to|can\s+you|could\s+you|would\s+you)\s*,?\s*",
            text,
            flags=re.IGNORECASE,
        )
        if len(parts) <= 1:
            return [text]
        candidates = [part.strip(" \t\r\n,.;") for part in parts if part.strip(" \t\r\n,.;")]
        if len(candidates) <= 1:
            return [text]

        def _surfaces(value: str) -> set[str]:
            surfaces = {app.lower() for app in cls._extract_apps(value)}
            web_surface = cls._web_document_url(value)
            if web_surface:
                surfaces.add(web_surface)
            lowered_value = value.lower()
            if any(
                token in lowered_value
                for token in ("search", "look up", "article", "articles", "news", "source", "sources")
            ) or re.search(r"\bread\s+(?:about|on)\b", lowered_value):
                surfaces.add("web_research")
            if cls._extract_image_query(value):
                surfaces.add("image_search")
            for domain, _ in detect_os_settings(value):
                surfaces.add(f"os_setting:{domain}")
            return surfaces

        def _completes_product(value: str) -> bool:
            lowered = value.lower()
            return bool(
                re.search(r"\b(?:export|save|render)\b[^.;\n]{0,80}\b(?:pdf|file|document|artifact)\b", lowered)
                or (
                    re.search(r"\b(?:write|compose|draft|create)\b", lowered)
                    and any(token in lowered for token in ("note", "document", "essay", "report", "summary"))
                    and bool(_surfaces(value))
                )
            )

        segments = [candidates[0]]
        for candidate in candidates[1:]:
            previous = segments[-1]
            starts_distinct_surface = bool(_surfaces(candidate) - _surfaces(previous))
            if _completes_product(previous) and starts_distinct_surface:
                segments.append(candidate)
            else:
                segments[-1] = f"{previous}. Then {candidate}"
        return segments

    @staticmethod
    def _has_explicit_folder_name(objective: str) -> bool:
        text = str(objective or "")
        return bool(
            re.search(
                r"\b(?:folder|directory)\s+(?:named|called|titled)\b",
                text,
                flags=re.IGNORECASE,
            )
            or re.search(
                r"(?:\"[^\"]+\"|'(?:[^']|'(?=\w))+')\s+(?:folder|directory)\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _inherit_shared_destination(cls, segment: str, objective: str) -> str:
        """Resolve a later phase's explicit "same folder" reference.

        This carries only a destination the user explicitly named. It does not
        invent cross-phase state or make every artifact share one directory.
        """
        if not re.search(
            r"\b(?:same|that|the previously (?:named|created))\b[^.\n]{0,80}\b(?:folder|directory)\b",
            segment,
            flags=re.IGNORECASE,
        ):
            return segment
        if cls._has_explicit_folder_name(segment) or not cls._has_explicit_folder_name(objective):
            return segment

        folder_name = cls._extract_folder_name(objective)
        root_hint = cls._extract_root_hint(objective)
        root_phrase = {
            "~/Desktop": " on my Desktop",
            "~/Documents": " in my Documents folder",
            "~/Downloads": " in my Downloads folder",
        }.get(root_hint, "")
        return (
            f'{segment.rstrip(" .")}, using the folder titled '
            f'"{folder_name}"{root_phrase}.'
        )

    @classmethod
    def _deduplicate_segment_artifact_paths(
        cls,
        segment_steps: list[DesktopTaskStep],
        used_paths: set[str],
    ) -> list[DesktopTaskStep]:
        """Give each phase distinct durable outputs inside shared folders."""
        resolved: list[DesktopTaskStep] = []
        for step in segment_steps:
            if step.action not in {"write_text_file", "render_text_pdf"}:
                resolved.append(step)
                continue
            payload = cls._target_payload(step.target)
            path = str(payload.get("path") or "").strip()
            if not path:
                resolved.append(step)
                continue
            if path in used_paths:
                candidate = Path(path)
                path = next(
                    str(candidate.with_name(f"{candidate.stem}_{index}{candidate.suffix}"))
                    for index in range(2, 42)
                    if str(candidate.with_name(f"{candidate.stem}_{index}{candidate.suffix}"))
                    not in used_paths
                )
            used_paths.add(path)
            payload["path"] = path
            resolved.append(step.model_copy(update={"target": payload}))
        return resolved

    def _derive_steps_from_objective(
        self,
        objective: str,
        context: dict[str, Any] | None,
    ) -> list[DesktopTaskStep]:
        """Derive a focus-safe plan for one or more explicit task phases."""
        segments = self._sequenced_objective_segments(objective)
        if len(segments) <= 1:
            return self._derive_single_objective_steps(objective, context)

        steps: list[DesktopTaskStep] = []
        created_folders: set[str] = set()
        used_artifact_paths: set[str] = set()
        global_preferred_browser = self._preferred_browser(objective)
        for segment in segments:
            resolved_segment = self._inherit_shared_destination(segment, objective)
            if (
                global_preferred_browser
                and not self._preferred_browser(resolved_segment)
                and (
                    self._extract_search_query(resolved_segment)
                    or self._extract_image_query(resolved_segment)
                    or self._web_document_url(resolved_segment)
                    or any(token in resolved_segment.lower() for token in ("browser", "web", "article", "source", "news"))
                )
            ):
                resolved_segment = f"{resolved_segment.rstrip(' .')}, using {global_preferred_browser}."
            segment_steps = self._derive_single_objective_steps(resolved_segment, context)
            segment_steps = self._deduplicate_segment_artifact_paths(
                segment_steps,
                used_artifact_paths,
            )
            for step in segment_steps:
                if step.action == "create_folder":
                    folder_path = str(self._target_payload(step.target).get("path") or step.target)
                    if folder_path in created_folders:
                        continue
                    created_folders.add(folder_path)
                steps.append(step)
        return steps

    def _derive_single_objective_steps(
        self,
        objective: str,
        context: dict[str, Any] | None,
    ) -> list[DesktopTaskStep]:
        text = str(objective or "").strip()
        lowered = text.lower()
        steps: list[DesktopTaskStep] = []
        # A named file is an unambiguous instruction, and it has to be read
        # before the folder heuristics get a vote.
        #
        # Live 2026-07-27: "create a file on my Desktop called aura_hello.txt
        # containing one sentence you choose" produced a create_folder step
        # named "Aura Desktop Task 1785195330". The folder was really created,
        # so the task reported 1/1 steps completed — a true receipt for the
        # wrong action, which is worse than a failure: she then told the user
        # the objective had completed, and the only thing on the Desktop was
        # a junk folder. The word "file" was right there in the request.
        named_paths = extract_target_paths(text)
        if named_paths and not any(
            token in lowered for token in ("folder", "directory")
        ):
            body = self._inline_sentence_for(text) or self._document_body(text, context)
            return [
                DesktopTaskStep(
                    action="write_text_file",
                    target=json.dumps({"path": named_paths[0], "content": body, "overwrite": True}),
                    reason="The request names a file to create.",
                    expect=f"{named_paths[0]} exists on disk with the requested content.",
                    critical=True,
                )
            ]

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
        image_reference_only = bool(image_query) and not any(
            token in lowered
            for token in (
                "article",
                "articles",
                "news",
                "research",
                "report",
                "reports",
                "sources",
            )
        )
        wants_search = (not image_reference_only) and (
            any(token in lowered for token in ("search", "look up", "news", "article"))
            or ("google" in lowered and not web_document_url)
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

        def _open_url_target(url: str, *, requires_editable_focus: bool = False):
            if preferred_browser:
                payload = {"url": url, "browser": preferred_browser}
                if requires_editable_focus:
                    payload["requires_editable_focus"] = True
                return payload
            if requires_editable_focus:
                return {"url": url, "requires_editable_focus": True}
            return url

        query = self._extract_search_query(text)
        search_url = self._search_url(query, engine=engine_hint) if query else ""
        image_search_surface_deferred = False
        if wants_search and query:
            steps.append(
                DesktopTaskStep(
                    action="open_url",
                    target=_open_url_target(search_url),
                    reason="Open a browser/search tab for the requested live research topic.",
                    expect=f"{browser_label} accepts the search URL.",
                )
            )
            visible_source_count = self._requested_visible_source_count(text)
            if visible_source_count > 0:
                opened_source_urls: set[str] = set()
                for source in (context or {}).get("desktop_task_research_sources") or []:
                    if not isinstance(source, dict):
                        continue
                    source_url = str(source.get("url") or source.get("link") or "").strip()
                    if (
                        not source_url.startswith(("http://", "https://"))
                        or source_url in opened_source_urls
                        or source_url == search_url
                    ):
                        continue
                    opened_source_urls.add(source_url)
                    steps.append(
                        DesktopTaskStep(
                            action="open_url",
                            target=_open_url_target(source_url),
                            reason="Open one governed research source so the user can inspect the evidence visibly.",
                            expect=f"{browser_label} accepts the research source URL.",
                        )
                    )
                    if len(opened_source_urls) >= visible_source_count:
                        break
        if web_document_url:
            steps.append(
                DesktopTaskStep(
                    action="open_url",
                    target=_open_url_target(
                        web_document_url,
                        requires_editable_focus=wants_interactive_text_entry,
                    ),
                    reason="Open the requested web document surface.",
                    expect=f"{browser_label} accepts the document URL.",
                )
            )
        image_search_url = (
            self._search_url(image_query or text, images=True, engine=engine_hint)
            if wants_image
            else ""
        )
        open_image_search_surface = bool(
            image_search_url
            and image_search_url != search_url
            and not (wants_artifact_file and image_query)
        )
        if open_image_search_surface:
            steps.append(
                DesktopTaskStep(
                    action="open_url",
                    target=_open_url_target(image_search_url),
                    reason="Open an image-search surface for the requested visual reference.",
                    expect=f"{browser_label} accepts the image search URL.",
                )
            )
        elif image_search_url and image_search_url != search_url:
            image_search_surface_deferred = True

        if wants_interactive_text_entry:
            body = self._document_body_with_references(
                text,
                context,
                image_query=image_query,
                image_search_url=image_search_url,
                search_url=search_url,
            )
            writing_app = "" if web_document_url else self._writing_app_from_apps(apps)
            if writing_app and not (
                steps and self._step_opens_app(steps[-1], writing_app)
            ):
                steps.append(
                    DesktopTaskStep(
                        action="open_app",
                        target=writing_app,
                        reason=(
                            f"Re-focus {writing_app} immediately before text entry so "
                            "browser/image/search tabs cannot steal the paste target."
                        ),
                        expect=f"{writing_app} is frontmost before writing.",
                    )
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
            if (not web_document_url) and any(
                marker in lowered for marker in ("note", "textedit", "pages", "word", "document", "journal")
            ):
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
                        target=_open_url_target(FETCHED_IMAGE_SOURCE_SENTINEL),
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
                filename_stem = self._artifact_filename_stem(text)
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
        if image_search_surface_deferred and not self._wants_image_source_shown(text):
            # Leave the core artifact chain uninterrupted. The verified image
            # download receipt is the proof that the visual came from the web;
            # a browser tab for image search is useful ambience, but it must not
            # steal focus before Notes/Docs writing or block the PDF if Chrome
            # cannot semantically confirm its active tab.
            steps.append(
                DesktopTaskStep(
                    action="open_url",
                    target=_open_url_target(image_search_url),
                    reason="Optionally leave the image-search surface visible after the requested writing/PDF artifact is complete.",
                    expect=f"{browser_label} accepts the image search URL.",
                    critical=False,
                )
            )
        if image_query and wants_artifact_file and self._wants_image_source_shown(text):
            steps.append(
                DesktopTaskStep(
                    action="open_url",
                    target=_open_url_target(FETCHED_IMAGE_SOURCE_SENTINEL),
                    reason="Show the user where the fetched image was found after the artifact has been created.",
                    expect=f"{browser_label} accepts the fetched image source page URL.",
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
        return steps

    @staticmethod
    def _primitive_steps_are_only_observational(steps: list[DesktopTaskStep]) -> bool:
        if not steps:
            return True
        non_effect_actions = {"inspect_screen", "read_screen_text", "wait", "get_clipboard"}
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

    @staticmethod
    def _steps_cover_durable_artifact_intent(objective: str, steps: list[DesktopTaskStep]) -> bool:
        """Prefer verified primitives for file/document/PDF objectives.

        Free-form OS automation is useful for true window/UI manipulation,
        but it is the least deterministic lane. If the planner already
        derived a bounded artifact plan with read-backable effects, do not
        discard it merely because the natural-language objective also says
        "click", "copy", or "type".
        """
        lowered = str(objective or "").lower()
        actions = {step.action for step in steps}
        if not actions:
            return False
        # An objective that names a concrete path IS a durable-artifact
        # objective, whatever else the sentence says. Without this, "create a
        # file on my Desktop called aura_hello.txt containing one sentence you
        # choose" matched none of the document tokens below, so the derived
        # write_text_file plan was discarded and the turn was escalated to
        # AppleScript on the strength of the word "choose" — where a file has
        # no observable postcondition and the objective was refused outright.
        # Same path extractor the effect contract uses, so router and verifier
        # cannot disagree about what the objective is about.
        target_paths = extract_target_paths(objective)
        wants_file = bool(target_paths) or bool(re.search(r"\bfiles?\b", lowered))
        wants_folder = any(token in lowered for token in ("folder", "directory"))
        wants_document = any(
            token in lowered
            for token in (
                "write",
                "summary",
                "summarize",
                "note",
                "document",
                "doc",
                "pdf",
                "save",
                "journal",
                "essay",
                "report",
                "artifact",
            )
        )
        wants_pdf = "pdf" in lowered or bool(
            re.search(r"\b(?:export|save)\b[^.\n]{0,60}\bas\s+(?:a\s+)?pdf\b", lowered)
        )
        if not (wants_folder or wants_document or wants_pdf or wants_file):
            return False
        if wants_file and not actions & {
            "write_text_file",
            "render_text_pdf",
            "move_file",
            "create_folder",
        }:
            return False
        if wants_folder and "create_folder" not in actions:
            return False
        if wants_document and not actions & {"write_text_file", "set_clipboard", "render_text_pdf"}:
            return False
        if wants_pdf and "render_text_pdf" not in actions:
            return False
        if "image" in lowered and "fetch_topic_image" not in actions and "open_url" not in actions:
            return False
        return any(
            action in actions
            for action in ("create_folder", "write_text_file", "render_text_pdf", "move_file", "fetch_topic_image")
        )

    @staticmethod
    def _steps_cover_visible_writing_intent(objective: str, steps: list[DesktopTaskStep]) -> bool:
        """Keep visible writing chains on verified primitives.

        Mixed browser/native requests are common live-demo and daily-use tasks.
        Escalating them to one generated AppleScript blob removes per-step focus
        evidence and was the source of URL-bar pastes. If the derived primitive
        plan already opens the requested surfaces, stages text, and performs a
        paste/type action, keep the chain auditable.
        """
        lowered = str(objective or "").lower()
        if not re.search(r"\b(?:write|type|paste|compose|draft|summari[sz]e|note|doc|document)\b", lowered):
            return False
        actions = [step.action for step in steps]
        action_set = set(actions)
        if not action_set:
            return False
        opens_surface = bool(action_set & {"open_app", "open_url"})
        stages_text = "set_clipboard" in action_set
        commits_text = "type" in action_set or any(
            step.action == "hotkey" and "v" in str(step.target).lower()
            for step in steps
        )
        return opens_surface and stages_text and commits_text

    @staticmethod
    def _objective_requires_true_window_automation(objective: str) -> bool:
        return bool(
            re.search(
                r"\b(?:arrange|resize|drag|minimi[sz]e|maximi[sz]e|organize|tile|snap)\b",
                str(objective or ""),
                flags=re.IGNORECASE,
            )
        )

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
        # Multi-surface/multi-app objectives require coordinated focus and
        # clipboard control. Prefer verified primitives when they already cover
        # the durable artifact and UI intent; escalate only when coverage is
        # incomplete. This keeps demo-class tasks in the receipt-producing lane
        # instead of hiding them behind a single generated AppleScript blob.
        if cls._steps_cover_durable_artifact_intent(objective, steps) and cls._steps_cover_general_os_intent(objective, steps):
            return False
        lowered_obj = objective.lower()
        has_browser_or_web = any(
            re.search(rf"\b{re.escape(marker)}s?\b", lowered_obj)
            for marker in ("chrome", "browser", "safari", "web", "url", "doc", "sheet", "slide", "drive", "notion")
        )
        has_local_editor_or_file = any(
            re.search(rf"\b{re.escape(marker)}s?\b", lowered_obj)
            for marker in ("note", "textedit", "pages", "word", "document", "folder", "desktop", "journal", "file")
        )
        if has_browser_or_web and has_local_editor_or_file:
            if (
                cls._steps_cover_visible_writing_intent(objective, steps)
                and not cls._objective_requires_true_window_automation(objective)
            ):
                return False
            return True
        if cls._objective_needs_general_os_automation(objective) and not any(
            step.action == "run_applescript" for step in steps
        ):
            if (
                cls._steps_cover_durable_artifact_intent(objective, steps)
                and not cls._objective_requires_true_window_automation(objective)
            ):
                return False
            if cls._steps_cover_general_os_intent(objective, steps):
                return False
            return looks_like_desktop_objective(objective) or any(
                step.action in {"open_app", "open_url"} for step in steps
            )
        if not cls._primitive_steps_are_only_observational(steps):
            return False
        return looks_like_desktop_objective(objective)



    # A step proves the step's effect. The task proves the task's.
    #
    # Child contexts were built with `dict(task_context)`, which carries the
    # task-level action expectation down into every step — so a `create_folder`
    # step was required to produce `steps_requested` and `steps_completed`,
    # fields only the task result has. It could not, so the contract layer
    # failed it with "expectation incomplete: steps_requested; steps_completed"
    # and the whole objective died on a step that had, in fact, worked.
    # Measured live 2026-07-27 on "create a file on my Desktop".
    #
    # No step of any desktop objective can satisfy a task-level contract, so
    # this was never one action misbehaving — it was every multi-step desktop
    # objective inheriting a contract its parts cannot meet.
    _TASK_LEVEL_EXPECTATION_KEYS: tuple[str, ...] = (
        "action_expectation",
        "expectation",
        "acceptance_criteria",
        "criteria",
        "required_evidence",
        "evidence_required",
        "required_evidence_present",
        "user_visible_effect",
        "visible_effect",
        "repair_hint",
        "rollback_hint",
        "allow_partial",
    )

    @classmethod
    def _child_step_context(cls, task_context: dict[str, Any] | None) -> dict[str, Any]:
        """A step's context, without the contract that belongs to the task."""
        child = dict(task_context or {})
        for key in cls._TASK_LEVEL_EXPECTATION_KEYS:
            child.pop(key, None)
        return child

    @staticmethod
    def _failure_cause(failures: list[dict[str, Any]], *, objective: str = "") -> str:
        """Why the desktop task failed, in the words of the step that failed.

        Every failing receipt already knows: the step's action, what it expected,
        the effect evidence, and the child result's own error. None of that was
        lifted into the skill's `error` field, so BaseSkill fell back to
        "desktop_task reported failure without a cause (status=failed)" — which
        is what reached Bryan, twice, for "create a file on my Desktop". An
        undiagnosable failure is barely better than a silent one: he cannot act
        on it, she cannot explain it, and the surprise engine banks a
        maximal-surprise signal carrying no information.
        """
        for receipt in failures or []:
            if not isinstance(receipt, dict):
                continue
            result = receipt.get("result")
            detail = ""
            if isinstance(result, dict):
                detail = str(
                    result.get("error") or result.get("status") or result.get("reason") or ""
                ).strip()
            if not detail:
                detail = str(receipt.get("effect_evidence") or "").strip()
            action = str(receipt.get("action") or "step").strip()
            expected = str(receipt.get("expect") or "").strip()
            if detail:
                suffix = f" (expected: {expected})" if expected else ""
                return f"{action} failed: {detail}{suffix}"[:400]
            if expected:
                return f"{action} did not produce its expected effect: {expected}"[:400]
            return f"{action} failed without reporting why"
        if objective:
            return (
                "no step reported a failure, yet the objective was not verified as "
                f"complete: {objective[:160]}"
            )
        return "the desktop task did not complete and no step reported a cause"

    @staticmethod
    def _os_automation_effect_evidence(result: dict[str, Any]) -> tuple[bool, str]:
        if not bool(result.get("ok")):
            return False, str(result.get("error") or result.get("status") or "os automation reported failure")
        if not bool(result.get("effect_verified")):
            return False, "os automation did not verify the requested effect"
        contract = result.get("effect_contract")
        if not isinstance(contract, dict) or not bool(contract.get("verifiable")):
            return False, "missing verifiable os automation effect contract"
        checks = result.get("verification_results")
        if not isinstance(checks, list) or not checks:
            return False, "missing structured os automation verification checks"
        failed_required = any(
            isinstance(check, dict)
            and bool(check.get("required", True))
            and not bool(check.get("passed"))
            for check in checks
        )
        strong_passed = any(
            isinstance(check, dict)
            and bool(check.get("passed"))
            and bool(check.get("strong", True))
            for check in checks
        )
        if failed_required or not strong_passed:
            return False, "os automation checks do not prove every required strong effect"
        effect_evidence = str(result.get("effect_evidence") or "").strip()
        if effect_evidence and not effect_evidence.startswith("receipt_id="):
            return True, effect_evidence[:240]
        receipt_id = str(result.get("receipt_id") or "").strip()
        if receipt_id:
            return False, (
                f"receipt_id={receipt_id} is audit evidence only; missing observable "
                "verification proving the requested desktop effect."
            )
        return False, "missing objective-specific os automation effect evidence"

    async def _execute_os_automation_escalation(
        self,
        *,
        capability_engine: Any,
        objective: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        step_context = self._child_step_context(context)
        document_body = (
            self._document_body(objective, step_context)
            if self._objective_requests_written_artifact(objective)
            else ""
        )
        step_context.update(
            {
                "origin": step_context.get("origin") or "desktop_task",
                "route": "desktop_task.os_automation",
                "objective": objective,
                "foreground_request": True,
                "user_requested_action": True,
                "user_explicitly_authorized": True,
                "user_visible_desktop_action": True,
                "local_desktop_action": True,
                "desktop_task_reason": (
                    "Primitive desktop actions were not sufficient for this objective; "
                    "escalating to governed OS automation."
                ),
                "desktop_task_expect": (
                    "OS automation returns a verifiable effect contract with every required "
                    "strong objective-specific check passed."
                ),
                "desktop_task_document_body": document_body,
                "document_body": document_body,
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
            **({} if ok else {"error": self._failure_cause([receipt], objective=objective)}),
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
        task_context.setdefault("objective", objective)
        steps = list(params.steps)
        planner = "explicit_steps" if steps else ""
        if not steps:
            plan_error = self._declared_plan_validation_error(task_context)
            if plan_error:
                return {
                    "ok": False,
                    "status": "invalid_desktop_task_plan",
                    "error": plan_error,
                    "objective": objective,
                    "steps_requested": 0,
                    "steps_completed": 0,
                    "receipts": [],
                    "failures": [],
                }
            steps, planner = self._steps_with_provenance_from_context(task_context)
        requires_structured_plan = bool(task_context.get("desktop_execution_contract")) and not bool(
            task_context.get("allow_heuristic_desktop_plan")
        )
        if not steps and requires_structured_plan:
            return {
                "ok": False,
                "status": "desktop_task_plan_required",
                "error": (
                    "The live CognitiveEngine response did not contain a valid structured "
                    "desktop plan, so no desktop action was attempted."
                ),
                "objective": objective,
                "steps_requested": 0,
                "steps_completed": 0,
                "receipts": [],
                "failures": [],
                "planner": "required_cognitive_plan_missing",
            }

        research_context = await self._collect_research_context(
            capability_engine=capability_engine,
            objective=objective,
            context=task_context,
        )
        if research_context:
            task_context.update(research_context)
            if task_context.get("desktop_task_research_error") and self._objective_requests_research_document(objective):
                failure_receipt = {
                    "index": 0,
                    "action": "web_search",
                    "ok": False,
                    "critical": True,
                    "effect_verified": False,
                    "effect_evidence": str(task_context.get("desktop_task_research_error") or ""),
                    "result": {
                        "query": task_context.get("desktop_task_research_query"),
                        "deep": task_context.get("desktop_task_research_deep"),
                        "pressure_limited": task_context.get("desktop_task_research_pressure_limited"),
                    },
                }
                await self._emit_durable_step_receipt(
                    failure_receipt,
                    objective=objective,
                    planner=planner or "research_preflight",
                    tool="web_search",
                )
                return {
                    "ok": False,
                    "status": "desktop_task_research_unavailable",
                    "error": str(task_context.get("desktop_task_research_error") or "research evidence unavailable"),
                    "objective": objective,
                    "steps_requested": 0,
                    "steps_completed": 0,
                    "receipts": [],
                    "failures": [failure_receipt],
                    "planner": planner or "research_preflight",
                    "research": {
                        "query": task_context.get("desktop_task_research_query"),
                        "sources": [],
                        "error": task_context.get("desktop_task_research_error"),
                    },
                }
        document_provenance = "cognitive_context"
        if self._objective_requests_self_summary(objective):
            authored = self._self_summary_from_context(task_context)
            if not authored and self._allow_desktop_task_model_synthesis(task_context):
                authored = await self._synthesize_self_summary_document(
                    objective=objective,
                    context=task_context,
                )
                if authored:
                    task_context["desktop_task_document_body"] = authored
                    document_provenance = "local_cortex_synthesis"
            if not authored:
                task_context["desktop_task_document_body"] = self._compose_self_summary_body(
                    objective
                )
                document_provenance = "runtime_substrate_synthesis"
        elif task_context.get("desktop_task_research_synthesis"):
            document_provenance = (
                "local_cortex_research_synthesis"
                if self._allow_research_model_synthesis(task_context)
                else "source_grounded_deterministic_synthesis"
            )
        if not steps:
            steps = self._derive_steps_from_objective(objective, task_context)
            planner = "heuristic_compat"
        steps = self._resolve_document_body_tokens(
            steps,
            self._document_body(objective, task_context),
        )
        if len(steps) > MAX_DESKTOP_TASK_STEPS:
            return {
                "ok": False,
                "status": "desktop_task_plan_too_large",
                "error": (
                    f"Desktop task requires {len(steps)} steps, exceeding the "
                    f"{MAX_DESKTOP_TASK_STEPS}-step bounded execution limit."
                ),
                "objective": objective,
                "steps_requested": len(steps),
                "steps_completed": 0,
                "receipts": [],
                "failures": [],
                "planner": planner,
            }

        if planner == "heuristic_compat" and self._should_escalate_to_os_automation(
            objective,
            steps,
            task_context,
        ):
            return await self._execute_os_automation_escalation(
                capability_engine=capability_engine,
                objective=objective,
                context=task_context,
            )

        last_image_page_url = ""
        expected_frontmost_app = ""
        current_surface_requires_editable_focus = False
        expected_clipboard_sha256 = ""
        expected_clipboard_chars: int | None = None
        for index, step in enumerate(steps, start=1):
            references_ok, resolved_step, reference_error = self._resolve_step_target(step, receipts)
            if not references_ok:
                receipt = {
                    "index": index,
                    "action": step.action,
                    "reason": step.reason,
                    "expect": step.expect,
                    "critical": step.critical,
                    "ok": False,
                    "effect_verified": False,
                    "effect_evidence": reference_error,
                    "attempts": 0,
                    "result": {
                        "ok": False,
                        "status": "desktop_step_reference_unresolved",
                        "error": reference_error,
                    },
                }
                receipts.append(receipt)
                await self._emit_durable_step_receipt(
                    receipt,
                    objective=objective,
                    planner=planner,
                    tool="desktop_task",
                )
                failures.append(receipt)
                self._emit_progress(
                    index=index,
                    total=len(steps),
                    action=step.action,
                    state="blocked",
                    detail=reference_error,
                    level="warning",
                )
                if step.critical and params.stop_on_error:
                    break
                continue

            target = resolved_step.target
            if resolved_step.action == "open_url":
                # Resolve the fetched-image source sentinel from the
                # fetch receipt — the source page is only known at runtime.
                if isinstance(target, dict) and target.get("url") == FETCHED_IMAGE_SOURCE_SENTINEL:
                    if not last_image_page_url:
                        reference_error = "no fetched-image source URL available to show"
                        receipt = {
                            "index": index,
                            "action": resolved_step.action,
                            "reason": resolved_step.reason,
                            "expect": resolved_step.expect,
                            "critical": resolved_step.critical,
                            "ok": False,
                            "effect_verified": False,
                            "effect_evidence": reference_error,
                            "attempts": 0,
                            "result": {
                                "ok": False,
                                "status": "desktop_step_reference_unresolved",
                                "error": reference_error,
                            },
                        }
                        receipts.append(receipt)
                        await self._emit_durable_step_receipt(
                            receipt,
                            objective=objective,
                            planner=planner,
                            tool="desktop_task",
                        )
                        failures.append(receipt)
                        if resolved_step.critical and params.stop_on_error:
                            break
                        continue
                    target = dict(target, url=last_image_page_url)
                elif target == FETCHED_IMAGE_SOURCE_SENTINEL:
                    if not last_image_page_url:
                        reference_error = "no fetched-image source URL available to show"
                        receipt = {
                            "index": index,
                            "action": resolved_step.action,
                            "reason": resolved_step.reason,
                            "expect": resolved_step.expect,
                            "critical": resolved_step.critical,
                            "ok": False,
                            "effect_verified": False,
                            "effect_evidence": reference_error,
                            "attempts": 0,
                            "result": {
                                "ok": False,
                                "status": "desktop_step_reference_unresolved",
                                "error": reference_error,
                            },
                        }
                        receipts.append(receipt)
                        await self._emit_durable_step_receipt(
                            receipt,
                            objective=objective,
                            planner=planner,
                            tool="desktop_task",
                        )
                        failures.append(receipt)
                        if resolved_step.critical and params.stop_on_error:
                            break
                        continue
                    target = last_image_page_url
            target_payload = self._target_payload(target)
            if isinstance(target, dict):
                target = json.dumps(target)
            payload = {
                "action": resolved_step.action,
                "target": str(target or ""),
                "x": int(resolved_step.x),
                "y": int(resolved_step.y),
            }
            step_context = self._child_step_context(task_context)
            target_text = str(target or "").lower()
            write_commit_action = (
                resolved_step.action == "type"
                or (
                    resolved_step.action == "hotkey"
                    and (
                        "command" in target_text
                        or "cmd" in target_text
                    )
                    and any(token in target_text for token in ("+v", "+n", "enter", "return"))
                )
            )
            if (
                write_commit_action
                and expected_frontmost_app
            ):
                step_context["desktop_task_expected_frontmost_app"] = expected_frontmost_app
                step_context["desktop_task_write_surface_app"] = expected_frontmost_app
                step_context["desktop_task_prior_verified_frontmost_app"] = expected_frontmost_app
                step_context["desktop_task_allow_unavailable_frontmost_from_prior"] = True
            if write_commit_action and current_surface_requires_editable_focus:
                step_context["desktop_task_requires_editable_focus"] = True
            if (
                resolved_step.action == "hotkey"
                and "v" in target_text
                and ("command" in target_text or "cmd" in target_text)
                and expected_clipboard_sha256
            ):
                step_context["desktop_task_expected_clipboard_sha256"] = expected_clipboard_sha256
                step_context["desktop_task_expected_clipboard_chars"] = expected_clipboard_chars
            step_context.update(
                {
                    "origin": step_context.get("origin") or "desktop_task",
                    "route": "desktop_task.computer_use",
                    "objective": objective,
                    "foreground_request": True,
                    "user_requested_action": True,
                    "user_explicitly_authorized": True,
                    "desktop_task_step": index,
                    "desktop_task_step_total": len(steps),
                    "desktop_task_planner": planner,
                    "desktop_task_reason": resolved_step.reason,
                    "desktop_task_expect": resolved_step.expect,
                }
            )
            self._emit_progress(
                index=index,
                total=len(steps),
                action=resolved_step.action,
                state="starting",
                detail=resolved_step.reason or "Executing governed desktop action.",
            )
            attempt_limit = (
                2 if resolved_step.action in DESKTOP_TASK_RETRY_SAFE_ACTIONS else 1
            )
            attempt = 0
            result: dict[str, Any] = {}
            effect_verified = False
            effect_evidence = "step did not execute"
            while attempt < attempt_limit:
                attempt += 1
                step_context["desktop_task_attempt"] = attempt
                try:
                    result = await capability_engine.execute(
                        "computer_use",
                        payload,
                        context=step_context,
                    )
                except (
                    AttributeError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    OSError,
                    TimeoutError,
                ) as exc:
                    record_degradation(
                        "desktop_task",
                        exc,
                        action="recorded failed desktop step after governed computer_use exception",
                        severity="degraded",
                    )
                    result = {
                        "ok": False,
                        "status": "computer_use_exception",
                        "error": str(exc),
                    }
                if not isinstance(result, dict):
                    result = {"ok": bool(result), "result": result}
                effect_verified, effect_evidence = self._verify_step_effect(
                    resolved_step,
                    result,
                )
                if bool(result.get("ok")) and effect_verified:
                    break
                if attempt < attempt_limit:
                    self._emit_progress(
                        index=index,
                        total=len(steps),
                        action=resolved_step.action,
                        state="retrying",
                        detail=effect_evidence,
                        level="warning",
                    )
                    await asyncio.sleep(0.1)
            receipt = {
                "index": index,
                "action": resolved_step.action,
                "reason": resolved_step.reason,
                "expect": resolved_step.expect,
                "critical": resolved_step.critical,
                "ok": bool(result.get("ok")) and effect_verified,
                "effect_verified": effect_verified,
                "effect_evidence": effect_evidence,
                "attempts": attempt,
                "result": result,
            }
            receipts.append(receipt)
            await self._emit_durable_step_receipt(
                receipt,
                objective=objective,
                planner=planner,
                tool="computer_use",
            )
            if resolved_step.action == "fetch_topic_image" and receipt["ok"]:
                last_image_page_url = str(result.get("page_url") or "") or last_image_page_url
            if receipt["ok"] and resolved_step.action == "set_clipboard":
                expected_clipboard_sha256 = str(result.get("sha256") or "").strip()
                chars = result.get("chars")
                expected_clipboard_chars = chars if isinstance(chars, int) else None
            if receipt["ok"] and resolved_step.action == "open_app":
                expected_frontmost_app = str(result.get("frontmost_app") or result.get("opened") or "").strip()
                current_surface_requires_editable_focus = False
            elif receipt["ok"] and resolved_step.action == "open_url":
                expected_frontmost_app = str(result.get("frontmost_app") or "").strip()
                current_surface_requires_editable_focus = bool(
                    target_payload.get("requires_editable_focus")
                    or target_payload.get("require_editable_focus")
                )
                if current_surface_requires_editable_focus:
                    editor_focus_verified = bool(
                        result.get("doc_focused")
                        or result.get("editable_focus_verified")
                    )
                    task_context["desktop_task_editor_focus_verified"] = editor_focus_verified
                    task_context["desktop_task_verified_editor_url"] = str(
                        result.get("active_url") or ""
                    ).strip()
                    task_context["desktop_task_editor_focus_evidence"] = str(
                        result.get("focus_error")
                        or result.get("verification")
                        or ""
                    ).strip()
            if not receipt["ok"]:
                failures.append(receipt)
                self._emit_progress(
                    index=index,
                    total=len(steps),
                    action=resolved_step.action,
                    state="failed",
                    detail=effect_evidence,
                    level="warning",
                )
                if resolved_step.critical and params.stop_on_error:
                    break
            else:
                self._emit_progress(
                    index=index,
                    total=len(steps),
                    action=resolved_step.action,
                    state="verified",
                    detail=effect_evidence,
                )

        critical_failures = [receipt for receipt in failures if receipt.get("critical", True)]
        completed_all_steps = len(receipts) == len(steps)
        ok = not critical_failures and completed_all_steps
        status = (
            "completed_with_warnings"
            if ok and failures
            else "completed"
            if ok
            else "failed"
        )
        completed_count = sum(1 for receipt in receipts if receipt.get("ok"))
        return {
            "ok": ok,
            "status": status,
            **(
                {}
                if ok
                else {
                    "error": self._failure_cause(
                        critical_failures or failures, objective=objective
                    )
                }
            ),
            "objective": objective,
            "steps_requested": len(steps),
            "steps_completed": completed_count,
            "receipts": receipts,
            "failures": failures,
            "planner": planner,
            "document_provenance": document_provenance,
            "research": {
                "query": task_context.get("desktop_task_research_query"),
                "sources": task_context.get("desktop_task_research_sources") or [],
                "error": task_context.get("desktop_task_research_error"),
                "summary": task_context.get("desktop_task_research_summary"),
                "synthesis": task_context.get("desktop_task_research_synthesis"),
                "deep": task_context.get("desktop_task_research_deep"),
                "pressure_limited": task_context.get(
                    "desktop_task_research_pressure_limited"
                ),
            } if research_context else None,
            "summary": (
                f"Desktop task completed {completed_count}/{len(steps)} governed "
                f"computer-use steps through {planner or 'unknown'} planning."
            ),
        }
