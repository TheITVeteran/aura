from __future__ import annotations

import re
from typing import Any


_RESEARCH_PREFIXES = (
    "research and learn something new about ",
    "research ",
    "learn about ",
    "explore ",
    "investigate ",
    "look into ",
    "find out about ",
    "self-directed exploration of ",
)

_STALE_OR_RECEIPT_MARKERS = (
    "unresolved: stalled goal:",
    "stalled goal:",
    "desktop task receipt",
    "canonical computer-use gateway",
    "governed desktop actuators",
    "artifact references:",
)

_PROMPT_SCAFFOLD_MARKERS = (
    "subconscious synthesis",
    "concept a:",
    "concept b:",
    "task:",
    "strategic heuristic",
    "universal principle",
    "predict how self will react if i take this action",
    "{'type': 'autonomous_goal'",
    '"type": "autonomous_goal"',
    "mastery of: user asked about:",
    "reply \"no_connection\"",
    "reply 'no_connection'",
    "json schema",
    "system prompt",
    "you are an ai",
)

_DESKTOP_ACTION_MARKERS = (
    "create a folder",
    "write a file",
    "write a note",
    "open notes",
    "notes app",
    "google docs",
    "export",
    "pdf",
    "wallpaper",
    "desktop folder",
    "documents folder",
    "open chrome",
    "open safari",
    "type out",
    "keyboard",
    "mouse",
)

_RESEARCHABLE_HINTS = (
    "why ",
    "how ",
    "what ",
    "which ",
    "research",
    "learn",
    "explore",
    "investigate",
    "study",
    "papers",
    "evidence",
    "sources",
)


def normalize_goal_text(value: Any) -> str:
    text = str(value or "").replace("\x00", " ")
    return " ".join(text.split()).strip(" -:;,.?!")


def is_prompt_shaped_goal(value: Any) -> bool:
    text = normalize_goal_text(value)
    if not text:
        return False
    lowered = text.casefold()
    marker_hits = sum(1 for marker in _PROMPT_SCAFFOLD_MARKERS if marker in lowered)
    if marker_hits >= 2:
        return True
    if len(text) > 500 and marker_hits >= 1:
        return True
    return bool(re.search(r"\btask:\s*\d+\.\s", lowered))


def is_stale_or_prompt_scaffold_goal(value: Any) -> bool:
    text = normalize_goal_text(value)
    if not text:
        return False
    lowered = text.casefold()
    if any(marker in lowered for marker in _STALE_OR_RECEIPT_MARKERS):
        return True
    return is_prompt_shaped_goal(text)


def is_desktop_action_goal(value: Any) -> bool:
    lowered = normalize_goal_text(value).casefold()
    if not lowered:
        return False
    hits = sum(1 for marker in _DESKTOP_ACTION_MARKERS if marker in lowered)
    if hits >= 2:
        return True
    return lowered.startswith(("open ", "create ", "write ", "export ", "change ")) and hits >= 1


def is_unresearchable_goal(value: Any) -> bool:
    """True when a pending initiative should not be fed to background research.

    This keeps autonomy active while preventing stale action receipts, desktop
    tasks, and internal prompt scaffolds from becoming web-search queries.
    """

    text = normalize_goal_text(value)
    if not text:
        return True
    lowered = text.casefold()
    if any(marker in lowered for marker in _STALE_OR_RECEIPT_MARKERS):
        return True
    if is_prompt_shaped_goal(text):
        return True
    if is_desktop_action_goal(text):
        return True
    return False


def research_query_for_goal(value: Any, *, limit: int = 220) -> str:
    text = normalize_goal_text(value)
    if not text or is_unresearchable_goal(text):
        return ""
    lowered = text.casefold()
    for prefix in _RESEARCH_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip(" -:;,.?!")
            break
    if not text or is_unresearchable_goal(text):
        return ""
    clauses = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    if len(text) > limit and clauses:
        text = clauses[0]
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].strip(" -:;,.?!")
    lowered = text.casefold()
    if not any(hint in lowered for hint in _RESEARCHABLE_HINTS) and len(text.split()) > 24:
        return ""
    return text
