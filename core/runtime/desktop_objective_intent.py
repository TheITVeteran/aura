from __future__ import annotations

import re

from core.utils.intent_normalization import normalize_memory_intent_text

_DESKTOP_OBJECTIVE_ACTION_TERMS = (
    "attach",
    "browse",
    "click",
    "compose",
    "create",
    "download",
    "export",
    "find",
    "google",
    "insert",
    "look up",
    "move",
    "navigate",
    "open",
    "paste",
    "pdf",
    "save",
    "search",
    "show me",
    "tab",
    "timestamp",
    "type",
    "write",
)

_DESKTOP_OBJECTIVE_SURFACE_TERMS = (
    "app",
    "application",
    "browser",
    "chrome",
    "computer",
    "desktop",
    "doc",
    "docs",
    "document",
    "drive",
    "file",
    "finder",
    "folder",
    "google",
    "notes",
    "pages",
    "pdf",
    "safari",
    "screen",
    "tab",
    "textedit",
    "web",
    "website",
    "window",
    "word",
)

_DIRECT_DESKTOP_ACTION_RE = re.compile(
    r"\b(?:please\s+)?(?:open|create|write|save|export|search|google|look\s+up|"
    r"type|paste|compose|download|navigate|click|show\s+me)\b",
    re.IGNORECASE,
)
_EXPLANATORY_DESKTOP_QUESTION_RE = re.compile(
    r"^\s*(?:how|what|why)\s+(?:would|could|should|do|does|can)\s+(?:you\s+)?"
    r"(?:open|create|write|save|export|search|google|look\s+up|type|paste|"
    r"compose|download|navigate|click)\b",
    re.IGNORECASE,
)


def _contains_desktop_objective_term(text: str, terms: tuple[str, ...]) -> bool:
    for term in terms:
        escaped = re.escape(term)
        if re.search(rf"\b{escaped}\b", text, flags=re.IGNORECASE):
            return True
    return False


def looks_like_desktop_objective(user_message: str) -> bool:
    """Return true for user requests that need visible desktop/computer action.

    This is intentionally shared by typed chat and voice so Aura does not have
    two drifting definitions of "this request needs the desktop body." The
    function only classifies the objective; actual execution still goes through
    CognitiveEngine, CapabilityEngine, desktop_task, computer_use, and the
    permission/governance gates.
    """

    text = normalize_memory_intent_text(user_message).lower()
    if not text:
        return False
    if _EXPLANATORY_DESKTOP_QUESTION_RE.search(text):
        return False
    if not _contains_desktop_objective_term(text, _DESKTOP_OBJECTIVE_ACTION_TERMS):
        return False
    if not _contains_desktop_objective_term(text, _DESKTOP_OBJECTIVE_SURFACE_TERMS):
        return False

    try:
        from core.phases.action_intent import detect_action_intent

        intent = detect_action_intent(user_message)
        if bool(getattr(intent, "should_execute", False)):
            return True
        if bool(getattr(intent, "has_action_request", False)) and re.search(
            r"\b(?:can|could|will|would)\s+you\b",
            text,
            flags=re.IGNORECASE,
        ):
            return True
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        pass

    return bool(_DIRECT_DESKTOP_ACTION_RE.search(text))


__all__ = ["looks_like_desktop_objective"]
