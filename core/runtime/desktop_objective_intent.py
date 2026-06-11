from __future__ import annotations

import re

from core.runtime.skill_task_bridge import (
    looks_like_capability_inventory_dialogue_request,
    strip_negated_action_spans,
)
from core.utils.intent_normalization import normalize_memory_intent_text

_WEB_SEARCH_REQUEST_SPAN_RE = re.compile(
    r"\b(?:search|google|look\s*up|research)\b[^.?!]{0,48}?"
    r"\b(?:the\s+)?(?:internet|web|online)\b",
    re.IGNORECASE,
)

_DESKTOP_OBJECTIVE_ACTION_TERMS = (
    "attach",
    "arrange",
    "browse",
    "click",
    "compose",
    "close",
    "create",
    "download",
    "export",
    "find",
    "focus",
    "google",
    "insert",
    "look up",
    "maximize",
    "minimize",
    "move",
    "navigate",
    "open",
    "organize",
    "paste",
    "pdf",
    "resize",
    "save",
    "search",
    "select",
    "show me",
    "switch",
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
    r"type|paste|compose|download|navigate|click|show\s+me|arrange|resize|drag|"
    r"focus|select|switch|close|minimi[sz]e|maximi[sz]e|organize)\b",
    re.IGNORECASE,
)
_EXPLANATORY_DESKTOP_QUESTION_RE = re.compile(
    r"^\s*(?:how|what|why)\s+(?:would|could|should|do|does|can)\s+(?:you\s+)?"
    r"(?:open|create|write|save|export|search|google|look\s+up|type|paste|"
    r"compose|download|navigate|click|arrange|resize|drag|focus|select|switch|"
    r"close|minimi[sz]e|maximi[sz]e|organize)\b",
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
    sanitized_text = strip_negated_action_spans(text).lower()
    # Explicit web-search phrasing ("search the internet/web for X") is a
    # research request, not a desktop objective; classifying it as desktop
    # let the response contract suppress requires_search while desktop_task
    # had nothing visible to do — the search silently went dark on both
    # lanes. Strip the span so only OTHER action/surface terms ("...and
    # save it to Notes") can still classify the request as desktop.
    sanitized_text = _WEB_SEARCH_REQUEST_SPAN_RE.sub(" ", sanitized_text)
    if looks_like_capability_inventory_dialogue_request(user_message):
        return False
    if _EXPLANATORY_DESKTOP_QUESTION_RE.search(text):
        return False
    if not _contains_desktop_objective_term(sanitized_text, _DESKTOP_OBJECTIVE_ACTION_TERMS):
        return False
    if not _contains_desktop_objective_term(sanitized_text, _DESKTOP_OBJECTIVE_SURFACE_TERMS):
        return False

    try:
        from core.phases.action_intent import detect_action_intent

        intent = detect_action_intent(user_message)
        if bool(getattr(intent, "should_execute", False)):
            return True
        if bool(getattr(intent, "has_action_request", False)) and re.search(
            r"\b(?:can|could|will|would)\s+you\b",
            sanitized_text,
            flags=re.IGNORECASE,
        ):
            return True
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        pass

    return bool(_DIRECT_DESKTOP_ACTION_RE.search(sanitized_text))


__all__ = ["looks_like_desktop_objective"]
