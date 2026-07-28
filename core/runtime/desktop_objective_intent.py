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
_CANONICAL_RESEARCH_TOOL_SPAN_RE = re.compile(
    r"\b(?:use|run|call|invoke|route\s+through|with|via)?\s*"
    r"(?:web_search|grounded_search|search_web|free_search)\b"
    r"[^.?!]{0,80}",
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
    # Changing a machine setting is a desktop objective too. The list had
    # every verb for moving files and windows and none for altering the
    # system itself, so "change my desktop background to an orca" — which
    # desktop_task can do through system_control — did not route at all and
    # was answered conversationally. Measured live 2026-07-27.
    "change",
    "set",
    "turn on",
    "turn off",
    "enable",
    "disable",
    "adjust",
    "increase",
    "decrease",
    "mute",
    "unmute",
)

_DESKTOP_OBJECTIVE_SURFACE_TERMS = (
    # System surfaces a setting verb acts on, so "set the volume" and "change
    # my wallpaper" reach the lane that can actually do them.
    "background",
    "wallpaper",
    "brightness",
    "volume",
    "dark mode",
    "do not disturb",
    "night shift",
    "setting",
    "settings",
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
    r"focus|select|switch|close|minimi[sz]e|maximi[sz]e|organize|"
    # Setting verbs, but only when they act on a machine surface — "change my
    # wallpaper" is a desktop objective, "change your mind" is not. The verb
    # alone is far too common in ordinary speech to admit on its own.
    r"(?:change|set|adjust|increase|decrease|turn\s+(?:on|off)|enable|disable|"
    r"mute|unmute)\s+(?:the\s+|my\s+|your\s+)?(?:desktop\s+)?"
    r"(?:background|wallpaper|brightness|volume|dark\s+mode|night\s+shift|"
    r"do\s+not\s+disturb|setting|settings|screen\s+saver))\b",
    re.IGNORECASE,
)
_EXPLANATORY_DESKTOP_QUESTION_RE = re.compile(
    r"^\s*(?:how|what|why)\s+(?:would|could|should|do|does|can)\s+(?:you\s+)?"
    r"(?:open|create|write|save|export|search|google|look\s+up|type|paste|"
    r"compose|download|navigate|click|arrange|resize|drag|focus|select|switch|"
    r"close|minimi[sz]e|maximi[sz]e|organize)\b",
    re.IGNORECASE,
)
# Screen-observation requests need the desktop BODY (read_screen_text), but
# they carry no action+surface verb pair, so the generic classifier missed
# them and "what's on my screen" silently did nothing. Treat them as desktop
# objectives directly. Kept in sync with desktop_task's observation markers.
# Talking ABOUT a past observation is not asking for a new one.
#
# Live 2026-07-27: a message that began "Earlier you described what was on
# his screen and I decided you had made it up" was routed to the governed
# desktop lane and refused, because "described ... screen" reads exactly like
# "describe my screen". Recounting what already happened, or discussing the
# faculty itself, is conversation.
_PAST_SCREEN_NARRATION_RE = re.compile(
    r"\b(?:earlier|previously|before|a\s+moment\s+ago|last\s+time|"
    r"you\s+(?:described|said|told|reported|showed|mentioned|claimed|were)|"
    r"i\s+(?:decided|thought|assumed|concluded|said))\b",
    re.IGNORECASE,
)

_SCREEN_OBSERVATION_RE = re.compile(
    r"\b(?:read|look\s+at|inspect|describe|check|examine|capture|view)\b"
    r"[^.?!]{0,40}\bscreen\b"
    r"|\bwhat(?:'s|\s+is|\s+are|\s+do\s+you\s+see)\b[^.?!]{0,40}\bscreen\b"
    r"|\bscreenshot\b",
    re.IGNORECASE,
)


# Surfaces whose ordinary English meaning is far more common than the app that
# shares the name. A bare word-boundary match on these turns any sentence
# containing an everyday noun into a desktop-control request.
#
# Measured live: "Aura, it's Bryan. Remember the WORD lantern ... show me the
# real output. Run a Python snippet that prints the PID and CPU cores." The
# action term "show me" came from one sentence and the surface term "word" —
# meaning Microsoft Word — came from another, so a code-execution request was
# routed into desktop OS automation, which then correctly refused for lack of an
# observable acceptance contract. The user got a failure for a request the
# sandbox could have answered.
#
# These now require actual app context: a vendor name, an app/document noun, or
# a preposition/verb that only makes sense against an application.
_APP_CONTEXT_SURFACE_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "word",
        r"(?:\b(?:microsoft|ms)\s+word\b"
        r"|\bword\s+(?:doc|docs|document|documents|file|files|app)\b"
        r"|\b(?:open|in|into|to|from|using|with|launch|quit|close|switch\s+to)\s+word\b)",
    ),
    (
        "pages",
        r"(?:\bapple\s+pages\b"
        r"|\bpages\s+(?:doc|docs|document|documents|file|files|app)\b"
        r"|\b(?:open|into|using|launch|quit|close|switch\s+to)\s+pages\b)",
    ),
    (
        "drive",
        r"(?:\bgoogle\s+drive\b"
        r"|\bdrive\s+(?:folder|folders|file|files)\b"
        r"|\b(?:in|into|on|from|to)\s+(?:my\s+|the\s+)?drive\b)",
    ),
)

_PLAIN_ENGLISH_APP_NAMES = frozenset(term for term, _ in _APP_CONTEXT_SURFACE_PATTERNS)


def _contains_desktop_objective_term(text: str, terms: tuple[str, ...]) -> bool:
    for term in terms:
        if term in _PLAIN_ENGLISH_APP_NAMES:
            continue
        escaped = re.escape(term)
        if re.search(rf"\b{escaped}\b", text, flags=re.IGNORECASE):
            return True
    requested = set(terms)
    for term, pattern in _APP_CONTEXT_SURFACE_PATTERNS:
        if term in requested and re.search(pattern, text, flags=re.IGNORECASE):
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
    # A named canonical research tool is not itself a visible desktop
    # objective. "Use web_search from the desktop lane" should stay in the
    # research/tool path; "open Chrome and search" still routes to desktop
    # because the visible app action remains after this span is stripped.
    sanitized_text = _CANONICAL_RESEARCH_TOOL_SPAN_RE.sub(" ", sanitized_text)
    if looks_like_capability_inventory_dialogue_request(user_message):
        return False
    # Screen observation ("read my screen", "what's on my screen") needs the
    # desktop body even though it carries no action+surface verb pair.
    if _SCREEN_OBSERVATION_RE.search(sanitized_text) and not (
        _PAST_SCREEN_NARRATION_RE.search(sanitized_text)
    ):
        return True
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
