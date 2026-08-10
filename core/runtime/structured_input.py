from __future__ import annotations

import re
from dataclasses import dataclass

_LEARNING_BUNDLE_INTRO_MARKERS = (
    "i have some suggestions",
    "places to start",
    "journey to life",
    "understanding yourself",
    "understanding us",
    "learn about humans",
    "general education",
    "science education",
    "tv shows and movies about artificial intelligence",
    "uploaded intelligence",
)

_LEARNING_BUNDLE_SECTION_MARKERS = (
    "learn about humans",
    "general education",
    "science education",
    "tv shows and movies",
    "sci-fi",
    "ai media",
)

_INTERROGATIVE_LINE_RE = re.compile(
    r'^\s*(?:["“”]\s*)?(?:what|why|how|who|when|where|which|can|could|would|should|do|does|did|is|are|if)\b',
    re.IGNORECASE,
)

_DIRECTIVE_LINE_RE = re.compile(
    r"^\s*(?:then|and then|also|next|after that|give|tell|describe|name|answer|pick|"
    r"recall|compare|contrast|choose|explain|verify|evaluate|trace|test)\b",
    re.IGNORECASE,
)

_COORDINATED_DIRECTIVE_RE = re.compile(
    r"(?:^|[,;:]\s*(?:and\s+)?|[.!?]\s+|\b(?:and|then|also|next|finally)\s+)"
    r"(?:please\s+)?(?:"
    r"answer|build|calculate|choose|compare|contrast|debug|derive|describe|design|"
    r"diagnose|enumerate|evaluate|explain|fix|give|identify|implement|justify|list|"
    r"name|plan|prove|recommend|review|select|show|summarize|tell|test|trace|validate|verify"
    r")\b",
    re.IGNORECASE | re.MULTILINE,
)

_CONNECTOR_RE = re.compile(
    r"\b(?:then|and then|after that|also)\s+(?:give|tell|describe|name|answer|pick|"
    r"recall|list|compare|contrast|choose|explain|verify|evaluate|trace|test)\b",
    re.IGNORECASE,
)

_REPEATED_CLAUSE_RE = re.compile(
    r"(?:^|[,;]\s*)(?:what|why|how|which)\b",
    re.IGNORECASE,
)

_NUMBERED_ITEM_RE = re.compile(r"(?:^|\n)\s*\d+[.)]\s+")


def _looks_like_learning_bundle_header(line: str) -> bool:
    stripped = str(line or "").strip()
    if not stripped or "http://" in stripped or "https://" in stripped:
        return False
    if not stripped.endswith(":") or len(stripped) > 120:
        return False
    lowered = stripped[:-1].strip().lower()
    return any(marker in lowered for marker in _LEARNING_BUNDLE_SECTION_MARKERS)


def _parse_learning_resource_line(line: str, category: str = "") -> dict[str, str] | None:
    cleaned = re.sub(r"^\s*(?:[-*]|\d+\.)\s*", "", str(line or "").strip())
    if not cleaned or _looks_like_learning_bundle_header(cleaned):
        return None

    head, sep, tail = cleaned.rpartition(":")
    if not sep:
        return None

    description = tail.strip().lstrip(":").strip()
    if len(description) < 8:
        return None

    title = head.strip()
    url = ""
    creator = ""
    url_match = re.match(r"^(?P<title>.+?)\s+\((?P<url>https?://[^)]+)\)\s*$", title)
    if url_match:
        title = url_match.group("title").strip()
        url = url_match.group("url").strip()
    elif " - " in title:
        title, creator = title.rsplit(" - ", 1)
        title = title.strip()
        creator = creator.strip()

    if not title:
        return None

    return {
        "category": str(category or "").strip(),
        "title": title,
        "url": url,
        "creator": creator,
        "description": description,
    }


def looks_like_learning_resource_bundle(text: str) -> bool:
    raw = str(text or "")
    if len(raw) < 280:
        return False

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) < 6:
        return False

    lowered = raw.lower()
    url_count = len(re.findall(r"https?://[^\s<>\"')\]]+", raw))
    header_count = sum(1 for line in lines if _looks_like_learning_bundle_header(line))

    category = ""
    resource_count = 0
    for line in lines:
        if _looks_like_learning_bundle_header(line):
            category = line.rstrip(":").strip()
            continue
        if _parse_learning_resource_line(line, category):
            resource_count += 1

    intro_hit = any(marker in lowered for marker in _LEARNING_BUNDLE_INTRO_MARKERS)
    return (
        (url_count >= 4 and resource_count >= 5)
        or (header_count >= 2 and resource_count >= 5)
        or (intro_hit and resource_count >= 4)
    )


@dataclass(frozen=True)
class PromptShape:
    question_parts: int = 1
    explicit_question_marks: int = 0
    question_like_lines: int = 0
    connector_parts: int = 0
    repeated_clause_parts: int = 0
    numbered_parts: int = 0
    imperative_parts: int = 0
    prefers_extended_answer: bool = False
    requires_single_reply_coverage: bool = False
    #: The actual text of each ask, not just how many there were.
    #:
    #: The count alone can shape a prompt ("3 parts detected") and size a
    #: voice budget, and it cannot check whether a reply covered them —
    #: checking needs to know WHAT was asked. Retained so
    #: validate_dialogue_response can hold the answer against the question.
    question_segments: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, int | bool | tuple[str, ...]]:
        return {
            "question_parts": self.question_parts,
            "explicit_question_marks": self.explicit_question_marks,
            "question_like_lines": self.question_like_lines,
            "connector_parts": self.connector_parts,
            "repeated_clause_parts": self.repeated_clause_parts,
            "numbered_parts": self.numbered_parts,
            "imperative_parts": self.imperative_parts,
            "prefers_extended_answer": self.prefers_extended_answer,
            "requires_single_reply_coverage": self.requires_single_reply_coverage,
            "question_segments": self.question_segments,
        }


#: Splits an utterance into the units a person would count as separate asks:
#: sentence enders, and the line breaks / numbered items that carry a list.
_ASK_SPLIT_RE = re.compile(r"(?<=[.?!])\s+|\n+")


def _question_segments(text: str) -> tuple[str, ...]:
    """The individual asks in an utterance, as text.

    LIVE DEFECT, 2026-08-10. "give me one concrete example of a preposition
    doing more work than it should. and separately — do you actually enjoy
    that, or is 'interesting' a word you reach for because it's safe?" She
    answered the example and said nothing whatever about enjoyment. The same
    failure was called out earlier in the same session — "you dodged half of
    it. I asked two things and you answered one."

    The runtime already KNEW it was compound: question_parts was computed,
    the prompt was told "this prompt contains multiple asks (2 detected)",
    and the voice budget was widened for it. Nothing ever checked the reply
    against it, because the count was all that survived analysis. Keeping the
    segments is what makes coverage checkable at all.

    An ask is a SENTENCE that either ends in a question mark or opens with a
    directive verb. Sentences, not lines, because everything else here counts
    per line and that is what missed the case above: it arrived as one line,
    so _INTERROGATIVE_LINE_RE — which requires the LINE to begin with what,
    why, do, is — never matched, the line began with "give", and a two-part
    utterance scored one part. Anyone typing in a chat box writes several
    sentences on one line constantly.
    """
    raw = str(text or "").strip()
    if not raw:
        return ()
    segments = [part.strip() for part in _ASK_SPLIT_RE.split(raw) if part.strip()]
    return tuple(
        part
        for part in segments
        if part.endswith("?") or _DIRECTIVE_LINE_RE.match(part)
    )


def analyze_prompt_shape(text: str) -> PromptShape:
    raw = str(text or "").strip()
    if not raw:
        return PromptShape()
    if looks_like_learning_resource_bundle(raw):
        return PromptShape()

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    explicit_question_marks = raw.count("?")
    question_like_lines = 0
    directive_lines = 0
    for line in lines:
        if len(line) < 12:
            continue
        if "?" in line and _INTERROGATIVE_LINE_RE.match(line):
            question_like_lines += 1
        elif _DIRECTIVE_LINE_RE.match(line):
            directive_lines += 1

    connector_parts = len(_CONNECTOR_RE.findall(raw))
    numbered_parts = len(_NUMBERED_ITEM_RE.findall(raw))
    repeated_clause_parts = max(0, len(_REPEATED_CLAUSE_RE.findall(raw)) - 1)
    imperative_parts = len(_COORDINATED_DIRECTIVE_RE.findall(raw))

    ask_segments = _question_segments(raw)

    part_candidates = [
        1,
        # Sentence-level asks. Every other candidate below counts per LINE or
        # per verb list, and a chat box is one line: "give me an example of X.
        # and separately — do you enjoy it?" scored 1 part, so the prompt was
        # never told it was compound and the reply dropped half of it.
        len(ask_segments),
        explicit_question_marks,
        question_like_lines,
        numbered_parts,
        connector_parts + 1 if connector_parts else 0,
        repeated_clause_parts + 1 if repeated_clause_parts else 0,
        directive_lines if directive_lines >= 2 else 0,
        imperative_parts if imperative_parts >= 2 else 0,
    ]
    question_parts = max(1, min(6, max(part_candidates)))

    prefers_extended_answer = bool(
        question_parts >= 2
        or (len(raw) >= 320 and ("\n" in raw or ":" in raw))
        or (explicit_question_marks >= 1 and len(raw.split()) >= 60)
    )
    requires_single_reply_coverage = bool(
        question_parts >= 2 or connector_parts > 0 or repeated_clause_parts >= 2
    )

    return PromptShape(
        question_segments=ask_segments,
        question_parts=question_parts,
        explicit_question_marks=explicit_question_marks,
        question_like_lines=question_like_lines,
        connector_parts=connector_parts,
        repeated_clause_parts=repeated_clause_parts,
        numbered_parts=numbered_parts,
        imperative_parts=imperative_parts,
        prefers_extended_answer=prefers_extended_answer,
        requires_single_reply_coverage=requires_single_reply_coverage,
    )
