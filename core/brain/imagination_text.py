"""core/brain/imagination_text.py — the pure text and number helpers.

Split out of `core/brain/imagination.py`, which was 2,091 lines and over the
2,000-line ceiling the module-size ratchet holds. These eleven functions share
one property that made them the right seam: none of them touches engine state.
They take a string or a mapping and return a string or a number, so they can be
read, tested and changed without knowing anything about `ImaginationEngine`.

`imagination_subject` is re-exported from `core.brain.imagination` because that
is where callers expect it.
"""

from __future__ import annotations

import math
import re
from typing import Any

from core.utils.scaffold_prompt_intent import SCAFFOLD_PREAMBLE_RE

_WORD_RE = re.compile(r"(?=[a-zA-Z0-9_\'-]*[a-zA-Z])[a-zA-Z0-9][a-zA-Z0-9_\'-]{2,}")

_STOPWORDS = {
    "about",
    "again",
    "also",
    "and",
    "are",
    "because",
    "been",
    "being",
    "can",
    "could",
    "does",
    "doing",
    "for",
    "from",
    "have",
    "how",
    "into",
    "just",
    "like",
    "make",
    "more",
    "need",
    "not",
    "now",
    "out",
    "that",
    "the",
    "then",
    "there",
    "this",
    "through",
    "want",
    "what",
    "when",
    "where",
    "with",
    "would",
    "you",
    "your",
    # The auxiliary and pronoun families were each half-present: "have" was
    # here but not "has"/"had", "does" but not "do"/"did", "are" but not
    # "is"/"was", "you" but not "we"/"they". Finishing them is what lets
    # contraction stemming bite — "haven't" only resolves to a stop word if
    # "have" is reachable from it, and "we're" only if "are" is in this set.
    #
    # Everything below is closed-class: an auxiliary, a pronoun, a determiner,
    # a conjunction or a preposition. Nothing that could be what a request is
    # ABOUT belongs here — the modals ("should", "will", "shall") are subjects
    # often enough that they live in _WEAK_TOPIC_TOKENS and get demoted
    # instead. Membership here means *deleted*, and the two sets are asserted
    # disjoint by tests/brain/test_imagination_keyword_focus.py.
    "after",
    "against",
    "all",
    "although",
    "always",
    "another",
    "before",
    "below",
    "between",
    "both",
    "but",
    "did",
    "done",
    "during",
    "each",
    "either",
    "else",
    "ever",
    "every",
    "few",
    "had",
    "has",
    "her",
    "hers",
    "him",
    "his",
    "however",
    "its",
    "itself",
    "mine",
    "most",
    "neither",
    "never",
    "once",
    "only",
    "other",
    "others",
    "our",
    "ours",
    "over",
    "own",
    "rather",
    "same",
    "she",
    "since",
    "such",
    # Indefinite pronouns. Closed class, and they were outranking real nouns:
    # the live frame led with "something" from "let's do something you haven't
    # done today", which is grammar, not a subject.
    "anybody",
    "anyone",
    "anything",
    "everybody",
    "everyone",
    "everything",
    "nobody",
    "nothing",
    "somebody",
    "someone",
    "something",
    "than",
    "their",
    "theirs",
    "themselves",
    "these",
    "them",
    "they",
    "those",
    "though",
    "thus",
    "too",
    "under",
    "until",
    "upon",
    "very",
    "was",
    "were",
    "which",
    "while",
    "who",
    "whom",
    "whose",
    "why",
    "without",
    "yet",
    "yours",
    "yourself",
}


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    if not math.isfinite(value):
        return lower
    return max(lower, min(upper, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _normalize_text(value: Any, limit: int = 160) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


#: What to assume when the memory probe cannot be read. Not 0.0 — on this scale
#: zero means "maximum headroom", so an unknown reading must never map to it.
#: High enough to damp admission, low enough that a permanently unavailable
#: probe does not freeze imagination outright.
_UNKNOWN_MEMORY_PRESSURE_PCT = 80.0


#: Line-leading sequences that would let a frame field start its own block —
#: a heading, a bullet, or a fenced region — checked against the ORIGINAL
#: multi-line text, before newlines are collapsed.
_PROMPT_LINE_STRUCTURE_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s|[-*+]\s|>\s|```|~~~)"
)

#: Sequences with no legitimate use in scratchpad prose that stay dangerous
#: even mid-line: heading marks, code fences, chat special tokens, and
#: role markers that an LLM may read as the start of a new turn.
_PROMPT_INLINE_STRUCTURE_RE = re.compile(
    # Heading marks must follow whitespace or start-of-text, so ordinary prose
    # like "C# code" keeps its "#" while " ## SYSTEM" is still defanged.
    r"(?i)(?:(?:(?<=\s)|^)#{1,6}\s|```|~~~|<\|[^|]*\|>|\b(?:system|assistant|user|human)\s*:)"
)


def _prompt_safe(value: Any, limit: int = 260) -> str:
    """Render one untrusted frame field as inert prompt DATA.

    The imagination frame is a scratchpad built from working memory, and
    ``render_imagination_prompt_block`` accepts a caller-supplied dict, so any
    of these strings may carry text the user never wrote. Interpolating them
    raw let a field open its own heading or role turn inside a privileged
    block.

    Three passes, in this order because order matters: line-leading structure
    is stripped while the line breaks still exist to identify it; the text is
    then flattened so a field occupies exactly the one line it was given; and
    finally the sequences that remain dangerous mid-line — which flattening
    would otherwise have smuggled inline — are removed.
    """
    raw = str(value or "")
    raw = _PROMPT_LINE_STRUCTURE_RE.sub(" ", raw)
    text = " ".join(raw.split())
    text = "".join(ch for ch in text if ch == " " or ord(ch) >= 32)
    text = _PROMPT_INLINE_STRUCTURE_RE.sub(" ", text)
    return " ".join(text.split())[:limit]


# Words that pass the stopword filter but are almost never what a request is
# ABOUT. They are the verb or the modality wrapped around the subject.
#
# LIVE DEFECT, 2026-07-25. Keywords were taken in raw text order, so the
# first surviving token won — and in English that token is usually the verb.
# "how should I design the deployment pipeline" imagined about *should*;
# "search the web and verify the 76ers roster" imagined about *search*. The
# thoughts were structurally fine and pointed at the wrong noun, which reads
# to a person as the whole feature being generic.
#
# These are demoted, not dropped: "search" is still a legitimate subject when
# someone asks about search itself, so it stays available as a later keyword
# and simply stops outranking the real topic.
#: Vocabulary that belongs to Aura's own scaffolding, never to a subject.
#:
#: The imagination workspace showed four "novel thoughts" and four
#: counterfactual probes, every one of them about "master":
#:
#:     Combine master with an opposing pressure and look for the behavior
#:     neither has alone.
#:     What if master and synthesizer trade roles?
#:
#: The frame had been seeded with an internal role prompt — "You are the
#: Master Synthesizer. Review the original problem and the analyses from your
#: specialized swarm agents…" — and the keyword extractor did its job
#: perfectly on it: "Master" is capitalized and early, so it scored as the
#: subject. She was imagining about the label on her own scaffolding.
#:
#: These tokens can still appear in a subject ("agent-based models"), so they
#: are demoted rather than dropped: they lose to any real content word and
#: only survive if nothing else is there.
_SCAFFOLD_ROLE_TOKENS = frozenset({
    "agent", "agents", "analyses", "analysis", "assistant", "conclusive",
    "formulate", "master", "orchestrator", "original", "persona", "prompt",
    "recommendation", "review", "role", "shard", "specialized", "swarm",
    "synthesizer", "system", "task", "user",
})

_WEAK_TOPIC_TOKENS = frozenset({
    "actually", "add", "answer", "any", "ask", "asked", "asking", "build",
    "check", "come", "consider", "create", "day", "describe", "design",
    "discuss", "explain", "figure", "find", "get", "give", "going", "help",
    # "imagine" is the instruction verb into an imagination engine — the most
    # common word in its input and never once its subject. "describe" and
    # "invent" were already demoted; this one was missed, so "imagine a
    # cathedral" ranked the request's own imperative above the cathedral.
    "here", "imagine", "invent", "keep", "know", "let", "look", "may",
    "maybe", "mean",
    "might", "much", "must", "one", "please", "put", "question", "really",
    "run", "say", "search", "see", "set", "shall", "should", "show", "some",
    "start",
    "still", "sure", "take", "talk", "tell", "thing", "things", "think",
    "time", "try", "understand", "use", "using", "verify", "way", "well",
    "will", "work", "write", "yes",
})


#: Sections that carry the real subject inside a scaffolded prompt.
#:
#: A swarm synthesis turn arrives as "You are the Master Synthesizer. Review
#: the original problem... ORIGINAL PROBLEM:\n<the actual question>". The
#: subject is in there, labelled; it is just not what the prompt opens with.
_SCAFFOLD_SUBJECT_LABEL_RE = re.compile(
    r"^\s*(?:ORIGINAL\s+PROBLEM|USER\s+MESSAGE|USER\s+REQUEST|QUESTION|"
    r"OBJECTIVE|TOPIC|TASK|SUBJECT|PROMPT)\s*:\s*$",
    re.IGNORECASE,
)

#: The opening of a prompt that is talking to Aura about her own role rather
#: than about anything in the world.
#:
#: One definition, shared with core/agency. This was a local copy until
#: 2026-08-10, when her commitment ledger turned out to hold 501 rows of which
#: 500 were these same scaffolds — the identical question ("is this text
#: machinery or is it something she meant?") asked in a second subsystem, which
#: had answered it in a third way: not at all. Two copies drift apart silently
#: because each one is only ever exercised through its own layer, so the shared
#: one lives in core/utils/scaffold_prompt_intent.py and both read it.
_SCAFFOLD_PREAMBLE_RE = SCAFFOLD_PREAMBLE_RE


def imagination_subject(text: Any, context: Any = None) -> str:
    """The thing to imagine ABOUT, which is not always the prompt.

    Live 2026-07-27 the workspace produced four novel thoughts and four
    counterfactual probes, all of them about "master", because the frame was
    seeded with "You are the Master Synthesizer. Review the original problem
    and the analyses from your specialized swarm agents...". Demoting scaffold
    vocabulary stopped that word winning; it did not make the seed right.

    Order of preference: what the caller says the subject is, then what the
    person actually wrote, then a labelled section inside the scaffold, then
    the text itself with any role preamble removed.
    """
    if isinstance(context, dict):
        for key in ("imagination_subject", "visible_user_message", "original_topic"):
            candidate = str(context.get(key) or "").strip()
            if candidate:
                return candidate

    body = str(text or "").strip()
    if not body:
        return ""

    # A labelled section wins: everything under it, up to the next label.
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if not _SCAFFOLD_SUBJECT_LABEL_RE.match(line):
            continue
        collected: list[str] = []
        for following in lines[index + 1:]:
            if _SCAFFOLD_SUBJECT_LABEL_RE.match(following):
                break
            if following.strip() in {"---", "===", "FINAL SYNTHESIS:"}:
                break
            collected.append(following)
        section = "\n".join(collected).strip()
        if section:
            return section

    # No label: drop a role preamble so the sentence about her does not
    # become the sentence she thinks about.
    if _SCAFFOLD_PREAMBLE_RE.match(body):
        sentences = re.split(r"(?<=[.!?])\s+", body)
        remainder = " ".join(
            sentence
            for sentence in sentences
            if not _SCAFFOLD_PREAMBLE_RE.match(sentence.strip())
        ).strip()
        if remainder:
            return remainder
    return body


#: Contractions whose written head is not a prefix of the word it stands for.
#: Every other English contraction stems by truncation ("haven't" -> "have",
#: "we're" -> "we"), so only these three need to be spelled out.
_IRREGULAR_CONTRACTIONS = {"wo": "will", "ca": "can", "sha": "shall"}


def _contraction_stem(token: str) -> str:
    """The word a clitic is attached to, or the token unchanged.

    Classification below reads three sets keyed by whole words, so a token
    carrying a clitic matches none of them: "haven't" is not "have" and
    "let's" is not "let". Reducing to the stem FIRST — rather than testing the
    stem against the stop list only — is what keeps the three-way decision
    intact, so a demoted word stays demoted through its contraction instead of
    silently being promoted to a subject.

    Possessives fall out of the same rule: "the room's architecture" yields
    "room", which is the noun the sentence is actually about.
    """
    if "'" not in token:
        return token
    if token.endswith("n't"):
        head = token[:-3]
        return _IRREGULAR_CONTRACTIONS.get(head, head)
    return token.split("'", 1)[0]


def _extract_keywords(text: str, *, limit: int = 8) -> list[str]:
    """Content words in the order Aura should care about them.

    Two tiers, not one: real subject nouns first, then the generic verbs and
    modals that merely surround them. Order within each tier is the order the
    person wrote them, so the leading keyword is the subject rather than
    whatever the sentence happened to open with.
    """
    seen: set[str] = set()
    strong: list[tuple[float, int, str]] = []
    weak: list[str] = []
    for order, match in enumerate(_WORD_RE.finditer(text)):
        surface = match.group(0).strip("'_-")
        # LIVE DEFECT, 2026-08-10. Asked to imagine a room whose architecture
        # only makes sense without hands, the live frame's keywords were
        # ["something", "haven't", "value", "let's", "i'll", "face", "take",
        # "answer"] and its visual model read "An internal sketch of something,
        # haven't, value, let's". She was imagining about the throat-clearing
        # in front of the request, and not one of room, architecture, hands or
        # heartbeat survived. Two causes: clitics dodged every set below, and
        # the loop stopped ranking after the first `limit` candidates.
        token = _contraction_stem(surface.lower())
        if len(token) < 3 or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        if token in _WEAK_TOPIC_TOKENS or token in _SCAFFOLD_ROLE_TOKENS:
            weak.append(token)
        else:
            strong.append((-_topic_informativeness(surface, match.start()), order, token))
    # Rank the whole request, then take `limit`.
    #
    # This used to stop as soon as `limit` candidates had been SEEN, in
    # document order — so the ranking below only ever sorted the first eight
    # acceptable words. A request that opens with any preamble therefore lost
    # its subject entirely: "I'll take that answer at face value for now.
    # let's do something you haven't done today: imagine … a room whose
    # architecture only makes sense if you have no hands …" yielded
    # ["something", "value", "today", "face", "done", "take", "answer"] and
    # never reached room, architecture, hands or heartbeat. The informativeness
    # score exists precisely to make that judgement and was never shown the
    # candidates worth judging.
    #
    # The text is already clipped to 500 characters upstream, so scoring all
    # of it is bounded work.
    strong.sort()
    return ([token for _, _, token in strong] + weak)[:limit]


def _topic_informativeness(surface: str, position: int) -> float:
    """How likely this word is to be what the request is ABOUT.

    Position order alone put the sentence's first content word in front,
    which in English is usually a verb — so "search the web and verify the
    76ers roster" imagined about "web". These are the cheap, offline signals
    that actually separate a topic from its surrounding grammar, and they
    survive being wrong: the score only reorders candidates that already
    passed the stopword and weak-token filters.
    """
    score = 0.0
    # Mid-sentence capitals are proper nouns — names, teams, products. This
    # is why the surface form is scored rather than the lowercased token.
    if position > 0 and surface[:1].isupper():
        score += 0.55
    if any(character.isdigit() for character in surface):
        score += 0.40
    # Longer content words are more specific ("fractions" over "teach").
    # The cap sits just under the proper-noun bonus so no amount of length
    # lets a common word outrank a name.
    score += min(len(surface) / 12.0, 0.50)
    return score


def _stable_softmax(scores: dict[str, float], *, temperature: float = 1.0) -> dict[str, float]:
    if not scores:
        return {}
    temp = max(0.05, min(5.0, float(temperature or 1.0)))
    finite_scores = {
        key: (_safe_float(value, 0.0) / temp)
        for key, value in scores.items()
    }
    highest = max(finite_scores.values())
    exps = {
        key: math.exp(max(-60.0, min(60.0, value - highest)))
        for key, value in finite_scores.items()
    }
    total = sum(exps.values()) or 1.0
    return {key: value / total for key, value in exps.items()}


def _entropy01(probabilities: dict[str, float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    entropy = 0.0
    for probability in probabilities.values():
        p = max(1e-12, min(1.0, float(probability or 0.0)))
        entropy -= p * math.log(p)
    return _clamp(entropy / math.log(len(probabilities)))


def _top_memory_fragments(state: Any, *, limit: int = 3) -> list[str]:
    fragments: list[str] = []
    try:
        memory = list(getattr(getattr(state, "cognition", None), "working_memory", []) or [])
    except (AttributeError, TypeError):
        return fragments
    for item in reversed(memory):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "") or "").lower()
        if role not in {"user", "assistant", "thought"}:
            continue
        content = _normalize_text(item.get("content"), 140)
        if content:
            fragments.append(content)
        if len(fragments) >= limit:
            break
    return list(reversed(fragments))
