"""Turn something Aura read into something Aura understood.

LIVE DEFECT, 2026-08-03. Aura browsed /r/philosophy, and what landed in memory
was the page, verbatim, with the navigation chrome still attached:

    Reddit read r/philosophy: Western philosophy has been at war with The
    Odyssey for 2,800 years -- and keeps losing. ... : r/philosophy Skip to
    main content ... Go to Reddit Answers ... | logged | stored_via_manager

``action="logged"``, ``outcome="stored_via_manager"``. The event that she read
something was recorded. What she made of it was not recorded, because nothing
ever asked. So a minute later the reading could not tell her anything: not
whether the claim was new, not whether it agreed with what she already
believed, not whether the source was worth believing at all — and the reply
she gave about it had drifted to an unrelated subject entirely.

Reading is not an event to log. It is an encounter with a claim, and the
things worth keeping are:

* **the claim** — what the source actually asserts, in her words rather than
  its markup;
* **her stance** — does this affirm, contradict, extend, or merely repeat
  something she already holds? Affirmation is not nothing: a belief that has
  survived contact with an independent source is stronger than one that has
  not;
* **the source's quality** — a forum post, an unsourced assertion, a hostile
  headline and a peer-reviewed result are not equal evidence, and being able
  to say "this is a bad argument" IS a thing learned;
* **what it touches** — the beliefs and topics it connects to, so it can be
  found again by something other than a keyword.

This module produces that record. It is deliberately honest about its own
limits: every field it cannot establish is left empty rather than guessed, and
``stance`` is ``unassessed`` until something actually assesses it.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger("Aura.Knowledge.SourceComprehension")

SCHEMA = "aura.knowledge.source_comprehension.v1"

#: How much of a source's text is worth keeping as evidence for a claim.
_MAX_CLAIM_CHARS = 400
_MAX_EVIDENCE_CHARS = 700

#: Site furniture that is not content. A record that keeps these has kept the
#: page rather than what the page said.
_CHROME_PATTERNS = (
    # Bounded on purpose. An unbounded ".*" here consumed the whole page from
    # the first navigation phrase onward, so stripping the chrome stripped the
    # article with it.
    re.compile(r"(?i)\bskip to main content\b"),
    re.compile(r"(?i)\bgo to reddit answers\b"),
    re.compile(r"(?i)\b(?:log ?in|sign ?up|subscribe|accept all cookies)\b"),
    re.compile(r"(?i)\bopen menu\b|\bclose menu\b|\bexpand user menu\b"),
    re.compile(r"(?i)^\s*(?:home|popular|all|topics|resources)\s*$", re.MULTILINE),
    re.compile(r"(?i)\br/\w+\s*:\s*r/\w+"),
)

#: What kind of source this is. Ordered: the first match wins, so the more
#: specific hosts come before the generic shapes.
_SOURCE_KINDS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "peer_reviewed",
        ("arxiv.org", "doi.org", "pubmed", "nature.com", "science.org", "acm.org", "ieee.org"),
        "A paper. Method and data are inspectable, which is the strongest "
        "thing a source can offer — it is not the same as being right.",
    ),
    (
        "reference",
        ("wikipedia.org", "britannica.com", "stanford.edu/entries", "plato.stanford.edu"),
        "A reference work. Good for orientation and for names and dates; its "
        "claims are summaries of other people's work, not evidence itself.",
    ),
    (
        "documentation",
        ("docs.", "readthedocs", "developer.mozilla.org", "github.com"),
        "Documentation or source. Authoritative about what a thing DOES, "
        "silent about whether it is a good idea.",
    ),
    (
        "forum",
        ("reddit.com", "news.ycombinator.com", "stackexchange", "stackoverflow", "quora"),
        "A forum post. This is somebody's opinion with a vote count attached. "
        "Votes measure agreement, not correctness, and the comments are often "
        "worth more than the post.",
    ),
    (
        "news",
        ("nytimes", "bbc.", "guardian", "reuters", "apnews", "wsj.", "cnn."),
        "A news article. Reliable about what happened, weaker about why, and "
        "the headline is usually written by someone other than the reporter.",
    ),
    (
        "social",
        ("twitter.com", "x.com", "facebook.com", "instagram.com", "tiktok.com"),
        "A social post. Optimised for reach, not for being checkable.",
    ),
)

#: Sentence shapes that carry a claim worth keeping, versus ones that are the
#: page talking about itself.
_CLAIM_HINT_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|does|do|can|cannot|will|should|must|"
    r"causes?|means?|shows?|proves?|argues?|claims?|suggests?|finds?)\b",
    re.IGNORECASE,
)

#: Rhetoric that should lower confidence in an argument regardless of topic.
_WEAK_ARGUMENT_MARKERS: tuple[tuple[str, str], ...] = (
    ("everyone knows", "appeals to consensus instead of evidence"),
    ("obviously", "asserts rather than argues"),
    ("it is well known", "appeals to consensus instead of evidence"),
    ("proves that", "claims proof, which almost nothing outside mathematics has"),
    ("always", "universal claim, which one counterexample defeats"),
    ("never", "universal claim, which one counterexample defeats"),
    ("literally", "usually intensifier, not a measurement"),
    ("destroys", "framing as a contest rather than an argument"),
    ("debunked", "framing as a contest rather than an argument"),
    ("keeps losing", "framing as a contest rather than an argument"),
)


@dataclass
class SourceComprehension:
    """What Aura made of something she read."""

    schema: str = SCHEMA
    url: str = ""
    title: str = ""
    source_kind: str = "unknown"
    source_caveat: str = ""
    claim: str = ""
    evidence_excerpt: str = ""
    #: affirms | contradicts | extends | repeats | unassessed
    stance: str = "unassessed"
    stance_basis: str = ""
    argument_weaknesses: list[str] = field(default_factory=list)
    related_beliefs: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    content_sha256: str = ""

    @property
    def understood(self) -> bool:
        """Whether anything was actually extracted. Empty is not understanding."""
        return bool(self.claim)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "url": self.url,
            "title": self.title,
            "source_kind": self.source_kind,
            "source_caveat": self.source_caveat,
            "claim": self.claim,
            "evidence_excerpt": self.evidence_excerpt,
            "stance": self.stance,
            "stance_basis": self.stance_basis,
            "argument_weaknesses": list(self.argument_weaknesses),
            "related_beliefs": list(self.related_beliefs),
            "topics": list(self.topics),
            "content_sha256": self.content_sha256,
        }

    def narrative(self) -> str:
        """What she would say she took from it, in one short paragraph."""
        if not self.understood:
            return "I opened it but couldn't get a claim out of it worth keeping."
        parts = [f"Claim: {self.claim}"]
        if self.source_caveat:
            parts.append(f"Source: {self.source_caveat}")
        if self.stance != "unassessed" and self.stance_basis:
            parts.append(f"Against what I hold: {self.stance_basis}")
        if self.argument_weaknesses:
            parts.append("Weak points: " + "; ".join(self.argument_weaknesses[:3]))
        return " ".join(parts)


def strip_site_chrome(text: str) -> str:
    """Remove navigation furniture so a claim is not stored with the menu."""
    cleaned = str(text or "")
    for pattern in _CHROME_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def classify_source(url: str) -> tuple[str, str]:
    """The kind of source, and what that kind is and is not good for."""
    text = str(url or "").strip().lower()
    if not text:
        return "unknown", ""
    try:
        host = (urlparse(text).hostname or text).lower()
    except (TypeError, ValueError):
        host = text
    haystack = f"{host}{urlparse(text).path if '://' in text else ''}"
    for kind, markers, caveat in _SOURCE_KINDS:
        if any(marker in haystack for marker in markers):
            return kind, caveat
    return "web_page", (
        "An ordinary web page. Nothing about it establishes who wrote it or "
        "whether anyone checked it."
    )


def extract_claim(text: str, *, title: str = "") -> tuple[str, str]:
    """The sentence a source is actually asserting, and the text around it.

    Prefers a titled claim, because a headline is usually the thesis; falls
    back to the first sentence that asserts something. Returns empty strings
    when nothing does — a page that makes no claim taught nothing, and saying
    so is better than storing its markup.
    """
    body = strip_site_chrome(text)
    heading = strip_site_chrome(title)
    if heading and _CLAIM_HINT_RE.search(heading):
        claim = heading[:_MAX_CLAIM_CHARS]
        return claim, body[:_MAX_EVIDENCE_CHARS]
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        candidate = sentence.strip()
        if len(candidate) < 25 or len(candidate) > _MAX_CLAIM_CHARS:
            continue
        if _CLAIM_HINT_RE.search(candidate):
            return candidate, body[:_MAX_EVIDENCE_CHARS]
    if heading:
        return heading[:_MAX_CLAIM_CHARS], body[:_MAX_EVIDENCE_CHARS]
    return "", ""


def argument_weaknesses(text: str) -> list[str]:
    """Rhetoric that should lower confidence, whatever the subject is.

    Being able to say "this is a bad argument" is itself something learned, so
    it is recorded rather than silently discounting the source.
    """
    lowered = str(text or "").lower()
    found: list[str] = []
    for marker, why in _WEAK_ARGUMENT_MARKERS:
        if marker in lowered and why not in found:
            found.append(why)
    return found


def assess_stance(
    claim: str,
    *,
    known_beliefs: list[str] | None = None,
) -> tuple[str, str, list[str]]:
    """Where this claim sits relative to what Aura already holds.

    Returns (stance, basis, related beliefs). ``unassessed`` when there is
    nothing to compare against — an unexamined claim is not a confirmed one,
    and pretending otherwise is how a source gets believed for no reason.
    """
    beliefs = [str(b or "").strip() for b in (known_beliefs or []) if str(b or "").strip()]
    if not claim or not beliefs:
        return "unassessed", "", []

    claim_terms = _content_terms(claim)
    if not claim_terms:
        return "unassessed", "", []

    related: list[str] = []
    best_overlap = 0
    for belief in beliefs:
        overlap = len(claim_terms & _content_terms(belief))
        if overlap >= 2:
            related.append(belief)
            best_overlap = max(best_overlap, overlap)

    if not related:
        return (
            "extends",
            "Nothing I hold speaks to this, so it is new ground rather than "
            "agreement or disagreement.",
            [],
        )
    negated_claim = _is_negated(claim)
    negated_belief = any(_is_negated(belief) for belief in related)
    if negated_claim != negated_belief:
        return (
            "contradicts",
            "This cuts against something I hold, so one of us is wrong and it "
            "is worth finding out which.",
            related[:5],
        )
    if best_overlap >= 4:
        return (
            "repeats",
            "I already held this; the source adds a voice but not evidence.",
            related[:5],
        )
    return (
        "affirms",
        "An independent source lands the same way I do, which makes the "
        "belief harder to dismiss than it was.",
        related[:5],
    )


_TERM_STOPWORDS = frozenset(
    {
        "about", "after", "again", "against", "because", "been", "being",
        "between", "could", "does", "from", "have", "into", "more", "most",
        "much", "other", "over", "should", "some", "such", "than", "that",
        "their", "them", "then", "there", "these", "they", "this", "those",
        "through", "under", "very", "were", "what", "when", "where", "which",
        "while", "with", "would", "your",
    }
)


def _content_terms(text: Any) -> set[str]:
    return {
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", str(text or ""))
        if word.lower() not in _TERM_STOPWORDS
    }


def _is_negated(text: Any) -> bool:
    return bool(
        re.search(
            r"\b(?:not|never|no|cannot|can't|isn't|aren't|wasn't|weren't|"
            r"doesn't|don't|fails?|false|wrong|refutes?|disproves?)\b",
            str(text or ""),
            re.IGNORECASE,
        )
    )


def comprehend_source(
    *,
    url: str = "",
    title: str = "",
    text: str = "",
    known_beliefs: list[str] | None = None,
) -> SourceComprehension:
    """Read a source into a record of what was understood from it.

    Never raises: a reading that cannot be comprehended returns a record whose
    ``understood`` is False, which is a truthful outcome and a storable one.
    """
    body = str(text or "")
    kind, caveat = classify_source(url)
    claim, evidence = extract_claim(body, title=title)
    stance, basis, related = assess_stance(claim, known_beliefs=known_beliefs)
    return SourceComprehension(
        url=str(url or "")[:400],
        title=strip_site_chrome(title)[:200],
        source_kind=kind,
        source_caveat=caveat,
        claim=claim,
        evidence_excerpt=evidence,
        stance=stance,
        stance_basis=basis,
        argument_weaknesses=argument_weaknesses(f"{title}\n{body}"),
        related_beliefs=related,
        topics=sorted(_content_terms(f"{title} {claim}"))[:12],
        content_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
    )


__all__ = [
    "SCHEMA",
    "SourceComprehension",
    "argument_weaknesses",
    "assess_stance",
    "classify_source",
    "comprehend_source",
    "extract_claim",
    "strip_site_chrome",
]
