"""Deciding what evidence a turn needs, by meaning rather than by phrasing.

Bryan, live 2026-08-04: "a lot of these requests are tied into specific
phrases and that shouldn't be the case. part of her reasoning has to include
general associations and a general understanding of what is being asked."

He was right, and he had the evidence. "Which file in your repository does
that function live in?" reached her source and was answered correctly.
"What python module is that from" — the same question — did not, because it
missed a regex. A keyword gate that misses does not merely mis-route: it
leaves her blind to something she can actually see, and then the phrasing
IS the behaviour.

So relevance is measured, not matched. Each kind of evidence is described by
a few sentences that say what the KIND OF QUESTION is about, the request is
embedded in the same space (MiniLM, local, already resident), and the
decision is a cosine distance. Paraphrases she has never seen land near the
concept; a question about arithmetic does not.

The anchors are concept descriptions, not triggers — nothing here fires
because a particular word appeared. The distinction that matters for Bryan's
objection: adding a new way to ask "show me your code" requires no change
here, because the sentence does not have to resemble any of these, only to
MEAN something similar.

Lexical patterns remain as a floor, and only as a floor: when the embedding
model is unavailable or has fallen back to hashing — where cosine distance
carries no meaning — a missed gate would put her back to answering from
weights. Degrading to the old behaviour is acceptable; degrading to blindness
is not.
"""
from __future__ import annotations

import logging
import math
import threading
from typing import Any, Callable, Sequence

logger = logging.getLogger("Aura.Cognition.EvidenceRelevance")

__all__ = [
    "SCREEN_PERCEPTION",
    "OWN_SOURCE",
    "relevance",
    "wants_evidence",
    "semantic_routing_available",
]

#: Questions about what is on the person's screen, in several unrelated
#: phrasings. These describe the CONCEPT; the match is by meaning.
SCREEN_PERCEPTION = "screen_perception"

#: Questions about Aura's own implementation — her files, her functions,
#: where a piece of her code lives.
OWN_SOURCE = "own_source"

_ANCHORS: dict[str, tuple[str, ...]] = {
    SCREEN_PERCEPTION: (
        "what is currently displayed on the computer screen",
        "describe the windows and applications that are open right now",
        "what can you see in front of you on the monitor",
        "is a particular thing visible somewhere on the display",
        "what was on the screen a moment ago",
        "what is hidden behind or underneath the front window",
        "what are you looking at at this moment",
    ),
    OWN_SOURCE: (
        "show me a piece of your own program code",
        "which file of your implementation does that function live in",
        "what module or path in your repository contains that code",
        "where in your codebase is that written",
        "let me see the actual source you are built from",
    ),
}

#: Sentences that are emphatically NOT about the above, used as a contrast
#: set. A question is routed to evidence only when it is closer to the
#: concept than to ordinary conversation — an absolute threshold alone drifts
#: with how verbose the person happens to be.
_BASELINE_ANCHORS: tuple[str, ...] = (
    "what is seventeen multiplied by four",
    "how are you feeling today",
    "tell me a joke about penguins",
    "write a python function that sorts a list",
    "what did we decide about the schedule",
    "explain how photosynthesis works",
    # Asking her to WRITE ABOUT a subject is a different act from asking her
    # to LOOK at something or to open a file. Without this contrast, "give me
    # two concise sentences about reliable desktop tool use" scored as a
    # question about the desktop and pulled a screen reading into a request
    # for prose.
    "write a couple of concise sentences about a general principle",
    "summarise a topic briefly in your own words",
    # Asking what she CAN DO is not asking to see what she is made of.
    # Live: "what external tools could you use from the live desktop
    # path" scored as a question about her source and had a capability
    # answer replaced with a file excerpt.
    "what tools and capabilities do you have available to use",
    "describe what you are able to do for me",
)

#: How much closer to the concept than to ordinary talk a request must be.
#: Calibrated against the live 2026-08-04 transcript: every phrasing Bryan
#: actually used, plus the unrelated turns from the same conversation that
#: must NOT pull evidence in.
_MARGIN = 0.12

#: How far behind the best-matching concept a kind may fall and still be
#: considered part of what was asked.
_DOMINANCE = 0.20

_LOCK = threading.Lock()
_ANCHOR_CACHE: dict[str, Any] = {}
_REQUEST_CACHE: dict[str, Any] = {}
#: Bounded: this is a per-turn lookup, not a store.
_REQUEST_CACHE_MAX = 256


def _embedder() -> Any | None:
    """The live embedding engine, or None when there is not one."""
    try:
        from core.container import get_container

        memory = get_container().get("vector_memory_engine", default=None)
        embedder = getattr(memory, "embedder", None)
        if embedder is not None:
            return embedder
    except (ImportError, AttributeError, RuntimeError, LookupError):
        pass
    try:
        from core.memory.vector_memory_engine import EmbeddingEngine

        with _LOCK:
            engine = _ANCHOR_CACHE.get("__engine__")
            if engine is None:
                engine = EmbeddingEngine()
                _ANCHOR_CACHE["__engine__"] = engine
            return engine
    except (ImportError, AttributeError, RuntimeError):
        return None


def semantic_routing_available() -> bool:
    """Whether cosine distance means anything here right now.

    The engine falls back to a character-hash embedding when
    sentence-transformers is missing. Hash vectors are stable and
    meaningless: two paraphrases are as far apart as two unrelated
    sentences. Routing on them would be worse than the lexical floor,
    because it would look like it was working.
    """
    embedder = _embedder()
    if embedder is None:
        return False
    try:
        embedder._checkout_model()  # noqa: SLF001 - availability probe
    except (AttributeError, RuntimeError, OSError):
        return False
    try:
        model = getattr(embedder, "_model", None)
    finally:
        try:
            embedder._return_model()  # noqa: SLF001
        except (AttributeError, RuntimeError):
            pass
    return model is not None


def _embed(text: str) -> Any | None:
    embedder = _embedder()
    if embedder is None:
        return None
    try:
        return embedder.embed(str(text or ""))
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        logger.debug("Evidence relevance embedding failed: %s", exc)
        return None


def _cosine(left: Any, right: Any) -> float:
    try:
        dot = float(sum(float(a) * float(b) for a, b in zip(left, right)))
        left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
        right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    except (TypeError, ValueError):
        return 0.0
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _anchor_vectors(key: str, sentences: Sequence[str]) -> list[Any]:
    with _LOCK:
        cached = _ANCHOR_CACHE.get(key)
    if cached is not None:
        return cached
    vectors = [vector for vector in (_embed(text) for text in sentences) if vector is not None]
    with _LOCK:
        _ANCHOR_CACHE[key] = vectors
    return vectors


def _request_vector(request: str) -> Any | None:
    with _LOCK:
        cached = _REQUEST_CACHE.get(request)
    if cached is not None:
        return cached
    vector = _embed(request)
    if vector is None:
        return None
    with _LOCK:
        if len(_REQUEST_CACHE) >= _REQUEST_CACHE_MAX:
            _REQUEST_CACHE.clear()
        _REQUEST_CACHE[request] = vector
    return vector


def relevance(request: Any, kind: str) -> float:
    """How much closer this request is to ``kind`` than to ordinary talk.

    Positive means the concept fits better than small talk does. Returns
    0.0 when nothing can be measured, so a caller falls through to its
    floor rather than acting on a number that means nothing.
    """
    text = " ".join(str(request or "").split())
    if not text or kind not in _ANCHORS:
        return 0.0
    vector = _request_vector(text)
    if vector is None:
        return 0.0
    concept = _anchor_vectors(kind, _ANCHORS[kind])
    baseline = _anchor_vectors("__baseline__", _BASELINE_ANCHORS)
    if not concept or not baseline:
        return 0.0
    best_concept = max(_cosine(vector, anchor) for anchor in concept)
    best_baseline = max(_cosine(vector, anchor) for anchor in baseline)
    return best_concept - best_baseline


def wants_evidence(
    request: Any,
    kind: str,
    *,
    lexical_floor: Callable[[str], bool] | None = None,
    margin: float = _MARGIN,
) -> bool:
    """Whether this turn should be given ``kind`` evidence.

    Meaning decides. ``lexical_floor`` is consulted as a floor — it can add
    a turn the embedding missed, and it is the whole decision when semantic
    routing is unavailable — but it can never veto one the meaning found.
    """
    text = " ".join(str(request or "").split())
    if not text:
        return False

    floor = False
    if lexical_floor is not None:
        try:
            floor = bool(lexical_floor(text))
        except (RuntimeError, TypeError, ValueError):
            floor = False
    if floor:
        return True
    if not semantic_routing_available():
        return False
    score = relevance(text, kind)
    if score < margin:
        return False
    # Competitive, not independent. "What's on my screen?" scores +0.59
    # against perception and +0.09 against her source — both above an
    # absolute floor, and only one of them is the question. A kind that is
    # plainly beaten by another kind is not what was asked about, and
    # attaching both would put a file listing in front of a question about
    # a window. A turn that genuinely concerns two things scores near-level
    # on both and keeps them.
    best = max(relevance(text, other) for other in _ANCHORS)
    return score >= best - _DOMINANCE
